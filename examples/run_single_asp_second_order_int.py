#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:03:05 2022

@author: This code runs single asperity model
"""
import sys
import os
sys.path.append(os.path.join(os.path.join(os.path.expanduser("~"), "sbiem_v0.0"), "src"))
import numba
import numpy as np
import time
from kernel import *
from friction import *
from equations import *
from second_order_integration import *
from tools import *
from multiprocessing import Process, Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explicit_integration import wrapper_scipy, motion_eq, f_transfer_fun
from scipy.integrate import ode

def run_simulations(Nasp, resolution, dim, sim, law, state_type, aminb_asp, aminb_bar, aminb_cor, b, dc, Lasp, Lbar, W, fpath):
    
    # Lengths 
    Lcor = Lbar
    lam = (Nasp) * Lasp + (Nasp-1) * Lbar + 2*Lcor 
    
    # Time conversion beetween seconds and years
    tyr = 365. * 24. * 3600.
    
    # Output intervals
    Ntout = 100 # Every "Ntout" time step write selected position to data
    Nxtout = 500    # Every "Nxtout" time step write the entire space domain to file
    Nxout = 16      # Every "Nxout" element is written to file  
    
    
    # friction parameters
    f_0 = 0.6 # reference friction
    aasp = aminb_asp + b    # direct velovity effect parameter at asperity
    acor = aminb_cor + b    # direct velovity effect parameter at border 
    sigma_n = 100.0E6
    
    abratio = aasp/b    
    beta = 0.25
    nw = 2.
    vpl = 20e-3/tyr
    v0 = 1.0e-6
    tol = 1.0e-6
    maxiter = 50
    
    mu = 30.0e9
    c = 3.0e3
    nu = 0.5*mu/c
    
    
    # Characteristic halflengths [m]
    Lb = mu*dc/sigma_n/b;
    Lnuc = 1.3774*Lb;
    Lc = Lb/(1-abratio);
    Linf = 1/np.pi *(1-abratio)**2 *Lb
    h_star_RR = np.pi/4 * mu * dc / (-aminb_asp * sigma_n)
    h_star_RA = 0.5*np.pi* mu * dc * b / ((-1*aminb_asp)**2 * sigma_n)
    
    N_lam = int(np.power(2, np.ceil(np.log2(resolution * lam / Lb))))
    N_kernel = 4 * N_lam # Frequency domain is 4 times larger than the space domain to avaoid replication 
    
    # space gridding and space domain
    dx = lam / N_lam
    x = np.arange(-N_lam/2,N_lam/2,1) * dx
    aa = np.ones(N_lam) * aasp
    bb = np.ones(N_lam) * b
    dcc = np.ones(N_lam) * dc
    sigma_nn = np.ones(N_lam)*sigma_n 
    

    aspm = (x <= Lasp / 2) & (x >= -Lasp / 2)
    cor = [not elem for elem in aspm]
    aa[cor] = bb[cor] + aminb_cor

    # The locations specificallly I choose to save
    indl = int( 0.75*(2*Lcor+Lasp) / lam * N_lam )
    indr = int( 0.25*(2*Lcor+Lasp) / lam * N_lam )
    iot = [ indl, N_lam//2, indr ]   
    
    
    # Define static kernel
    # kernel, thetas, tw, n, kn, dt_min = bessel_kernel(lam, W, N_kernel, c, beta, nw)
    # N_theta_max = thetas[N_kernel//2].shape[0]
    k = np.append(np.arange(0,N_kernel//2+1), np.arange(-N_kernel//2+1,0)) *2.0*np.pi/lam# wave number
    KK_dyn = np.zeros(N_kernel, dtype="float64")
    if dim == 0:
        KK = -0.5 * mu * np.abs(k)
    elif dim==1: 
        # If depth information is assigned
        KK = -0.5 * mu * np.sqrt(k*k + (2.0*np.pi/W)**2)

    # Integration parameters
    t, tf = 0.0, 500*tyr # start and stop time
    dt_min = beta * lam / N_lam / c    # Initial parameters
    dt = dt_min
    dt_max=1e6
    v_max=10
    vel_0 = np.ones(N_lam) * v0 * 0.9
    slip_0 = np.zeros(N_lam)
    tau_0 = sigma_n * acor * np.arcsinh(
        0.5*vel_0/v0 * np.exp((f_0 + b * np.log(v0/vel_0)) / acor))\
        + nu * vel_0
    theta_0 = dc/v0 * \
        np.exp(aa/b * np.log(2.0*v0/vel_0 *
                np.sinh((tau_0-nu*vel_0)/aa/sigma_n)) - f_0/b)
    theta_dot_0 = state_rate_fun(vel_0, theta_0, dc, state_type)
    
    # Prepare static and dynanmic kernels (D_dyn is used if sim=1, full inertial effects anre included)
    D_sta = np.fft.fftshift(np.fft.fft(slip_0, N_kernel))
    D_dyn = np.zeros_like(D_sta)
    
    coeff_hist = []
    for ii in range(N_kernel):
        coeff_hist.append(np.zeros(k[ii].size, dtype='complex128'))
    
    dname = str('Nasp{}_exp_res{}_sim{}_dim{}_state{}_law{}_aasp{:.4f}_dc{:.4f}_aminbasp{:.4f}_aminbbar{:.4f}_Lasp{:.0f}_Lbar{:.0f}_W{:.0f}_lam{:.0f}_sigma_n{:.0f}_Nele{}'
                    .format(
                        Nasp, resolution, sim, dim, state_type, law, aasp, dc, aminb_asp, aminb_bar, Lasp*1e-3, Lbar*1e-3, W*1E-3, lam*1e-3, sigma_n*1e-6,N_lam
                    )
                    )
    
    # Create folder to save results
    wdir = os.path.join(fpath, dname)
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    os.chdir(wdir)
    file = open("{}.inp".format(dname), "w") 
    file.write(" friction_law {}\n state_law {}\n Ntout {}\n Nxtout {}\n Nxout {}\n f0 {}\n mu {}\n sigma_n {}\n c {}\n W {}\n beta {}\n nw {}\n v0 {}\n vpl {}\n tol {}\n maxiter {}\n Lasp_n {}\n lam_n {}\n resolution {}\n Lb {}\n Lnuc {}\n Lc {}\n Linf {}\n hRR*/dx {}\n hRA*/dx {}\n"                
               .format(
                   law, state_type, Ntout, Nxtout, Nxout,f_0, mu, sigma_n, c, W, beta, nw, v0, vpl, tol, maxiter,Lasp, lam, resolution, Lb, Lnuc, Lc, Linf, h_star_RR/dx, h_star_RA/dx)
        )

    
    file.write("a \t b \t dc \t sigma_n \t tau_0 \t slip_0 \t state_0 \t vel_0 \t state_dot_0\n")
    for ii in range(N_lam):
        file.write("{:.8E} \t {:.8E} \t {:.8E} \t {:.8E} \t {:.8E} \t {:.8E} \t {:.8E} \t {:.8E} \t {:.8E}\n".format(
            aa[ii], bb[ii], dcc[ii], sigma_nn[ii], tau_0[ii], slip_0[ii], theta_0[ii], vel_0[ii], theta_dot_0[ii]))
    file.close()


    # PREPARE FILES THAT WILL BE WRITTEN DURING SIMULATION
    columns1 = list(map(lambda index: f'{index}', iot))
    columns2 = ['step', 'dt', 'time',
                'slip', 'rate', 'state', 'stress']
    dummy = '# t \t x \t slip \t rate \t theta \t tau\n'
    with open('output_vmax', 'w') as file:
        file.write(
            't \t x \t rate_max \t theta_max \t tau_max \t err_vel \t err_state\n')
    for fname in columns1:
        with open(f'output_ot_{fname}', 'w') as file:
            file.write('\t'.join(columns2)+"\n")
            
    ####################### SETUP PLOT ################################
    fig,ax = plt.subplots(4,1, sharex = True, clear=True)
    ax[0].plot(x * 1e-3, aa, label = "a"), ax[0].plot(x * 1e-3, np.zeros(N_lam), "k--", lw = 0.5)
    ax[0].plot(x * 1e-3, bb, color = "gray", ls = "-.", lw = 0.75, label = "b")
    ax[0].plot(x[iot] * 1e-3, aa[iot] * 3, color = "red", marker = "*", ls = "None")
    ax[0].set_yticks([aminb_asp, aminb_cor,b])
    # ax[0].set_ylabel("a-b")
    ax[0].legend(ncol = 2, loc=9)
    
    ax[1].semilogy(x * 1e-3, tau_0*1E-6)
    ax[1].set_ylabel("$\\tau_{ini}$ [MPa]")
    
    ax[2].semilogy(x * 1e-3, vel_0)
    ax[2].set_ylabel("$v_{ini}$ [m/s]")
    
    ax[3].semilogy(x * 1e-3, theta_0)
    ax[3].set_ylabel("$\\theta_{ini}$ [s]")
    ax[3].set_xlabel("position [km]")

    # ax[4].set_ylabel("$\\stress{ini}$ [s]")
    ax[3].set_xlabel("posiiton [km]")
    fig.savefig("setup.png", bbox_inches='tight',pad_inches=0.1)
   
    
   
    
    # INTEGRATION PARAMETERS
    print('dt[s] \t time[yr] \t rate[m/s] \t theta \t error')
    
    # RUNNING SIMULATION
    while t < tf:

        t, dt, N_try, slip_0, theta_0, vel_0, theta_dot_0, f_s, D_sta, D_d, r_vel, r_state, flag_22 = \
            stepping(t, dt, slip_0, theta_0, vel_0, theta_dot_0, D_sta, D_dyn, KK, KK_dyn, aa, bb, dcc, sigma_nn,
                     tau_0, f_0, vpl, v0, mu, c, dt_min, dt_max, tol, N_kernel, N_lam, maxiter, v_max, law, state_type, sim)
        
        # if sim==1:
        #     if N_try < N_theta_max:
        #         coeff_hist, D_dyn = dynamic_term(coeff_hist, kernel, D_d, D_dyn, dt_min, N_try, N_kernel)
        #     else:
        #         D_dyn = np.zeros_like(D_d)

        # stress = frictional_strength(
        #     vel_0, state_0, aa, bb, dc, sigma_n, v0, f_0,  law)
        ii += 1

        if ii % Ntout == 0:
            vel_max_index = np.argmax(vel_0)
            vel_max = vel_0[vel_max_index]
            theta_min = theta_0[vel_max_index]
            # state_max = np.max(state_0)
            stress_max = f_s[vel_max_index]

            print('{:.4E} \t {:.4E} \t {:.4E} \t {:.4E} \t {:.4E}'.format(
                dt, t/tyr, vel_max, theta_min, max(r_vel, r_state)))

            for iout in iot:
                write_tserie(int(iout), ii, dt, t, slip_0, vel_0, theta_0, f_s)
            # with Pool() as p:
            #     p.starmap(write_tserie, inputs)
            # write_tseriemax(t, x, vel_max_index, vel_max, state_min, stress_max, r_vel, r_state)


        if ii % Nxtout == 0:  # WRITING VALUES ALONG THE FAULT TO THE FILE
            with open('output_ox', 'a') as file:
                file.write(dummy)
                for iox in np.arange(0, N_lam, step=Nxout):
                    file.write(
                        '{:.15E}\t{:.12E}\t{:.12E}\t{:.12E}\t{:.12E}\t{:.12E}\n'
                        .format(t, x[iox], slip_0[iox], vel_0[iox], theta_0[iox], f_s[iox])
                    )
                    
                    

if __name__ == '__main__':
    from itertools import product
    resolution = 9 # Resolution coeffeicient to discretize domain adequately according to the cohesive zone
    W = 50E3 # Width of the fault (no discretization!). Apply only when dim=1
    Nasp = 1 #NUmber of asperities
    dim = 0 # dimesion of the fault 0=2D, 1=2.5D
    sim = 0 # 0 for quasi-dynamic and 1 for full-dynamic
    law = 0 #rsf law 0:original, 1:modified
    state_type = 0 # state law 0:Aging, 1:slip   
    Lasp = 30e3 # length of asperity
    Lbar = 15e3 # length of barrier 
    b = 0.02 # state effect parameter 
    aminb_asp = -0.01   #
    aminb_bar = 0.003
    aminb_cor = 0.005
    dc = 16e-3
    
    fpath = "/media/gauss/STORE1/" # the folder where results will be saved
    run_simulations(Nasp, resolution, dim, sim, law, state_type, aminb_asp, aminb_bar, aminb_cor, b, dc, Lasp, Lbar, W, fpath)
    
    # pars = product(Nasp, resolution, dim, sim, law, state_type, aminb_asp, aminb_bar, aminb_cor, b, dc, Lasp, Lbar, W,fpath)
    # pool=Pool()
    # res = pool.starmap_async(run_simulations, pars,)    # run_simulations()
    # res.get()
    # pool.close()
    # pool.join()
#    
