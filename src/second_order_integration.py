#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:48:23 2022
Second order Integration
@author: gauss
"""
import numba 
import numpy as np
from equations import func, dfunc
from friction import state_rate_fun
import pyfftw


@numba.njit(cache=True, nogil=True)
def newtons_search(vel_0, state_0, f, aa, bb, dcc, sigma_nn, tau_0, v0, vpl, f_0, mu, c, maxiter, v_max, tol, law):
    flag = 0
    tol*=v_max
    for ii in range(maxiter):
        vel_s = vel_0 - func(vel_0, state_0, f, aa, bb, dcc, sigma_nn, tau_0, v0, vpl, f_0, mu, c, law)\
            / dfunc(vel_0, state_0, f, aa, bb, dcc, sigma_nn, tau_0, v0, vpl, f_0, mu, c, law)
        errmax = np.max(np.abs(vel_s-vel_0))
        if errmax < tol:
            flag = 1
            break
        else:
            vel_0 = vel_s
    return vel_s, flag


@numba.njit(cache=True, nogil=True)
def one_step_process(t, dt, slip_0, state_0, vel_0, state_dot_0, dt_min, D_sta, D_dyn, KK, KK_dyn, aa, bb, dcc, sigma_nn, tau_0, f_0, vpl, v0, mu, c, tol, N_kernel, N_lam, maxiter, v_max, law, state_type, sim):
    # Nc = N_lam/N_kernel
    # STEP 1 : Making first prediction of slip and state
    state_s = state_0 + dt * state_dot_0

    # STEP 2 : stress strength functional relation and fourier coefficients of the slip rate and slip
    with numba.objmode(D_dot='complex128[:]'):
        a = pyfftw.empty_aligned(N_lam, dtype='complex128', n=16)
        a[:] = vel_0 - vpl
        fft_object = pyfftw.builders.fft(a, N_kernel, axis=0)
        D_dot = fft_object()        
    
    D_s = D_sta + dt * D_dot
    
    if sim == 1:
        F_s =  np.multiply(KK, (D_s - D_dyn - (D_dot + np.multiply(KK_dyn,D_dot))*dt_min))
    elif sim == 0:
        F_s = np.multiply(KK, D_s)

    with numba.objmode(f_s='complex128[:]'):
        FF = pyfftw.empty_aligned(N_kernel, dtype='complex128')
        FF[:] = F_s[:]
        fft_object = pyfftw.builders.ifft(FF, axis=0)
        f_s = fft_object()
    f_s = f_s[:N_lam].real* N_lam/N_kernel

    # STEP 3 : predict slip velocity by using newton-raphson algorithm and predict the first state rate
    vel_s, flag =  newtons_search(vel_0, state_s, f_s[:N_lam], aa, bb, dcc, sigma_nn, tau_0, v0, vpl, f_0, mu, c, maxiter, v_max, tol, law)
    # if np.isnan(np.sum(vel_0)):
        
    state_dot_s = state_rate_fun(vel_s, state_s, dcc, state_type)
    # vel_s = np.abs(vel_s)
    
    # STEP 4 : Calculate final predicted values of slip and state
    vel_s0 = 0.5*(vel_s+vel_0)
    vel_s0_delta = vel_s0-vpl
    slip_ss = slip_0 + dt * vel_s0
    state_ss = state_0 + 0.5*dt*(state_dot_0+state_dot_s)

    # STEP 5 : (Analoguous with step 2 but with updated sstate_rate_fun( time, state, vel, dc, state_type)lip rate (vel_0+vel_s)/2)
    with numba.objmode(D_dot_s='complex128[:]'):
        a = pyfftw.empty_aligned(N_lam, dtype='complex128', n=16)
        a[:] = vel_s0_delta[:]
        fft_object = pyfftw.builders.fft(a, N_kernel, axis=0)
        D_dot_s = fft_object() 
        
    D_dot_ss = 0.5*(D_dot + D_dot_s)
    D_ss = D_sta + dt * D_dot_s
    
    if sim == 1:
        F_ss =  np.multiply(KK, (D_ss - D_dyn - (D_dot_s + np.multiply(KK_dyn,D_dot_s))*dt_min))
    elif sim ==0:
        F_ss = np.multiply(KK, D_ss )
    
    with numba.objmode(f_ss='complex128[:]'):
        FF = pyfftw.empty_aligned(N_kernel, dtype='complex128')
        FF[:] = F_ss[:]
        fft_object = pyfftw.builders.ifft(FF, axis=0)
        f_ss = fft_object()
    f_ss = f_ss[:N_lam].real* N_lam/N_kernel
    
    # STEP 6: predict the final slip velocity anologuous to step 3 using predicted values
    vel_ss, flag =  newtons_search(vel_0, state_ss, f_ss[:N_lam], aa, bb, dcc, sigma_nn, tau_0, v0, vpl, f_0, mu, c, maxiter, v_max, tol, law)
    # vel_ss = np.abs(vel_ss)

    state_dot_ss = state_rate_fun(vel_ss, state_ss, dcc, state_type)
    t += dt
    
    # queue.put(t, slip_ss, state_ss, vel_ss, state_dot_ss, D_ss, D_dot_ss, flag)
    return t, slip_ss, state_ss, vel_ss, state_dot_ss, f_ss, D_ss, D_dot_ss, flag


@numba.njit(cache=True, nogil=True)
def two_steps_process(t, dt, slip_0, state_0, vel_0, state_dot_0, dt_min, D_sta, D_dyn, KK, KK_dyn, aa, bb, dcc, sigma_nn, tau_0, f_0, vpl, v0, mu, c, tol, N_kernel ,N_lam, maxiter, v_max, law, state_type, sim):

    # first half step
    t12, slip_12, state_12, vel_12, state_dot_12, f_ss_12, D_12, D_dot_12, flag12 =\
        one_step_process(
            t, 0.5*dt, slip_0, state_0, vel_0, state_dot_0, dt_min, D_sta, D_dyn, KK, KK_dyn, aa, bb, dcc, sigma_nn, tau_0, f_0, vpl, v0, mu, c, tol, N_kernel, N_lam, maxiter, v_max, law, state_type, sim
        )

    # second half step
    t22, slip_22, state_22, vel_22, state_dot_22, f_ss_22, D_22, D_dot_22, flag22 =\
        one_step_process(
            t12, 0.5*dt, slip_12, state_12, vel_12, state_dot_12, dt_min, D_12, D_dyn, KK, KK_dyn, aa, bb, dcc, sigma_nn, tau_0, f_0, vpl, v0, mu, c, tol, N_kernel, N_lam, maxiter, v_max, law, state_type, sim
        )
    return t22, slip_22, state_22, vel_22, state_dot_22, f_ss_22, D_22, D_dot_22, flag22

@numba.njit(cache=True, nogil=True)
def stepping(t, dt, slip_0, state_0, vel_0, state_dot_0, D_sta, D_dyn, KK, KK_dyn, aa, bb, dcc, sigma_nn, tau_0, f_0, vpl, v0, mu, c, dt_min, dt_max, tol, N_kernel, N_lam, maxiter, v_max, law, state_type, sim):
    grw = (-1.0/3.0)
    N_try = 1
    
    inputs = (t, dt, slip_0, state_0, vel_0, state_dot_0, dt_min, D_sta, D_dyn, KK, KK_dyn, aa, bb, dcc, sigma_nn, tau_0, f_0, vpl, v0, mu, c, tol, N_kernel, N_lam, maxiter, v_max, law, state_type, sim)

    for _ in range(maxiter):
        
        # It seems that multiprocessing.Pool is slower than numba serial process!!!!
        # pool = mp.Pool(processes=2)
        # p1 = pool.apply_async(one_step_process, inputs)
        # p2 = pool.apply_async(two_steps_process, inputs)
        # pool.close()
        # pool.join()
        # t1, slip_1, state_1, vel_1, state_dot_1, Ds_1, Dd_1, flag_1 = p1.get()
        # t22, slip_22, state_22, vel_22, state_dot_22, Ds_22, Dd_22, flag_22 = p2.get()
        
        t1, slip_1, state_1, vel_1, state_dot_1, fs_1, Ds_1, Dd_1, flag_1 = one_step_process(*inputs)
        t22, slip_22, state_22, vel_22, state_dot_22, fs_2, Ds_22, Dd_22, flag_22 = two_steps_process(*inputs)

        # state_flag = 1
        r_state = 0.0
        r_vel = 0.0
        r_max = 0.0


        for i, (d1,d2, v1, v2, sd1, sd2, s1, s2) in enumerate(zip(slip_1, slip_22, vel_1, vel_22, state_dot_1, state_dot_22, state_1, state_22)):
            if s1 < 0.0 or s2 < 0.0 or np.isnan(v1) or np.isnan(v2):
                # state_flag = 0
                dt *= 0.5
                if dt < dt_min:
                    raise ValueError(
                        'The integration can not continue!' 
                        'Use smaller grid size or error tolerance.'
                        )
                break
            else:
                r_state = max(r_state, abs(s1-s2))
                r_max = max(r_max, abs(s1-s2)/abs(s2+dt*sd2))
                r_vel = max(r_vel, abs(v1-v2))
                
        # r_max = max(r_vel/v_max,r_state)
        dt_try = dt * min(max(0.9 * np.power(r_max/tol,grw), 0.1), 4.0)
        # dt_try = dt * 0.9*np.power( tol / (r+1e-40),grw)
        N_try = int(dt_try/dt_min)
        dt = max(dt_min, N_try*dt_min)
        if r_max <= tol or dt<= dt_min:
            break

    return t22, dt, int(N_try), slip_22, state_22, vel_22, state_dot_22, fs_2, Ds_22, Dd_22, r_vel, r_state, flag_22
# %timeit stepping(t, dt, slip_0, state_0, vel_0, state_dot_0, D_sta, D_dyn, KK, KK_dyn, aa, bb, dcc, sigma_nn, tau_0, f_0, vpl, v0, mu, c, dt_min, dt_max, tol, N_kernel, N_lam, maxiter, v_max, law, state_type, sim)

@numba.njit(cache=True)
def stepping_lapusta_etal2000(vel, dt, epsi, dt_min, dcc, dt_max, N_lam):
    
    dt_ev = dt_max # initial value that something large
    for i in numba.prange(N_lam):

        dt_ev = min(epsi[i]*dcc[i]/vel[i],dt_ev)
   
    # dt_ev = min(max(dt_ev/dt, 0.1), 4.0)
    nuew = int(dt_ev/dt_min)
    dt_ev_hat = max(dt_min, nuew*dt_min)
    if nuew<1:
        nuew=1
    return dt_ev_hat, nuew


@numba.njit(cache=True)
def par_for_stepping(aa, bb, dcc, sigma_nn, mu, gam, dx):
    N=aa.size
    k = gam*mu/dx
    epsi = np.zeros_like(aa)
    for i in numba.prange(N):
        dumm = k*dcc[i]/(aa[i]*sigma_nn[i])
        xi = 0.25*( dumm - (bb[i]-aa[i])/aa[i])**2 - dumm
        if xi>0.0:
            epsi[i] = min(aa[i]*sigma_nn[i]/(k*dcc[i]-sigma_nn[i]*(bb[i]-aa[i])), 0.5)
        else:
            epsi[i] = min( 1.0-sigma_nn[i]*(bb[i]-aa[i])/(k*dcc[i]), 0.5)
    return epsi

@numba.njit(cache=True, nogil=True)
def one_step_process_lapusta(t, dt, slip_0, state_0, vel_0, state_dot_0, dt_min, D_sta, D_dyn, KK, KK_dyn, aa, bb, dcc, sigma_nn, tau_0, epsi, f_0, vpl, v0, mu, c, tol, N_kernel, N_lam, maxiter, v_max, law, state_type, sim):
    # Nc = N_lam/N_kernel
    dt, N_try = stepping_lapusta_etal2000(vel_0, dt, epsi, dt_min, dcc, 5e6, N_lam)

    # STEP 1 : Making first prediction of slip and state
    vel_0_delta = vel_0 - vpl
    state_s = state_0 + dt * state_dot_0

    # STEP 2 : stress strength functional relation and fourier coefficients of the slip rate and slip
    with numba.objmode(D_dot='complex128[:]'):
        D_dot = pyfftw.interfaces.scipy_fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft(vel_0_delta, N_kernel))
        # D_dot = np.fft.fftshift(np.fft.fft(vel_0_delta, N_kernel))
    D_s = D_sta + dt * D_dot
    
    if sim == 1:
        F_s =  np.multiply(KK, (D_s - D_dyn - (D_dot + np.multiply(KK_dyn,D_dot))*dt_min))
    elif sim == 0:
        F_s = np.multiply(KK, D_s)

    with numba.objmode(f_s='float64[:]'):
        f_s = pyfftw.interfaces.scipy_fftpack.ifft(pyfftw.interfaces.scipy_fftpack.fftshift(F_s)).real
        # f_s = np.fft.ifft(np.fft.fftshift(F_s)).real*Nc

    # STEP 3 : predict slip velocity by using newton-raphson algorithm and predict the first state rate
    vel_s, flag =  newtons_search(vel_0, state_s, f_s[:N_lam], aa, bb, dcc, sigma_nn, tau_0, v0, vpl, f_0, mu, c, maxiter, v_max, tol, law)
    state_dot_s = state_rate_fun(vel_s, state_s, dcc, state_type)

    # STEP 4 : Calculate final predicted values of slip and state
    vel_s0 = 0.5*(vel_s+vel_0)
    vel_s0_delta = vel_s0-vpl
    slip_ss = slip_0 + dt * vel_s0
    state_ss = state_0 + 0.5*dt*(state_dot_0+state_dot_s)

    # STEP 5 : (Analoguous with step 2 but with updated sstate_rate_fun( time, state, vel, dc, state_type)lip rate (vel_0+vel_s)/2)
    with numba.objmode(D_dot_s='complex128[:]'):
        D_dot_s = pyfftw.interfaces.scipy_fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft(vel_s0_delta, N_kernel))
        # D_dot_s = np.fft.fftshift(np.fft.fft(vel_s0_delta, N_kernel))
    D_dot_ss = 0.5*(D_dot + D_dot_s)
    D_ss = D_sta + dt * D_dot_s
    
    if sim == 1:
        F_ss =  np.multiply(KK, (D_ss - D_dyn - (D_dot_s + np.multiply(KK_dyn,D_dot_s))*dt_min))
    elif sim ==0:
        F_ss = np.multiply(KK, D_ss )
        
    # F_ss[N//2]=0.
    with numba.objmode(f_ss='float64[:]'):
        f_ss = pyfftw.interfaces.scipy_fftpack.ifft(pyfftw.interfaces.scipy_fftpack.fftshift(F_ss)).real    
        # f_ss = np.fft.ifft(np.fft.fftshift(F_ss)).real*Nc

    # STEP 6: predict the final slip velocity anologuous to step 3 using predicted values
    vel_ss, flag =  newtons_search(vel_0, state_ss, f_ss[:N_lam], aa, bb, dcc, sigma_nn, tau_0, v0, vpl, f_0, mu, c, maxiter, v_max, tol, law)
    state_dot_ss = state_rate_fun(vel_ss, state_ss, dcc, state_type)
    t += dt
    
    return t, dt, N_try, slip_ss, state_ss, vel_ss, state_dot_ss, D_ss, D_dot_ss, 0, 0, 1
     