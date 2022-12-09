#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:18:21 2022
Kernels
@author: gauss
"""


import numpy as np

import numba

# import timeit

# import matplotlib.pyplot as plt

@numba.jit(numba.float64[:](numba.float64[:]), parallel=True, nopython=True )
def Besseln(z):
    
    n = z.shape[0]
    fz = np.zeros(n)
    
    for i in numba.prange(n):
        if z[i] < 8.:
            t = z[i] / 8.
            fz[i] = (-0.2666949632 * t**14 + 1.7629415168000002 * t**12 + -5.6392305344 * t**10 + 11.1861160576 * t**8 + -14.1749644604 * t**6 + 10.6608917307 * t**4 + -3.9997296130249995 * t**2 + 0.49999791505)
        else:
            t = 8. / z[i]
            eta = z[i] - 0.75 * np.pi
            fz[i] = np.sqrt(2 / (np.pi * z[i])) * ((1.9776e-06 * t**6 + -3.48648e-05 * t**4 + 0.0018309904000000001 * t**2 + 1.00000000195) * np.cos(eta) - (-6.688e-07 * t**7 + 8.313600000000001e-06 * t**5 + -0.000200241 * t**3 + 0.04687499895 * t) * np.sin(eta)) / z[i]
        
    return fz

@numba.jit(numba.float64(numba.float64))
def Bessel1(z):
        
    if z < 8.:
        t = z/ 8.
        fz = z * (-0.2666949632 * t**14 + 1.7629415168000002 * t**12 + -5.6392305344 * t**10 + 11.1861160576 * t**8 + -14.1749644604 * t**6 + 10.6608917307 * t**4 + -3.9997296130249995 * t**2 + 0.49999791505)
    else:
        t = 8. / z
        eta = z - 0.75 * np.pi
        fz = np.sqrt(2 / (np.pi * z)) * ((1.9776e-06 * t**6 + -3.48648e-05 * t**4 + 0.0018309904000000001 * t**2 + 1.00000000195) * np.cos(eta) - (-6.688e-07 * t**7 + 8.313600000000001e-06 * t**5 + -0.000200241 * t**3 + 0.04687499895 * t) * np.sin(eta)) 
    return fz


@numba.jit
def time_kernel(lam, W, N, c, beta, nw):
    
    n = np.arange(-N/2, N/2, 1)
    
    aa = 2.0*np.pi*c/lam
    kn = np.append(np.arange(0,N//2+1), np.arange(-N//2+1,0)) *2.0*np.pi/lam# wave number
    kn[N//2] = 0
    # kn= 2.0 * np.pi * n / lam
    absn = np.abs(n)[:N//2]
    dtmin = beta * lam / N / c
    tw1 = nw *lam / c 
    # kw1 = aa*tw1
    # kwn2 = N/2.0*kw1
    
    # qw = N/2
    qw=4.
    # kw = kw1 + (kwn2-kw1) / (N/2-1.0) * (absn-1.0)
    kw = aa * tw1 * (1.0 + (qw-1.0)/(N/2-1.0) * (absn - 1.0) )
    tw = kw / (absn * aa )
    # tw[N//2] = 0
    return tw, n, kn, dtmin


@numba.jit
def bessel_kernel( lam,W, N, c, beta, nw):
    
    tw, n, kn, dtmin = time_kernel( lam, W, N, c, beta, nw )
    
    kernel = []
    
    thetas = []
    
    for jj in numba.prange(N//2):
        
        twj = tw[jj]
        
        kk = np.abs(kn[jj])

        theta = np.arange(dtmin, twj+dtmin, dtmin)
        
        Nt = theta.size
        
        I = np.zeros(Nt)

        for ii in numba.prange(Nt):
            
            q = theta[ii:] * kk * c
            
            # I[ii] = Besseln(q)/theta[ii]

            I[ii] = np.trapz( Besseln(q), q )
                    
        kernel.append(I)     
        
        thetas.append(theta)
    
    kernel.extend(kernel[::-1])
    
    thetas.extend(thetas[::-1])

            
    return kernel, thetas, tw, n, kn, dtmin

# # TEST

# N = np.power(2,9)
# c = 3e3
# lam = 10E3
# W= 50E3
# beta = 0.25
# nw = 2.
# # %timeit bessel_kernel( lam, N, c)

# kernel, thetas, tw, n, kk, dtmin = bessel_kernel(lam, W, N, c, beta = 0.25, nw = 2.)

# nplots = 8
# fig, ax = plt.subplots(nplots,1, figsize= (2,nplots*2))
# # ax.plot(thetas[N//2-10], kernel[N//2-10])
# for ii in range(nplots):
#     ax[ii].plot(thetas[ii * N//nplots], kernel[ii * N//nplots], label = f'{ii}')
    
# plt.tight_layout()

# fig.savefig("sondeneme.png")
# # ax.legend()
