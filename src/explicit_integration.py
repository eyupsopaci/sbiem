#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:02:23 2022
Explicit method
@author: gauss
"""
import numba
import numpy as np
from friction import state_rate_fun
import pyfftw

@numba.njit
def f_transfer_fun(a,N_lam,N_kernel,KK):
    
    with numba.objmode(D='complex128[:]'):
        AA = pyfftw.empty_aligned(N_lam, dtype='complex128')
        AA[:] = a[:]
        fft_object = pyfftw.builders.fft(AA, N_kernel, axis=0)
        D = fft_object()
    
    F = np.multiply( KK, D )
    with numba.objmode(f_transfer='complex128[:]'):
        FF = pyfftw.empty_aligned(N_kernel, dtype='complex128')
        FF[:] = F[:]
        fft_object = pyfftw.builders.ifft(FF, axis=0)
        f_transfer = fft_object()
    f_transfer = f_transfer[:N_lam].real* N_lam/N_kernel
    return f_transfer
    

@numba.njit
def motion_eq(y, t, aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type):

    A = y[:N_lam] - vpl
    f_transfer = f_transfer_fun(A,N_lam,N_kernel,KK) 
    dydt = np.zeros_like(y)
    for i in numba.prange(N_lam):
        dydt[N_lam+i] = state_rate_fun(y[i], y[N_lam+i], dcc[i], state_type)
        dydt[i] = (f_transfer[i] - bb[i]*sigma_nn[i] * dydt[N_lam+i] / y[N_lam+i]) / \
            (0.5 * mu / c + (aa[i]*sigma_nn[i]) / y[i])
    return dydt


def wrapper_scipy(t, y, aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type):

    return motion_eq(y, t, aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type)



@numba.njit
def rk_routine(t, y, dt, dt_max, dt_min, aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type, tol, maxiter):
    
    # Runge-Kutta Fehlberg coefficients
    a2 = 2.500000000000000e-01  # 1/4
    a3 = 3.750000000000000e-01  # 3/8
    a4 = 9.230769230769231e-01  # 12/13
    a5 = 1.000000000000000e+00  # 1
    a6 = 5.000000000000000e-01  # 1/2

    b21 = 2.500000000000000e-01  # 1/4
    b31 = 9.375000000000000e-02  # 3/32
    b32 = 2.812500000000000e-01  # 9/32
    b41 = 8.793809740555303e-01  # 1932/2197
    b42 = -3.277196176604461e+00  # -7200/2197
    b43 = 3.320892125625853e+00  # 7296/2197
    b51 = 2.032407407407407e+00  # 439/216
    b52 = -8.000000000000000e+00  # -8
    b53 = 7.173489278752436e+00  # 3680/513
    b54 = -2.058966861598441e-01  # -845/4104
    b61 = -2.962962962962963e-01  # -8/27
    b62 = 2.000000000000000e+00  # 2
    b63 = -1.381676413255361e+00  # -3544/2565
    b64 = 4.529727095516569e-01  # 1859/4104
    b65 = -2.750000000000000e-01  # -11/40

    r1 = 2.777777777777778e-03  # 1/360
    r3 = -2.994152046783626e-02  # -128/4275
    r4 = -2.919989367357789e-02  # -2197/75240
    r5 = 2.000000000000000e-02  # 1/50
    r6 = 3.636363636363636e-02  # 2/55

    c1 = 1.157407407407407e-01  # 25/216
    c3 = 5.489278752436647e-01  # 1408/2565
    c4 = 5.353313840155945e-01  # 2197/4104
    c5 = -2.000000000000000e-01  # -1/5

    for ii in range(maxiter):

        k1 = dt * motion_eq(y, t, 
                           aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type)
        k2 = dt * motion_eq(y + b21 * k1, t + a2 * dt,
                           aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type)
        k3 = dt * motion_eq(y + b31 * k1 + b32 * k2, t + a3 * dt,
                           aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type)
        k4 = dt * motion_eq(y + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * dt,
                           aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type)
        k5 = dt * motion_eq(y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * dt,
                           aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type)
        k6 = dt * motion_eq(y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * dt,
                           aa, bb, dcc, sigma_nn, vpl, N_lam, N_kernel, KK, mu, c, state_type)
        
        r = max(
            np.abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / dt)

        dt = dt * min(max(0.9 * np.power(r / tol, -0.20), 0.1), 4.0)
        # N_try = int(dt_try/dt_min)
        # dt = max(dt_min, N_try*dt_min)
        if r <= tol:
            break
    t += dt
    y += c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5

    return t, dt, y, r

# %timeit rk_routine(t, y, dt, dt_max, dt_min, aa, bb, dcc, sigma_nn, vpl, N_lam, KK, mu, c, state_type, tol, maxiter)


