#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:43:23 2022

@author: gauss
"""
import numba
from friction import frictional_strength
import numpy as np


@numba.njit(cache=True)
def elastic_stress(vel, f, tau_0, vpl, mu, c):
    """
    

    Parameters
    ----------
    vel : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    tau_0 : TYPE
        DESCRIPTION.
    vpl : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return (
        tau_0 + f - 0.5 * mu / c * (vel-vpl)
    )



@numba.njit(cache=True, nogil=True)
def func(vel_0, state_0, f, aa, bb, dcc, sigma_nn, tau_0, v0, vpl, f_0, mu, c, law):
    strength = frictional_strength(
        vel_0, state_0, aa, bb, dcc,sigma_nn, v0, f_0, law)
    
    stress = elastic_stress(vel_0, f, tau_0, vpl, mu, c)
    return (stress - strength)



@numba.njit( cache=True)
def dfunc(vel_0, state_0, f, aa, bb, dcc, sigma_nn, tau_0, v0, vpl, f_0, mu, c, law):
    return (-sigma_nn * aa / vel_0 -0.5*mu/c)



