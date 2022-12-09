#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:24:04 2022
Friction
@author: gauss
"""
import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def frictional_strength(vel, state, a, b, dc, sigma_n, v0, f_0, law):
    """
    

    Parameters
    ----------
    vel : array
        slip rates.
    state : array
        state of the frictional surface (contact history).
    a : array
        direct velocity effect.
    b : array
        state evolution effect.
    dc : array
        critical slip distance.
    sigma_n : array
        effective normal stress.
    v0 : scalar
        reference velocity.
    f_0 : scalar
        friction constant.
    law : integer
        0 (original) , 1 (regularized) rate and state friction.

    Returns
    -------
    strength : array
        frictional resistance against the driving plate.

    """

    if law == 1:
        strength= (a * sigma_n *
                np.arcsinh(0.5 * vel / v0 *
                           np.exp((f_0 + b * np.log(v0 * state / dc)) / a))
                )
    elif law == 0:
        strength = (sigma_n *
                (f_0 + a * np.log(vel / v0) + b * np.log(v0 * state / dc))
                )
    return strength


@numba.jit(nopython=True, cache =True)
def state_rate_fun(vel, state, dc, state_type):
    """
    

    Parameters
    ----------
    vel : array
        slip rates.
    state : array
        state of the frictional surface (contact history).
        DESCRIPTION.
    dc : array
        direct velocity effect.
    state_type : integer
        0 (aging), 1  (ruina) state evolution formula.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    vel = np.abs(vel)
    omega = np.multiply(vel, state) / dc
    if state_type == 0:
        return 1.0 - omega  # aging law
    elif state_type == 1:
        return -omega * np.log(omega)  # slip law
    # elif .... another friction law

