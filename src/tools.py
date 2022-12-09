#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:54:49 2022

@author: gauss
"""
import numba
import numpy as np
import os

def write_tserie(iout, ii, dt, t, slip_0, vel_0, state_0, stress):

    with open(f'output_ot_{iout}', 'a') as file:

        file.write(
            '{}\t{:.12E}\t{:.12E}\t{:.12E}\t{:.12E}\t{:.12E}\t{:.12E}\n'
            .format(ii, dt, t, slip_0[iout], vel_0[iout], state_0[iout], stress[iout])
        )
        
def write_tseriemax(t, x, vel_max_index, vel_max, state_min, stress_max, r_vel, r_state):

    with open('output_vmax', 'a') as file:  # WRITE MAXIMUM VALUES TO THE FILE
        file.write(
            '{:.12E}\t{:.12E}\t{:.12E}\t{:.12E}\t{:.12E}\t{:.12E}\t{:.12E}\n'
            .format(t, x[vel_max_index], vel_max, state_min, stress_max, r_vel, r_state)
        )

@numba.njit
def shift(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result



@numba.jit(forceobj=True, parallel=True, nogil=True)
def dynamic_term(coeff_hist, kernel, D_d, D_dyn, dt_min, N_try, N_kernel):
    for iii in numba.prange(N_kernel):
        Nk = kernel[iii].size
        if Nk>N_try:
            coeff_hist[iii] = shift(coeff_hist[iii], N_try, D_d[iii])
        else:
            coeff_hist[iii][:]=D_d[iii]
        D_dyn[iii] = np.dot(coeff_hist[iii], kernel[iii]) * dt_min
    return (coeff_hist, D_dyn)

        
        
def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first 
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment


def reversed_fp_iter(fp, buf_size=8192):
    """a generator that returns the lines of a file in reverse order
    ref: https://stackoverflow.com/a/23646049/8776239
    """
    segment = None  # holds possible incomplete segment at the beginning of the buffer
    offset = 0
    fp.seek(0, os.SEEK_END)
    file_size = remaining_size = fp.tell()
    while remaining_size > 0:
        offset = min(file_size, offset + buf_size)
        fp.seek(file_size - offset)
        buffer = fp.read(min(remaining_size, buf_size))
        remaining_size -= buf_size
        lines = buffer.splitlines(True)
        # the first line of the buffer is probably not a complete line so
        # we'll save it and append it to the last line of the next buffer
        # we read
        if segment is not None:
            # if the previous chunk starts right from the beginning of line
            # do not concat the segment to the last line of new chunk
            # instead, yield the segment first
            if buffer[-1] == '\n':
                #print 'buffer ends with newline'
                yield segment
            else:
                lines[-1] += segment
                #print 'enlarged last line to >{}<, len {}'.format(lines[-1], len(lines))
        segment = lines[0]
        for index in range(len(lines) - 1, 0, -1):
            if len(lines[index]):
                yield lines[index]
    # Don't yield None if the file was empty
    if segment is not None:
        yield segment

