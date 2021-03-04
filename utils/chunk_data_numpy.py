#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
based on https://gist.github.com/mattjj/5213172
"""
from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


def chunk_data(data, window_size, overlap_size=0, flatten_inside_window=True, pad=False):
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (
        window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows * window_size -
                                (num_windows - 1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if pad==True:
        if overhang != 0:
            num_windows += 1
            newdata = np.zeros((num_windows * window_size -
                                (num_windows - 1) * overlap_size, data.shape[1]))
            newdata[:data.shape[0]] = data
            data = newdata

    sz = data.dtype.itemsize
    ret = ast(
        data,
        shape=(num_windows, window_size * data.shape[1]),
        strides=((window_size - overlap_size) * data.shape[1] * sz, sz))

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))
