"""
This file is part of the accompanying code to our papers
Jiang, S., Zheng, Y., Wang, C., & Babovic, V. (2022) Uncovering flooding mecha-
nisms across the contiguous United States through interpretive deep learning on
representative catchments. Water Resources Research, 57, e2021WR030185.
https://doi.org/10.1029/2021WR030185.

Jiang, S., Bevacqua, E., & Zscheischler, J. (2022) River flooding mechanisms 
and their changes in Europe revealed by explainable machine learning, Hydrology 
and Earth System Sciences, 26, 6339–6359. https://doi.org/10.5194/hess-26-6339-2022.

Copyright (c) 2023 Shijie Jiang. All rights reserved.

You should have received a copy of the MIT license along with the code. If not,
see <https://opensource.org/licenses/MIT>
"""

import numpy as np
from scipy import signal
from keras import backend as K

def cal_nse(obs, sim):
    """
    Calculate Nash-Sutcliffe model efficinecy.

    Parameters
    ----------
    obs: observed data.
    sim: simulation data.

    Returns
    ----------
    nse: Nash-Sutcliff model efficiency
    """

    # compute numerator and denominator
    numerator   = np.nansum((obs - sim)**2)
    denominator = np.nansum((obs - np.nanmean(obs))**2)
    # compute coefficient
    return 1 - (numerator / denominator)


def identify_peaks(Q, distance=14, **kwargs):
    """
    Identify flood peaks based on scipy find_peaks function.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    Parameters
    ----------
    Q: pandas series of streamflow observations.
    distance: minimal horizontal distance in samples between neighboring peaks (default: 14).
    **kwargs: additional arguments with keywords passed to the scipy.signal.find_peaks() call.

    Returns
    ----------
    peak_time: a sequence of flood peaks' occurrence dates
    """
    peaks_index, _ = signal.find_peaks(
        Q,
        distance=distance,
        prominence=np.quantile(Q, 0.95),
        width=1,
        rel_height=0.5,
        **kwargs
    )

    peak_time = Q.iloc[peaks_index].index

    print(f"A total of {len(peak_time)} flood peaks are identified.")

    return peak_time


def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return SS_res/(SS_tot + K.epsilon())


def identify_AM_peaks(Q):

    candidate_peak_indices, _ = signal.find_peaks(Q)

    peak_year = Q.iloc[candidate_peak_indices].index.year.unique()
    am_peaks_date = Q.groupby(Q.index.year).idxmax()
    am_peaks_date = am_peaks_date[Q.groupby(Q.index.year).count()>=180]       
    peaks = Q.loc[am_peaks_date]
        
    peak_time = peaks.index

    print(f"A total of {len(peak_time)} AM flood peaks are identified.")

    return peak_time
