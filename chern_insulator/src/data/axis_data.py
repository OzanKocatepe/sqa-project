import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class AxisData:
    """Contains all of the relevant data for the axes that we use.
    
    Parameters
    ----------
    tauAxisDim : ndarray[float]
        The non-dimensional points in time to evaluate our operators, offset from
        the time t. For the single-time correlations, this time is offset from t = 0, while
        for the double-time correlations this time is offset from some time t.
    tauAxisSec : ndarray[float]
        Same as tauAxisDim, but in units of seconds.
    tAxisDim: ndarray[float]
        The non-dimensional points in time which we use as the starting points for our double-time operators.
        The double time correlations are calculated as sigma_i (t) sigma_j (t + tau), where we take t to be in the steady state.
        Since we are solving the single-time correations via Fourier series, we can assume they are already in the steady state,
        so we simple take the tAxis to be some number of points from 0 to the period of the driving, in non-dimensional units,
        which we will evaluate the single-time correlations at when required.
    tAxisSec: ndarray[float]
        The same as tAxisDim, but in units of seconds.
    freqAxis : ndarray[float]
        The axis of the Fourier transform of our operators, calculated based on the
        sampling rate of tauAxis, in units of the driving frequency.
    """

    tau_axis_dim: np.ndarray[float]
    tau_axis_sec: np.ndarray[float]
    t_axis_dim: np.ndarray[float]
    t_axis_sec: np.ndarray[float]
    freq_axis: np.ndarray[float]