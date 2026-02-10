
import numpy as np
import pandas as pd
from dotmap import DotMap

from pyPPG import PPG
from pyPPG.preproc import Preprocessing
from pyPPG.fiducials import FpCollection

from .plots import *


def to_signal(t, y):
    fs = (t.size - 1) / np.ptp(t)
    sig = PPG(DotMap(
        start_sig = 0,
        end_sig = len(t),
        v = y,
        fs = fs,
    ))
    return sig


def filter_signal(t, y, f_lo=0.5000001, f_hi=12.0, order=4):
    sig = to_signal(t, y)
    sig.fL = f_lo
    sig.fH = f_hi
    sig.order = order
    sig.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10}

    result = Preprocessing(sig, filtering=True)
    return result[0]


def compute_fiducials(t, y):
    sig = to_signal(t, y)
    result = Preprocessing(sig, filtering=False)
    sig.filt_sig, sig.filt_d1, sig.filt_d2, sig.filt_d3 = result

    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction=pd.DataFrame({x:[True] for x in corr_on})
    sig.correction=correction
    fpex = FpCollection(sig)
    df_fid = fpex.get_fiducials(sig)
    
    return df_fid
