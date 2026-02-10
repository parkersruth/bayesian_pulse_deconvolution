
import pickle
import pandas as pd
from pathlib import Path

import numpy as np
from scipy.signal import resample
from jax import random

from utils.signal import find_ecg_peaks
from utils.signal import segment_pulses

from utils.plots import *

from snakemake.script import snakemake as sm

PAT_OFFSET = 0.3


if __name__ == '__main__':
    params = sm.params['params']

    i = params['i']
    duration = params['duration']
    fs_obs = params['sample_freq']

    df = pd.read_csv(sm.input['df'])

    # load waveform data
    fpath = Path(sm.input['data_dir']) / df.iloc[int(i)].waveform_file_path
    df_wav = pd.read_csv(fpath, delimiter='\t')
    t = df_wav['t'].values
    fs = (len(t)-1)/(t.max() - t.min())

    # find ECG peaks ("true" peak timing)
    ecg = df_wav['ekg'].values # electrocardiogram
    result = find_ecg_peaks(ecg, fs)
    r_peaks = result[0]
    z_true = t[r_peaks]
    z_true += PAT_OFFSET

    t_full = df_wav['t'].values

    y_full = df_wav['optical'].values
    y_full = y_full - y_full.mean()
    y_full = y_full / y_full.std()

    # trim
    t_start = t_full.min() + 2
    t_end = t_start + duration
    assert t_start >= t_full.min()
    assert t_end <= t_full.max()

    # get mean ECG-segmented pulse ("true" pulse shape)
    obs_window = (t_full >= t_start) & (t_full <= t_end)
    t_obs = t_full[obs_window]
    y_obs = y_full[obs_window]
    segments, t_h = segment_pulses(z_true, y_obs, t_obs)
    h_true = np.mean(segments, 0)

    # resample
    N_obs = int(np.round(y_full.size*fs_obs/fs))
    y_ds, t_ds = resample(y_full-y_full.mean(), N_obs, t_full)
    y_ds += y_full.mean()

    # extract observed signal
    obs_window = (t_ds >= t_start) & (t_ds <= t_end)
    t_obs = t_ds[obs_window]
    y_clean = y_ds[obs_window]

    y_obs = y_clean.copy()

    true = {
        't': t_obs,
        'z': z_true,
    }
    true = {**true, **df.iloc[int(i)].to_dict(), **params}

    obs = pd.DataFrame({'t':t_obs, 'y':y_obs})

    with open(sm.output['true'], 'wb') as f:
        pickle.dump(true, f)

    obs.to_csv(sm.output['obs'], index=False)
