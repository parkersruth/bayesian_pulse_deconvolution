
import pickle
import numpy as np
import pandas as pd

from utils import models
from utils.plots import *

from snakemake.script import snakemake as sm

MAX_HR = 150

def sample_sim(seed, duration, fs_obs, snr, prior):
    t_obs = np.arange(0, duration, 1/fs_obs)

    config = dict(
        t0 = t_obs.mean(),
        M = int(np.ceil(duration * MAX_HR/60)+4),
    )
    obs = dict(
        t = t_obs,
    )
    init = dict(
        sigma_e = 1.0, # arbitrary dummy value
    )

    # take initial sample
    sample = models.sample_prior(seed, prior, config, obs, init)

    # update prior with sigma_e scaled to desired SNR
    y_true = sample['y_bar']
    y_power = np.mean((y_true-y_true.mean())**2)
    init['sigma_e'] = np.sqrt(y_power/snr)

    # resample (using same seed, all other vars will be unchanged)
    sample = models.sample_prior(seed, prior, config, obs, init)

    # save true data
    true = {
        **sample,
        't': t_obs,
        'duration': duration,
        'sample_freq': fs_obs,
        'snr': snr,
    }

    # save observed data
    y_obs = np.array(sample['y'])
    obs = pd.DataFrame({'t':t_obs, 'y':y_obs})    
    return true, obs


if __name__ == '__main__':
    params = sm.params['params']
    seed = int(params['i'])
    duration = params['duration']
    fs_obs = params['sample_freq']
    snr = params['snr']

    with open(sm.input['prior'], 'rb') as f:
        prior = pickle.load(f)

    true, obs = sample_sim(seed, duration, fs_obs, snr, prior)

    with open(sm.output['true'], 'wb') as f:
        pickle.dump(true, f)

    obs.to_csv(sm.output['obs'], index=False)

