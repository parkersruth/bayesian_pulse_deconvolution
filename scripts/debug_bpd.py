from pathlib import Path
import pickle

import pandas as pd

from utils.plots import *
from utils import models

from snakemake.script import snakemake as sm

vars = ['z', 'w', 'f_h', 'g', 'a', 'b', 'f_b']

if __name__ == '__main__':
    true_fpath = sm.input['true']
    obs_fpath = sm.input['obs']
    bpd_fpath = sm.input['bpd']

    df = pd.read_csv(obs_fpath)
    t, y_obs = df.t.values, df.y.values

    plt.figure(figsize=(10,4))
    plt.plot(t, y_obs, '.-', c=cp[-1])

    with open(true_fpath, 'rb') as f:
        true = pickle.load(f)
    if 'w' in true:
        kwargs_true = {var:true.get(var) for var in vars}
        _, y_true = models.evaluate(t, **kwargs_true)
        plt.plot(t, y_true, '-', c='k')

    with open(bpd_fpath, 'rb') as f:
        bpd = pickle.load(f)
    kwargs = {var:bpd.get(var) for var in vars}
    y_bar, y_bar_ab = models.evaluate(t, **kwargs)
    plt.plot(t, y_bar_ab, c=cp[1])

    for z in bpd['z']:
        kwargs_z = {**kwargs}
        kwargs_z['z'] = [z]
        h = models.evaluate(t, **kwargs_z)[1]
        plt.plot(t, h, c=cp[-1], zorder=-1)

    plt.savefig(sm.output['pdf'])
    plt.close()
