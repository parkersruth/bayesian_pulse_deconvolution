from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from utils.metrics import timing_metrics

from snakemake.script import snakemake as sm


config_vars = ['duration', 'sample_freq', 'snr']

if __name__ == '__main__':
    true_fpaths = sm.input['true']
    obs_fpaths = sm.input['obs']
    pred_fpaths = sm.input['pred']

    all_metrics = []
    for true_fpath, obs_fpath, pred_fpath in zip(true_fpaths, obs_fpaths, pred_fpaths):
        i = Path(true_fpath).stem.split('_')[0]
        assert i == Path(obs_fpath).stem.split('_')[0] == Path(pred_fpath).stem.split('_')[0]

        with open(true_fpath, 'rb') as f:
            true = pickle.load(f)
        z_true = true['z']

        df_obs = pd.read_csv(obs_fpath)
        t_obs = df_obs.t.values

        df_pred = pd.read_csv(pred_fpath)
        z_pred = df_pred.z
        
        metrics = timing_metrics(z_true, z_pred, t_obs)
        metadata = {'i':i, **{var:true.get(var) for var in config_vars}}
        metrics = {**metadata, **metrics}
        all_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(sm.output['metrics'], index=False)

