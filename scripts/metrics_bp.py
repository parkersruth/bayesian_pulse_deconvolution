from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from utils.metrics import bp_metrics

from snakemake.script import snakemake as sm


if __name__ == '__main__':
    true_fpaths = sm.input['true']
    obs_fpaths = sm.input['obs']
    pred_fpaths = sm.input['pred']

    params = sm.params

    all_true = []
    all_h = []
    for true_fpath, obs_fpath, pred_fpath in zip(true_fpaths, obs_fpaths, pred_fpaths):
        i = Path(true_fpath).stem.split('_')[0]
        assert i == Path(obs_fpath).stem.split('_')[0] == Path(pred_fpath).stem.split('_')[0]

        with open(true_fpath, 'rb') as f:
            true = pickle.load(f)
        all_true.append(true)

        df_pred = pd.read_csv(pred_fpath)
        t_pred, h_pred = df_pred.t_h.values, df_pred.h.values
        all_h.append(h_pred)
            
    df_true = pd.DataFrame(all_true)
    df_true.drop(columns=['t', 'z'], inplace=True)

    metrics = bp_metrics(df_true, all_h)

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(sm.output['metrics'], index=False)
