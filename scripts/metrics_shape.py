from pathlib import Path
import pickle
import pandas as pd

from utils.metrics import shape_metrics

from snakemake.script import snakemake as sm


config_vars = ['duration', 'sample_freq', 'snr']

if __name__ == '__main__':
    true_fpaths = sm.input['true']
    obs_fpaths = sm.input['obs']
    pred_fpaths = sm.input['pred']

    params = sm.params

    all_metrics = []
    for true_fpath, obs_fpath, pred_fpath in zip(true_fpaths, obs_fpaths, pred_fpaths):
        i = Path(true_fpath).stem.split('_')[0]
        assert i == Path(obs_fpath).stem.split('_')[0] == Path(pred_fpath).stem.split('_')[0]

        with open(true_fpath, 'rb') as f:
            true = pickle.load(f)

        df_pred = pd.read_csv(pred_fpath)
        t_pred, h_pred = df_pred.t_h.values, df_pred.h.values
        
        metrics = shape_metrics(true, t_pred, h_pred)
        metadata = {'i':i, **{var:true.get(var) for var in config_vars}}
        metrics = {**metadata, **metrics}
        all_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(sm.output['metrics'], index=False)
