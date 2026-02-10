import pandas as pd
import numpy as np

import neurokit2 as nk

from snakemake.script import snakemake as sm


if __name__ == '__main__':
    # read input
    df = pd.read_csv(sm.input['obs'])
    t_obs, y_obs = df.t.values, df.y.values

    # normalize for numerical stability
    y_norm = (y_obs - y_obs.mean()) / y_obs.std()

    # filter
    fs = (t_obs.size - 1) / np.ptp(t_obs)

    if fs <= 16.0:
        df.y = np.full_like(df.y, np.nan)
    else:
        df.y = nk.ppg_clean(y_norm, sampling_rate=fs, method='elgendi')

    # write output
    df.to_csv(sm.output['filt'], index=False)

