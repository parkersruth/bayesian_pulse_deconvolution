import pandas as pd
import numpy as np

from utils import pyPPG as upy

from snakemake.script import snakemake as sm


if __name__ == '__main__':
    # read input
    df = pd.read_csv(sm.input['obs'])
    t_obs, y_obs = df.t.values, df.y.values
    fs = (len(t_obs)-1)/(t_obs[-1]-t_obs[0])

    duration = t_obs[-1] - t_obs[0]

    # normalize for numerical stability
    y_norm = (y_obs - y_obs.mean()) / y_obs.std()

    # filter
    if duration < 5 or fs <= 32:
        df.y = np.full_like(df.y, np.nan)
    else:
        df.y = upy.filter_signal(t_obs, y_norm)

    # write ouput
    df.to_csv(sm.output['filt'], index=False)

