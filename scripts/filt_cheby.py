import pandas as pd

from utils.plots import *
from utils.signal import filter_cheby

from snakemake.script import snakemake as sm


if __name__ == '__main__':
    # read input
    df = pd.read_csv(sm.input['obs'])
    t_obs, y_obs = df.t.values, df.y.values

    # normalize for numerical stability
    y_norm = (y_obs - y_obs.mean()) / y_obs.std()

    # filter
    df.y = filter_cheby(y_norm, t_obs, btype='bandpass')

    # write ouptut
    df.to_csv(sm.output['filt'], index=False)

