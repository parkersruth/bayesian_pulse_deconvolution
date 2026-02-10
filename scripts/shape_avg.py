import numpy as np
import pandas as pd

from utils.signal import segment_pulses

from snakemake.script import snakemake as sm


if __name__ == '__main__':
    df_filt = pd.read_csv(sm.input['filt'])
    df_time = pd.read_csv(sm.input['time'])
    t = df_filt.t.values
    y = df_filt.y.values
    z = df_time.z.values

    if np.all(np.isnan(y)) or len(z) == 0:
        t_h = t[t <= t[0]+1] - t[0]
        h = np.full_like(t_h, np.nan)
    else:
        segments, t_h = segment_pulses(z, y, t)
        h = np.nanmean(segments, axis=0)

    df_h = pd.DataFrame({'t_h':t_h, 'h':h})
    df_h.to_csv(sm.output['shape'], index=False)
