import pandas as pd
import numpy as np

import neurokit2 as nk

from snakemake.script import snakemake as sm


if __name__ == '__main__':
    # read input
    df = pd.read_csv(sm.input['filt'])
    t_filt, y_filt = df.t.values, df.y.values

    # timing
    fs = (t_filt.size - 1) / np.ptp(t_filt)

    if fs < 16.0:
        z = []
    else:
        _, info = nk.ppg_peaks(y_filt, sampling_rate=fs)
        locs = info['PPG_Peaks']
        z = t_filt[locs]

    # write output
    df_z = pd.DataFrame({'z': z})
    df_z.to_csv(sm.output['time'], index=False)

