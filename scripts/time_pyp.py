import pandas as pd
import numpy as np

from utils import pyPPG as upy

from snakemake.script import snakemake as sm


if __name__ == '__main__':
    # read input
    df = pd.read_csv(sm.input['filt'])
    t_filt, y_filt = df.t.values, df.y.values

    # timing
    if np.all(np.isnan(y_filt)):
        df_fid = None
    else:
        try:
            df_fid = upy.compute_fiducials(t_filt, y_filt)
            df_fid = df_fid.map(lambda x: t_filt[int(x)] if not pd.isna(x) else np.nan)
        except:
            df_fid = None

    # write outputs
    for key, outpath in sm.output.items():
        if df_fid is None:
            z = []
        else:
            marker = key.split('_')[-1]
            z = df_fid[marker].dropna().values
        df_z = pd.DataFrame({'z': z})
        df_z.to_csv(outpath, index=False)
