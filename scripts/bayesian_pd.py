import pickle
import numpy as np
import pandas as pd

from utils import infer
from utils import models
from utils.relaxations import relax_prior
from utils.plots import *

from snakemake.script import snakemake as sm


if __name__ == '__main__':
    with open(sm.input['prior'], 'rb') as f:
        prior = pickle.load(f)

    df = pd.read_csv(sm.input['obs'])
    t, y = df.t.values, df.y.values

    seed = int(sm.params['i'])

    relax_time = sm.params.get('relax_time', 'X')
    relax_shape = sm.params.get('relax_shape', 'X')
    prior = relax_prior(prior, relax_time, relax_shape)

    soln = infer.run_bayesian_pd(t, y, prior, seed)

    # full solution
    with open(sm.output['soln'], 'wb') as f:
        pickle.dump(soln, f)

    # filter
    y_filt = soln['y_bar']
    y_filt = (y_filt - y_filt.mean()) / y_filt.std()
    df_filt = pd.DataFrame({'t':t, 'y':y_filt})
    df_filt.to_csv(sm.output['filt'], index=False)

    # time
    z = soln['z']
    df_time = pd.DataFrame({'z': z})
    df_time.to_csv(sm.output['time'], index=False)

    # shape
    fs = (len(t)-1)/(t[-1]-t[0])
    t_h = np.arange(-0.5, 2.0, 1/fs)
    h = models.h_fun(t_h, soln['w'], soln['f_h'], soln['g'])
    df_shape = pd.DataFrame({'t_h':t_h, 'h':h})
    df_shape.to_csv(sm.output['shape'], index=False)


