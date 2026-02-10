
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
from scipy import integrate
import scipy.signal as ss

from utils import models
from utils.plots import *



def match_z_pairs(za, zb):
    '''find pulses in zb matching each pulse in za'''

    # lists of pulse times must be sorted
    za = np.sort(za)
    zb = np.sort(zb)

    if za.size < 3:
        return np.empty((0,2))

    # initialize empty data structure for matches
    z_pairs = np.full((za.size-2, 2), np.nan)

    for i in range(1,len(za)-1):
        za_last, za_current, za_next = za[i-1:i+2]

        # search for zb estimated pulse time that matches z_current
        # search within window [t_min, t_max)
        t_min = (za_last + za_current) / 2
        t_max = (za_current + za_next) / 2

        # find zb times that are close to z_current
        zb_close = zb[(zb >= t_min) & (zb < t_max)]
        
        if len(zb_close) == 0:
            # no MAP pulse found in range -- false negative
            z_pairs[i-1,:] = za_current, np.nan
        elif len(zb_close) > 1:
            # more than one pulse found in range -- false positive(s)
            # only keep the closest one
            zb_closest = zb_close[np.argmin(np.abs(zb_close - za_current))]
            z_pairs[i-1,:] = za_current, zb_closest
        else:
            # exactly one matching pulse found in search window
            z_pairs[i-1,:] = za_current, zb_close[0]
          
    return z_pairs


def get_rmse(v1, v2):
    v1 = np.squeeze(v1)
    v2 = np.squeeze(v2)
    assert v1.shape == v2.shape
    if np.all(np.isnan(v1)) or np.all(np.isnan(v2)):
        return np.nan
    v_error = v2 - v1
    return np.sqrt(np.nanmean(v_error**2))


def get_y_snr(y_true, y_obs, y_pred):
    if np.std(y_pred) == 0:
        return np.nan
    
    y_pred_scale = (y_pred - y_pred.mean()) / y_pred.std()
    y_pred_scale = y_pred_scale * y_true.std() + y_true.mean()

    # weight by hann function to ignore edge artifacts
    window = ss.windows.hann(y_true.size)
    window /= np.sum(window)

    noise = y_obs - y_true
    error = y_pred_scale - y_true
    true_power = np.sum((y_true-y_true.mean())**2*window)
    error_power = np.sum(error**2*window)
    noise_power = np.sum(noise**2*window)
    error_db = 10 * np.log(error_power / true_power)
    noise_db = 10 * np.log(noise_power / true_power)
    loss_db = error_db - noise_db
    power_feats = (true_power, noise_power, error_power)
    db_feats = (noise_db, error_db, loss_db)
    return  power_feats, db_feats


def filter_metrics(t, y_true, y_obs, y_pred):
    M = {}
    M['rmse'] = get_rmse(y_true, y_pred)
    power_feats, db_feats = get_y_snr(y_true, y_obs, y_pred)
    M['true_power'], M['noise_power'], M['error_power'] = power_feats
    M['noise_db'], M['error_db'], M['loss_db'] = db_feats
    return M
  

def timing_metrics(z_true, z_pred, t_obs):
    z_true = z_true[(z_true >= min(t_obs)) & (z_true <= max(t_obs))]
    z_pred = z_pred[(z_pred >= min(t_obs)) & (z_pred <= max(t_obs))]

    z_pairs_true = match_z_pairs(z_true, z_pred)
    z_pairs_pred = match_z_pairs(z_pred, z_true)
    tp = len(z_pairs_true) - np.isnan(z_pairs_true).sum()
    fn = np.isnan(z_pairs_true).sum()
    fp = np.isnan(z_pairs_pred).sum()
    
    z_error = np.diff(z_pairs_true, axis=1)
    if sum(~np.isnan(z_error)) < 3:
        z_me = np.nan
        z_rmse = np.nan
    else:
        z_me = np.nanmean(z_error)
        z_rmse = np.sqrt(np.nanmean(z_error**2))

    zp_pairs = np.diff(z_pairs_true, axis=0)
    zp_error = np.diff(zp_pairs, axis=1)

    if np.sum(~np.isnan(zp_error)) < 3:
        zp_rmse = np.nan
    else:
        zp_rmse = np.sqrt(np.nanmean(zp_error**2))

    zp_pairs[np.any(np.isnan(zp_pairs), axis=1)] = np.nan

    metrics = dict(
        z_tp = tp,
        z_fn = fn,
        z_fp = fp,
        z_me = z_me,
        z_rmse = z_rmse,
        zp_rmse = zp_rmse,
    )
    return metrics


def shape_metrics(true, t_h, h_pred):
    t_h = np.array(t_h)
    h_pred = np.array(h_pred)

    if np.all(np.isnan(h_pred)):
        return {}
    
    if 'w' not in true:
        return {}

    t_h = np.array(t_h)
    h_pred = np.array(h_pred)

    h_true_fun = lambda t: models.h_fun(t, true['w'], true['f_h'], true['g'])
    h_pred -= np.min(h_pred)

    def loss(x):
        h_true = h_true_fun(t_h - x[0])
        return -pearsonr(h_true, h_pred)[0]
    bounds = [(-0.2, 0.2)]
    res = minimize(loss, [0,], method='Nelder-Mead', tol=1e-6, bounds=bounds)
    shift = res.x[0]
    t_h -= shift

    fs = (len(t_h)-1)/(t_h[-1]-t_h[0])

    # trim if too long
    if np.any(t_h < -0.5):
        trim_l = np.argmax(t_h >= -0.5)
        t_h = t_h[trim_l:]
        h_pred = h_pred[trim_l:]
    if np.any(t_h > 2.0):
        trim_r = np.argmax(t_h > 2.0)
        t_h = t_h[:trim_r]
        h_pred = h_pred[:trim_r]

    # pad if too short
    if t_h[0] > -0.5 + 1/fs:
        lpad = int(np.floor((t_h[0] + 0.5) * fs))
        t_h_pad = t_h[0] - lpad/fs + np.arange(lpad)/fs
        t_h = np.concatenate([t_h_pad, t_h])
        h_pred = np.concatenate([np.zeros(lpad), h_pred])
    if t_h[-1] < 2.0 - 1/fs:
        rpad = int(np.floor((2.0 - t_h[-1]) * fs))
        t_h_pad = t_h[-1] + np.arange(1, rpad+1)/fs
        t_h = np.concatenate([t_h, t_h_pad])
        h_pred = np.concatenate([h_pred, np.zeros(rpad),])

    def compute_errea(h_true, h_pred):
        h_error = np.abs(h_pred - h_true)
        h_error_area = integrate.trapezoid(h_error, t_h)
        h_true_area = integrate.trapezoid(h_true, t_h)
        h_errea = h_error_area / h_true_area
        return h_errea

    h_true = h_true_fun(t_h)
    
    h_pred /= integrate.trapezoid(h_pred, t_h)
    h_true /= integrate.trapezoid(h_true, t_h)

    h_errea = compute_errea(h_true, h_pred)
    h_pearson = pearsonr(h_true, h_pred)[0]
    h_fisher = np.arctanh(h_pearson)

    metrics = dict(
        h_errea = h_errea,
        h_pearson = h_pearson,
        h_fisher = h_fisher,
    )
    return metrics

