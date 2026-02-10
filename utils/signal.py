
import numpy as np
import scipy.signal as ss


MIN_HR = 30
MAX_HR = 200

# ECG_NOISE_CAP = 0.06
ECG_NOISE_CAP = 0.085
ECG_HRV_CAP = 0.1
ECG_AMPV_CAP = 0.1


def normalize(v):
    if np.all(np.isnan(v)):
        return v, np.nan, np.nan
    offset = np.nanmean(v)
    scale = np.nanstd(v)
    v_norm = (v - offset) / scale
    return v_norm, offset, scale


def find_ecg_peaks(ecg, fs):
    # ecg data is all zeros
    if np.all(np.isclose(ecg, 0)):
        return None
        
    # apply filters to accentuate sharp R peaks
    b, a = ss.iirfilter(5, 40.0, btype='low', ftype='butter', output='ba', fs=fs)
    ecg_filt = ss.filtfilt(b, a, ecg)
    b, a = ss.iirfilter(7, 10.0, btype='high', ftype='butter', output='ba', fs=fs)
    ecg_filt = ss.filtfilt(b, a, ecg_filt)


    ecg_filt = np.abs(ecg_filt)

    # heuristic noise score
    noisiness = np.mean(ecg_filt) / np.max(ecg_filt)
    if noisiness > ECG_NOISE_CAP:
        return None

    # max and min r-r intervals (in integer samples)
    max_rri = int(np.ceil(fs*60/MIN_HR))
    min_rri = int(np.floor(fs*60/MAX_HR))
    
    # smooth filtered ECG and find peaks
    window = ss.windows.hann(min_rri//2*2+1)
    window /= np.sum(window)
    smoothie = ss.convolve(ecg_filt, window, mode='same')
    locs, _ = ss.find_peaks(smoothie, distance=min_rri)

    # Only some of the find_peaks locations should be kept
    # Heuristic: adaptively set height threshold that minimizes HRV

    # sort peaks by height
    order = np.argsort(-smoothie[locs])

    # minimum number of peaks that must be kept
    min_keep = int(np.floor(len(ecg_filt) / max_rri))

    # try keeping the tallest nkeep peaks
    # keep track of heuristics for each threshold option
    gaps = [] # number of pulse intervals with invalid length
    hrv = [] # heart rate variability
    ampv = [] # peak amplitude variability
    for nkeep in range(min_keep, len(locs)+1):
    # nkeep = min_keep
        locs_keep = np.sort(locs[order][:nkeep])

        rri = np.diff([0, *locs_keep, len(ecg_filt)])
        gaps.append(np.sum(rri > max_rri))

        rri = np.diff(locs_keep)
        hrv.append(np.std(rri) / np.median(rri))
        amp = smoothie[locs_keep]
        ampv.append(np.std(amp) / np.max(amp))
    
    # apply heuristic criteria
    no_gaps = np.array(gaps) == 0
    valid_hrv = np.array(hrv) < ECG_HRV_CAP
    valid_ampv = np.array(ampv) < ECG_AMPV_CAP
    hrv = np.array(hrv)
    hrv[~(no_gaps & valid_hrv & valid_ampv)] = np.nan
    
    # if cannot find any thresholds that satisfy criteria
    if np.all(np.isnan(hrv)):
        return None
    
    # keep the best number of pulses
    nkeep = min_keep + np.nanargmin(hrv)
    locs_keep = np.sort(locs[order][:nkeep])

    # re-align pulses to local peaks in filtered ECG
    rpk_locs = []
    for loc in locs_keep:
        ia = max(0,loc-min_rri//4)
        ib = min(len(ecg_filt)-1, loc+min_rri//4)
        rpk_locs.append(ia + np.argmax(ecg_filt[ia:ib]))

    if rpk_locs[0] < min_rri:
        rpk_locs = rpk_locs[1:]
    if rpk_locs[-1] > len(ecg)-min_rri:
        rpk_locs = rpk_locs[:-1]

    heuristics = noisiness, hrv[nkeep-min_keep], ampv[nkeep-min_keep]
    return rpk_locs, ecg_filt, smoothie, heuristics


def segment_pulses(z, y, t=None, fs=None):
    if fs is None:
        fs = (t.size-1)/(t.max()-t.min())
    if t is None:
        t = np.arange(len(y)) / fs

    # trim only z within observed time window
    z = z[(z > min(t)) & (z < max(t))]

    fs = (len(t)-1)/(t.max()-t.min())

    N, M = len(y), len(z)
    idx = np.argmax(t >= z[:,None], axis=1)+1
    y_shift = np.full((M,N), np.nan)
    for j in range(M):
        k = N//2 - idx[j]
        y_shift[j,max(0,k):min(N,N+k)] = y[max(0,-k):min(N,N-k)]
    h_mean = np.nanmean(y_shift, axis=0)

    # get median R-R interval
    med_rri = np.median(np.diff(np.sort(z)))
    if np.isnan(med_rri):
        return np.array([[np.nan]]), np.array([0.0])
    W = int(np.floor(med_rri*fs))

    ia, ib = N//2-W//2, N//2+W//2
    loc_peak = ia + np.argmax(h_mean[ia:ib])

    ia, ib = loc_peak-W, loc_peak
    loc_start = ia + np.argmin(h_mean[ia:ib])

    ia, ib = loc_peak, loc_peak+W
    loc_end = ia + np.argmin(h_mean[ia:ib])

    ia, ib = loc_start-N//2, loc_end-N//2
    segments = []
    for i in range(M):
        idx_start, idx_end = idx[i] + ia, idx[i] + ib
        if idx_start >= 0 and idx_end < N:
            segments.append(y[idx_start:idx_end])
    segments = np.array(segments)
    t_h = np.arange(segments.shape[1]) / fs
    return segments, t_h



def filter_iir(y, t=None, fs=None, btype='bandpass',
               f_hp=None, f_lp=None, order=4, **kwargs):

    # Default cutoffs: Goda et al., 2024, https://doi.org/10.1088/1361-6579/ad33a2
    f_hp = f_hp or 0.5
    f_lp = f_lp or 12.0

    if fs is None:
        fs = (t.size-1)/(t.max()-t.min())

    # validate cutoff frequencies
    # gracefully handle Nyquist violations
    assert btype in ('lowpass', 'bandpass', 'highpass')

    if btype=='bandpass':
        if f_lp < fs/2:
            Wn = [f_hp, f_lp]
        else:
            btype='highpass'

    if btype=='lowpass':
        if f_lp < fs/2:
            Wn = f_lp
        else:
            return y
    
    if btype=='highpass':
        if f_hp < fs/2:
            Wn = f_hp
        else:
            return np.zeros_like(y)
    
    sos = ss.iirfilter(N=order, Wn=Wn, btype=btype, fs=fs, output='sos', **kwargs)


    # normalize before filtering to mitigate edge artifacts
    y_norm, offset, scale = normalize(y)
    y_filt = ss.sosfiltfilt(sos, y_norm) * scale
    if btype == 'lowpass':
        y_filt += offset
    return y_filt


def filter_butter(*args, **kwargs):
    # Default filter setup: Lapitan, et al. 2024, https://doi.org/10.1038/s41598-024-57297-3
    kwargs['ftype'] = kwargs.get('ftype') or 'butter'
    kwargs['order'] = kwargs.get('order') or 2
    return filter_iir(*args, **kwargs)


def filter_cheby(*args, **kwargs):
    # Liang et al., 2018, https://doi.org/10.1038/sdata.2018.76
    kwargs['ftype'] = kwargs.get('ftype') or 'cheby2'
    kwargs['order'] = kwargs.get('order') or 4
    kwargs['rs'] = kwargs.get('rs') or 20
    return filter_iir(*args, **kwargs)


def filter_downsample(t, y, fs_ds=16, **kwargs):
    N = t.size
    t_start, t_end = t.min(), t.max()
    fs = (N-1)/(t_end-t_start)
    
    y_filt = filter_butter(y, t, **kwargs)

    if fs_ds < fs:
        N_ds = int(np.round((t_end-t_start)*fs_ds+1))
        y_ds, t_ds = ss.resample(y_filt, N_ds, t)
        return t_ds, y_ds, fs_ds
    else:
        return t, y_filt, fs
