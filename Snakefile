from pathlib import Path
import yaml
import pickle
import numpy as np
import pandas as pd

CONFIG_DATA_PATH = 'config_data.csv'

pathvars:
    out = 'experiments',
    param = 'prior_params',
    fig = 'figs',
    aurora_data_dir = '../data/aurora',
    aurora_test_ppg_df_path = 'dfs/aurora_test_ppg_df.csv',
    aurora_test_ton_df_path = 'dfs/aurora_test_ton_df.csv',
    trends_df_path = 'dfs/df_trends.csv',

df_config_data = pd.read_csv(CONFIG_DATA_PATH)

figs = [
    'fig_demo_sim',
    'fig_demo_aurora',
    'fig1',
    'fig2',
    'fig3',
    'fig4',
    'fig_trends',
    ]

debugs = [
    ('simulation_demo', '0001', 'bpd'),
    ('simulation_demo', '0002', 'bpd'),
    ('simulation_demo', '0003', 'bpd'),
]


rule all:
    input:
        expand('figs/{fig}/{fig}_df.csv', fig=figs)

rule fig_demo_sim:
    input: 'figs/fig_demo_sim/fig_demo_sim_df.csv'
rule fig_demo_aurora:
    input: 'figs/fig_demo_aurora/fig_demo_aurora.csv'
rule fig1:
    input: 'figs/fig1/fig1_df.csv'
rule fig2:
    input: 'figs/fig2/fig2_df.csv'
rule fig3:
    input: 'figs/fig3/fig3_df.csv'
rule fig4:
    input: 'figs/fig4/fig4_df.csv'
rule fig_trends:
    input: 'figs/fig_trends/fig_trends_df.csv'
rule fig_ablations:
    input: 'figs/fig_ablations/fig_ablations_df.csv'
rule fig_relaxations:
    input: 'figs/fig_relaxations/fig_relaxations_df.csv'
    

rule debug:
    input:
        expand('<out>/{DSET}/debug/{i}_soln_{method}.pdf',
                    DSET=[x[0] for x in debugs],
                    i=[x[1] for x in debugs],
                    method=[x[2] for x in debugs])

### PRIORS ###

rule assemble_prior:
    input:
        alpha_hr = '<param>/alpha_hr.npy',
        lambd_hr = '<param>/lambd_hr.npy',
        alpha_zpp = '<param>/alpha_zpp.npy',
        mu_w = '<param>/mu_w.npy',
        L_w = '<param>/L_w.npy',
        f_h = '<param>/f_h.npy',
        g = '<param>/g.npy',
    output:
        pkl = '<out>/{DSET}/prior.pkl'
    run:
        prior = {k:np.load(v) for k,v in input.items()}
        with open(output['pkl'], 'wb') as f:
            pickle.dump(prior, f)


### SAMPLERS ###

def sample_config(W):
    row = df_config_data.query(f'dataset=="{W.D}" & tag=="{W.T}"').iloc[0]
    config = row.to_dict()
    return {'i': W.i, **config}

rule sample_sim:
    input:
        prior = '<out>/{D}_{T}/prior.pkl'
    output:
        true = '<out>/{D}_{T}/idx/{i}/{i}_soln_true.pkl',
        obs = '<out>/{D}_{T}/idx/{i}/{i}_sig_obs.csv',
    wildcard_constraints:
        pid = r'\d+',
        D = 'simulation',
    params:
        params = lambda W: sample_config(W)
    script:
        'scripts/sample_sim.py'

def aurora_input_map(W):
    if W.T in ('ppg', 'demo'):
        return {'df': '<aurora_test_ppg_df_path>'}
    elif W.T == 'ton':
        return {'df': '<aurora_test_ton_df_path>'}
    else:
        return {}

rule sample_aurora:
    input:
        unpack(aurora_input_map),
        data_dir = '<aurora_data_dir>',
    output:
        true = '<out>/{D}_{T}/idx/{i}/{i}_soln_true.pkl',
        obs = '<out>/{D}_{T}/idx/{i}/{i}_sig_obs.csv',
    wildcard_constraints:
        pid = r'\d+',
        D = 'aurora',
    params:
        params = lambda W: sample_config(W)
    script:
        'scripts/sample_aurora.py'


def trend_config(W):
    df_trend = pd.read_csv('dfs/df_trends.csv')
    config = df_trend.iloc[int(W.i)].to_dict()
    config = {**config, **W}
    return config

rule sample_trends:
    input:
        prior = '<out>/trends_base/prior.pkl'
    output:
        true = '<out>/trends_base/idx/{i}/{i}_soln_true.pkl',
        obs = '<out>/trends_base/idx/{i}/{i}_sig_obs.csv',
    wildcard_constraints:
        pid = r'\d+',
    params:
        params = lambda W: trend_config(W)
    script:
        'scripts/sample_sim.py'



### BAYESIAN PULSE DECONVOLUTION ###

rule bayesian_pd:
    input:
        prior = '<out>/{DSET}/prior.pkl',
        obs = '<out>/{DSET}/idx/{i}/{i}_sig_obs.csv',
    output:
        soln = '<out>/{DSET}/idx/{i}/{i}_soln_bpd.pkl',
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_F_bpd.csv',
        time = '<out>/{DSET}/idx/{i}/time/{i}_T_bpd.csv',
        shape = '<out>/{DSET}/idx/{i}/shape/{i}_S_bpd.csv',
    params:
        i = '{i}'
    script:
        'scripts/bayesian_pd.py'


rule bayesian_pd_relax:
    input:
        prior = '<out>/{DSET}/prior.pkl',
        obs = '<out>/{DSET}/idx/{i}/{i}_sig_obs.csv',
    output:
        soln = '<out>/{DSET}/idx/{i}/{i}_soln_bpd_RT{rt}_RS{rs}.pkl',
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_F_bpd_RT{rt}_RS{rs}.csv',
        time = '<out>/{DSET}/idx/{i}/time/{i}_T_bpd_RT{rt}_RS{rs}.csv',
        shape = '<out>/{DSET}/idx/{i}/shape/{i}_S_bpd_RT{rt}_RS{rs}.csv',
    params:
        i = '{i}',
        relax_time = '{rt}',
        relax_shape = '{rs}',
    script:
        'scripts/bayesian_pd.py'


### FILTERING ###

rule filt_ideal:
    input:
        true = '<out>/{DSET}/idx/{i}/{i}_soln_true.pkl'
    output:
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_F_ideal.csv'
    run:
        with open(input['true'], 'rb') as f:
            true = pickle.load(f)
        t, y = true['t'], true['y_bar']
        y_norm = (y - y.mean()) / y.std()
        df_filt = pd.DataFrame({'t':t, 'y':y_norm})
        df_filt.to_csv(output['filt'], index=False)

rule filt_butter:
    input:
        obs = '<out>/{DSET}/idx/{i}/{i}_sig_obs.csv'
    output:
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_F_butter.csv'
    script:
        'scripts/filt_butter.py'

rule filt_cheby:
    input:
        obs = '<out>/{DSET}/idx/{i}/{i}_sig_obs.csv'
    output:
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_F_cheby.csv'
    script:
        'scripts/filt_cheby.py'
        
rule filt_nk:
    input:
        obs = '<out>/{DSET}/idx/{i}/{i}_sig_obs.csv'
    output:
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_F_nk.csv'
    script:
        'scripts/filt_nk.py'
        
rule filt_pyp:
    input:
        obs = '<out>/{DSET}/idx/{i}/{i}_sig_obs.csv'
    output:
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_F_pyp.csv'
    conda:
        'pyppg'
    script:
        'scripts/filt_pyp.py'


### PULSE TIMING ###

rule time_ideal:
    input:
        true = '<out>/{DSET}/idx/{i}/{i}_soln_true.pkl'
    output:
        time = '<out>/{DSET}/idx/{i}/time/{i}_T_ideal.csv'
    run:
        with open(input['true'], 'rb') as f:
            true = pickle.load(f)
        df_filt = pd.DataFrame({'z':true['z']})
        df_filt.to_csv(output['time'], index=False)

rule time_nk:
    input:
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_{f_method}.csv'
    output:
        time = '<out>/{DSET}/idx/{i}/time/{i}_{f_method}_T_nk.csv'
    script:
        'scripts/time_nk.py'

rule time_pyp:
    input:
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_{f_method}.csv'
    output:
        time_sp = '<out>/{DSET}/idx/{i}/time/{i}_{f_method}_T_pyp.sp.csv',
        time_on = '<out>/{DSET}/idx/{i}/time/{i}_{f_method}_T_pyp.on.csv',
        time_u = '<out>/{DSET}/idx/{i}/time/{i}_{f_method}_T_pyp.u.csv',
        time_a = '<out>/{DSET}/idx/{i}/time/{i}_{f_method}_T_pyp.a.csv',
    conda:
        'pyppg'
    script:
        'scripts/time_pyp.py'


### PULSE SHAPE ###

rule shape_avg:
    input:
        filt = '<out>/{DSET}/idx/{i}/filt/{i}_{f_method}.csv',
        time = '<out>/{DSET}/idx/{i}/time/{i}_{t_method}.csv',
    output:
        shape = '<out>/{DSET}/idx/{i}/shape/{i}_{f_method}__{t_method}__S_avg.csv',
    script:
        'scripts/shape_avg.py'


### METRICS ###

def metrics_inputs(W, kind):
    row = df_config_data.query(f'dataset=="{W.D}" & tag=="{W.T}"').iloc[0]
    config = row.to_dict()
    idx = [f'{i:04d}' for i in range(config['num_samples'])]
    return dict(
        true = expand('<out>/{{D}}_{{T}}/idx/{i}/{i}_soln_true.pkl', i=idx),
        obs = expand('<out>/{{D}}_{{T}}/idx/{i}/{i}_sig_obs.csv', i=idx),
        pred = expand('<out>/{{D}}_{{T}}/idx/{i}/{kind}/{i}_{{method}}.csv', kind=kind, i=idx),
    )

rule metrics:
    input:
        unpack(lambda W: metrics_inputs(W, '{kind}'))
    output:
        metrics = '<out>/{D}_{T}/metrics/{kind}/metrics_{method}.csv'
    params:
        kind = '{kind}',
        dataset = '{D}',
        tag = '{T}',
    wildcard_constraints:
        kind = r'filt|time|shape'
    script:
        'scripts/metrics_{wildcards.kind}.py'


### FIGURES ###

def fig_inputs(W):
    config_fpath = f'figs/{W.fig}/{W.fig}.yml'
    with open(config_fpath, 'r') as f:
        config = yaml.safe_load(f)
    metrics = []
    for y in config['y']:
        exp = y['experiment']
        for x in y['x']:
            metrics.append(f"<out>/{exp}/metrics/{y['kind']}/metrics_{x['method']}.csv")
    return {
        'config': config_fpath,
        'metrics': metrics,
    }

rule fig_df:
    input:
        unpack(fig_inputs)
    output:
        df = 'figs/{fig}/{fig}_df.csv'
    run:
        dfs = []
        for fpath in input['metrics']:
            fpath = Path(fpath)
            df = pd.read_csv(fpath)
            df['method'] = fpath.stem.replace('metrics_', '')
            dfs.append(df)
        df = pd.concat(dfs)
        df.drop_duplicates(inplace=True)
        df.to_csv(output['df'], index=False)


### DEBUG ###

rule debug_bpd:
    input:
        true = '<out>/{DSET}/idx/{i}/{i}_soln_true.pkl',
        obs = '<out>/{DSET}/idx/{i}/{i}_sig_obs.csv',
        bpd = '<out>/{DSET}/idx/{i}/{i}_soln_{method}.pkl',
    output:
        pdf = '<out>/{DSET}/debug/{i}_soln_{method}.pdf'
    script:
        'scripts/debug_bpd.py'


