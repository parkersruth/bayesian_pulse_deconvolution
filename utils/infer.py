
from numpy.fft import fft

from jax import lax
from jax import random
from jax import numpy as jnp
import optax

from numpyro.optim import Adam
from numpyro.infer import Trace_ELBO
from numpyro.infer import SVI
from numpyro.infer import init_to_value
from numpyro.infer import init_to_sample
from numpyro.infer.autoguide import AutoDelta
from numpyro.distributions import CategoricalProbs
from numpyro.distributions import Uniform
from numpyro.distributions import Gamma
from numpyro.infer import MCMC, NUTS

from utils import models
from utils.signal import filter_downsample

from tqdm.auto import trange


MIN_HR = 30
MAX_HR = 200
Tw = 1.5


def first_guess_params(t_obs, y_obs):
    a_init = y_obs.std()
    Tb = (t_obs.max() - t_obs.min())*2
    f_b = jnp.arange(1,jnp.floor(Tb/Tw)+1)/Tb
    b_1, b_0 = jnp.polyfit(t_obs, y_obs, 1)
    b_init = jnp.array([b_0, b_1]+[0]*(len(f_b)*2))
    sigma_e_init = y_obs.std()
    return a_init, b_init, f_b, sigma_e_init


def svi_hr_init(
        seed,
        prior,
        obs,
        config = {},
        init = {},
        fs_ds = 16,
        num_samples = 16,
        num_steps = 64,
        step_size = 0.01,
        progress=False,
):
    rkey = random.key(seed)

    t_obs = obs['t']
    y_obs = obs['y']

    if fs_ds > 0:
        t_ds, y_ds, fs_ds = filter_downsample(t_obs, y_obs, fs_ds=fs_ds)
    else:
        t_ds, y_ds = t_obs, y_obs
        fs_ds = (y_obs.size-1)/(t_obs.max()-t_obs.min())

    obs_ds = {
        't': t_ds,
        'y': y_ds,
    }

    N_ds = len(y_ds)
    y_psd = jnp.abs(fft(y_ds))**2
    f_obs = jnp.arange(N_ds) / N_ds * fs_ds
    hr_region = (f_obs >= MIN_HR/60) & (f_obs <= MAX_HR/60)

    f_hr = f_obs[hr_region]
    p_f_hr = y_psd[hr_region]
    p_f_hr /= jnp.sum(p_f_hr * jnp.gradient(f_hr))

    hr_prior = Gamma(prior['alpha_hr'], prior['lambd_hr'])
    p_f_hr_prior = jnp.exp(hr_prior.log_prob(jnp.log(f_hr * 60))) + 0.5

    p_f_hr_post = p_f_hr * p_f_hr_prior
    p_f_hr_post /= jnp.sum(p_f_hr_post * jnp.gradient(f_hr))

    # get initial configurations (but allow caller overrides)
    a_init, b_init, f_b, sigma_e_init = first_guess_params(t_obs, y_obs)
    config = {
        't0': t_obs.mean(),
        'f_b': f_b,
        **config,
    }
    init = {
        'a': a_init,
        'b': b_init,
        'sigma_e': sigma_e_init,
        **init,
    }

    # initialize b
    model = models.model_factory(prior=prior,
                                config={'M':7, **config},
                                obs={'a':0.0, **obs_ds},
                                init=init)
    auto_guide = AutoDelta(model, init_loc_fn=init_to_sample())
    optimizer = Adam(step_size=0.1)
    loss = Trace_ELBO()
    svi = SVI(model, auto_guide, optimizer, loss)
    svi_state = svi.init(rkey, t=t_obs, y=y_obs)

    def loop_fun(i, svi_state):
        svi_state, loss = svi.update(svi_state, t=t_obs, y=y_obs)
        return svi_state
    b_init_state = lax.fori_loop(0, 100, loop_fun, svi_state)
    b_init = svi.get_params(b_init_state)['b']
    init['b'] = b_init

    best_loss = jnp.inf
    best_state = None
    map_params = None
    map_sample = None
    for i in trange(num_samples, disable=(not progress)):
        rkey_i = random.fold_in(rkey, i)

        # piecewise uniform distribution centered on f_hr with weights p_f_hr
        dist_hr = CategoricalProbs(p_f_hr_post)
        dist_uni = Uniform(-0.5, 0.5)
        rkey_dist_hr, rkey_dist_uni = random.split(rkey_i, 2)
        idx_hr = dist_hr.sample(rkey_dist_hr)
        f_hr_obs = f_hr[idx_hr] + dist_uni.sample(rkey_dist_uni) * jnp.gradient(f_hr)[idx_hr]
        hr_obs = f_hr_obs * 60
                
        # trim by default
        duration = t_obs.max() - t_obs.min()        
        M = int(jnp.ceil(duration * f_hr_obs + 4))
        model = models.model_factory(prior=prior,
                                    config={'M':M, **config},
                                    obs={'hr':hr_obs, **obs_ds},
                                    init=init)
        
        auto_guide = AutoDelta(model, init_loc_fn=init_to_sample())
        optimizer = Adam(step_size=step_size)
        loss = Trace_ELBO()
        svi = SVI(model, auto_guide, optimizer, loss)

        svi_state = svi.init(rkey_i, t=t_obs, y=y_obs)

        def loop_fun(i, svi_state):
            svi_state, loss = svi.update(svi_state, t=t_obs, y=y_obs)
            return svi_state
        svi_state = lax.fori_loop(0, num_steps, loop_fun, svi_state)

        loss = svi.evaluate(svi_state, t=t_obs, y=y_obs)
        if loss < best_loss:
            best_loss = loss
            best_state = svi_state
            map_params = svi.get_params(best_state)
            map_sample = auto_guide.sample_posterior(rkey, map_params, t=t_obs, y=y_obs)

            map_sample['hr'] = hr_obs

    solution = {**map_sample, **map_params}
    return solution


def svi_fine_tune(
    seed,
    prior,
    obs,
    config = {},
    init = {},
    num_steps=3000,
    progress=False,
):

    rkey1, rkey2 = random.split(random.key(seed))

    vals = ['hr', 'zpp', 'zm*', 'w*']
    init_vals = {v:init[v] for v in vals if v in init}

    # re-center t0 to avoid zm near edges
    zp = init['zp']
    m = init['m']
    z = init['z']
    M = init['M']
    config = {
        't0': z[m-1] - zp[m-2]/2,
        'f_b': init['f_b'],
        'M': M,
    }
    init_vals['zm*'] = 0.5
    for val in prior.keys() | obs.keys():
        init_vals.pop(val, None)

    model = models.model_factory(prior, config, obs, init)
    
    auto_guide = AutoDelta(model, init_loc_fn=init_to_value(values=init_vals))

    warmup = max(100, int(0.05 * num_steps))
    peak = 5e-3
    end = 1e-6
    decay_steps = max(1, num_steps - warmup)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak,
        warmup_steps=warmup,
        decay_steps=decay_steps,
        end_value=end,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule, b1=0.9, b2=0.999, eps=1e-8),
    )

    loss = Trace_ELBO(num_particles=1)
    svi = SVI(model, auto_guide, optimizer, loss=loss)

    kwargs_infer = {'t': obs['t'], 'y':obs['y']}
    svi_result = svi.run(rkey1, num_steps, progress_bar=progress, **kwargs_infer)
    map_params = svi_result.params
    map_sample = auto_guide.sample_posterior(rkey2, map_params, **kwargs_infer)

    solution = {**map_sample, **map_params}
    return solution


def run_bayesian_pd(t, y, prior, seed=0):
    y_norm = (y - y.mean()) / y.std()

    # run HR initialization step
    init = svi_hr_init(
        seed = seed,
        prior = prior,
        obs = {'t': t, 'y': y_norm},
    )

    init['b'] = init['b'].at[0].subtract(y.mean())
    init['b'] /= y.std()
    init['a'] /= y.std()
    init['sigma_e'] /= y.std()

    soln = svi_fine_tune(
        seed = seed,
        prior = prior,
        obs = {'t': t, 'y': y_norm},
        init = init,
        progress = False,
    )
    
    soln['y_bar_ab'] *= y.std()
    soln['y_bar'] *= y.std()
    soln['a'] *= y.std()
    soln['b'] *= y.std()
    soln['sigma_e'] *= y.std()
    
    soln['b'] = soln['b'].at[0].add(y.mean())
    soln['y_bar_ab'] += y.mean()
    soln['y_bar'] += y.mean()

    return soln


def bayesian_pd_mcmc(prior, obs, config, init, init_vals, num_samples=1000, num_warmup=100):

    model = models.model_factory(prior, config, init=init)

    nuts = NUTS(model, init_strategy=init_to_value(values=init_vals))
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.key(0), **obs)
    post_samples = mcmc.get_samples()
    return post_samples

