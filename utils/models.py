from jax import random
from jax import numpy as jnp
from jax.scipy.special import expit

from numpyro import param
from numpyro import sample
from numpyro import deterministic as determ
from numpyro.distributions import Gamma
from numpyro.distributions import Normal
from numpyro.distributions import MultivariateNormal
from numpyro.distributions import Uniform
from numpyro.distributions.constraints import positive
from numpyro.infer import Predictive

from utils.plots import *


# Analytical vascular waveform model

def gamma(g):
    g1, g2, g3, g4 = g
    gamma1 = lambda t: expit((t-g1)*g2)
    gamma2 = lambda t: expit(-(t-g3)*g4)
    return lambda tt: gamma1(tt) * gamma2(tt)


def h_fun_vec(t, w, f_h, g):
    if w.size != f_h.size*2:
        raise Exception(f'w.size({w.size}) != f_h.size*2({f_h.size*2})')
    tf = 2*jnp.pi * jnp.einsum('i,jk...', f_h, t)
    h = 1
    h += jnp.einsum('i,i...', w[0::2], jnp.sin(tf))
    h += jnp.einsum('i,i...', w[1::2], jnp.cos(tf))
    h *= gamma(g)(t)
    return h


def h_fun(t, w, f_h, g):
    return h_fun_vec(t[:,None], w, f_h, g).squeeze()


def y_fun(t, z, w, f_h, g):
    return h_fun_vec(t[:,None]-z, w, f_h, g).sum(1)


def b_fun(t, b, f_b):
    if b.size != f_b.size*2+2:
        raise Exception(f'b.size({b.size}) != f_b.size*2+2({f_b.size*2+2})')
    tf = 2*jnp.pi * jnp.einsum('i,j...', f_b, t)
    baseline = b[0]
    baseline += b[1] * t
    baseline += jnp.einsum('i,i...', b[2::2], jnp.sin(tf))
    baseline += jnp.einsum('i,i...', b[3::2], jnp.cos(tf))
    return baseline


# Bayesian priors

def prior_hr(alpha_hr, lambd_hr):
    hr = sample('hr', Gamma(alpha_hr, lambd_hr))
    return hr


def prior_z(hr, alpha_zpp, t0, m, D):
    zpp = sample('zpp', Gamma(alpha_zpp, alpha_zpp), sample_shape=(D.shape[1],))
    zp = determ('zp', zpp * 60/hr)
    zm_star = sample('zm*', Uniform(0, 1))
    zm = determ('zm', t0 + zm_star * zp[m-2])
    z = determ('z', zm + D @ zp)
    return z


def prior_w(mu_w, L_w):
    w_star = sample('w*', Normal(0, 1), sample_shape=mu_w.shape)
    w = determ('w', mu_w + w_star @ L_w)
    return w


# Numpyro model factory    

def model_factory(prior, config, obs={}, init={}):
    '''
    prior = {
        alpha_hr | None
        lambd_hr | None
        alpha_zpp | None
        mu_w | None
        L_w | None
        sigma_e | None
        f_h
        g
    }
    config = {
        t0
        M
        f_b | []
        m | None
    }
    obs = {
        sigma_e | None
        hr | None
        z | None
        w | None
        a | None
        b | None
    }
    init = {
        a | 1.0
        b | [0.0, 0.0]
        sigma_e | None
    }
    '''

    t0 = config['t0']

    if 'hr' in obs:
        hr = lambda : determ('hr', obs['hr'])
    else:
        alpha_hr = prior['alpha_hr']
        lambd_hr = prior['lambd_hr']
        hr = lambda : prior_hr(alpha_hr, lambd_hr)

    if 'z' in obs:
        M = len(obs['z'])
        m = config.get('m', M//2+1)
        z = lambda hr : determ('z', obs['z'])
    else:
        alpha_zpp = prior['alpha_zpp']
        M = config['M']
        m = config.get('m', M//2+1)
        D = jnp.block([
            [-jnp.tri(m-1).T, jnp.zeros((m-1,M-m))],
            [jnp.zeros((1,m-1)), jnp.zeros((1,M-m))],
            [jnp.zeros((M-m,m-1)), jnp.tri(M-m)],
        ])
        z = lambda hr: prior_z(hr, alpha_zpp, t0, m, D)
        
    f_h = lambda : determ('f_h', prior['f_h'])
    g = lambda : determ('g', prior['g'])
    if 'w' in obs:
        w = lambda : determ('w', obs['w'])
    else:
        mu_w = prior['mu_w']
        L_w = prior['L_w']
        w = lambda : prior_w(mu_w, L_w)

    if 'a' in obs:
        a = lambda : determ('a', obs['a'])
    else:
        a_init = init.get('a', 1.0)
        a = lambda : param('a', init_value=a_init, constraint=positive)

    f_b = lambda : determ('f_b', config.get('f_b', jnp.array([])))
    if 'b' in obs:
        b = lambda : determ('b', obs['b'])
    else:
        b_init = init.get('b', jnp.array([0.0]*2))
        b = lambda : param('b', init_value=b_init)

    if 'sigma_e' in prior:
        sigma_e = lambda : prior['sigma_e']
    else:
        sigma_e = lambda : param('sigma_e', init_value=init['sigma_e'])

    def model(t, y=None):
        determ('t0', t0)
        determ('M', M)
        determ('m', m)
        y_bar = determ('y_bar', y_fun(t, z(hr()), w(), f_h(), g()))
        y_bar_ab = determ('y_bar_ab', y_bar * a() + b_fun(t, b(), f_b()))
        return sample('y', Normal(y_bar_ab, sigma_e()), obs=y)

    return model


# Convenience functions

def sample_prior(seed, prior, config, obs={}, init={}):
    model = model_factory(prior, config, obs, init)
    rkey = random.key(seed)
    samples = Predictive(model, num_samples=1)(rkey, t=obs['t'])
    sample = {k:v[0] for k,v in samples.items()}
    return sample


def evaluate(t, z, w, f_h, g, a=None, b=None, f_b=None):
    y_bar = y_fun(t, z, w, f_h, g)
    y_bar_ab = y_bar.copy()
    if a is not None:
        y_bar_ab *= a
    if b is not None:
        y_bar_ab += b_fun(t, b, f_b)
    return y_bar, y_bar_ab


def hr_inverse_prior(hr, mu_log_hr, sigma_log_hr):
    log_hr_star = (jnp.log(hr) - mu_log_hr) / sigma_log_hr
    return log_hr_star

