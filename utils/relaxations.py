
import numpy as np

relax_factors = {
    'A': 2,
    'B': 4,
    'C': 8,
}

time_ablate_val = 1.0
shape_ablate_val = 2

def relax_time_prior(prior, relax_time):
    prior = prior.copy()
    if relax_time == 'X': # original prior
        pass
    elif relax_time == 'Z': # uninformative prior
        prior['alpha_zpp'] = time_ablate_val
    else: # relaxed prior
        prior['alpha_zpp'] /= relax_factors[relax_time]
    return prior

def relax_shape_prior(prior, relax_shape):
    prior = prior.copy()
    if relax_shape == 'X': # original prior
        pass
    elif relax_shape == 'Z': # uninformative prior
        prior['L_w'] = np.eye(prior['L_w'].shape[0]) * shape_ablate_val
    else: # relaxed prior
        prior['L_w'] *= relax_factors[relax_shape]
    return prior

def relax_prior(prior, relax_time=None, relax_shape=None):
    prior = relax_time_prior(prior, relax_time)
    prior = relax_shape_prior(prior, relax_shape)
    return prior
