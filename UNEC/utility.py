import numpy as np
import pandas as pd

# toolbox functions
def _expit(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x) # prevent overflow in exp(-x) for large negative x
    return out

def _get_rng(random_state):
    if random_state is None: # global random state
        return np.random
    if isinstance(random_state, np.random.Generator): # shared random state
        return random_state
    return np.random.default_rng(random_state) # whole number seed for reproducibility

# loss function
def loss_Jin2023(Y, tau, c=0):
    """Calculates the smoothened indicator loss function, similar to the data generation process in Jin and Candes (2023).
    
    The loss function is of the form sigmoid(-tau * (Y-c)).
    
    Args:
        Y (np.ndarray): The target values.
        tau (float): Hyperparameter for smoothing. Larger tau means closer to 1{Y <= c}. 
            If tau = np.inf, it returns strictly 1{Y <= c}.
        c (float, optional): The threshold value for the indicator function. Defaults to 0.
            
    Returns:
        np.ndarray: The computed loss values.
    """
    if tau != np.inf:
        return _expit(-tau * (Y - c)) # L = smoothened indicator of <= c
    else:
        return (Y <= c)

# Data generation function
def gen_data_Jin2023(n, sig=0.1, dim=20, random_state=None):
    """Generates artificial data using the data generation process in Jin and Candes (2023).
    
    Args:
        n (int): Number of samples to generate.
        sig (float): Noise scaling factor. Defaults to 0.1.
        dim (int, optional): Dimensionality of the feature space. Defaults to 20.
        random_state (int or np.random.Generator, optional): Random seed or generator for reproducible samples.
        
    Returns:
        tuple: A tuple (X, mu_x, eps, Y) representing the generated data and components.
    """
    rng = _get_rng(random_state)

    X = rng.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
    mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
    eps = rng.normal(size=n) * (5.5 - abs(mu_x)) / 2 * sig
    Y = mu_x + eps
    return X, mu_x, eps, Y

def gen_data_s1(n, sig=1, dim=1, random_state=None):
    """Generate the two-component Gaussian-mixture DGP used in S1.

    The first feature follows a standard normal distribution. Conditional on
    a Bernoulli component indicator, the outcome mean is either
    ``-1 + x + x**2`` or ``1 - 2*x``, with common standard deviation
    ``1.5 * sig``. Setting ``dim=1`` exactly matches the feature dimension in
    the R simulations; larger values add independent nuisance features.

    Args:
        n (int): Number of samples to generate.
        sig (float, optional): Noise scaling factor. Defaults to 1.
        dim (int, optional): Number of feature columns. Defaults to 1.
        random_state (int or np.random.Generator, optional): Random seed or
            generator for reproducible samples.

    Returns:
        tuple: ``(X, mu_x, eps, Y)`` containing features, component-specific
        means, noise, and outcomes.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1 for S1 data generation.")

    rng = _get_rng(random_state)
    X = rng.normal(size=n * dim).reshape((n, dim))
    x = X[:, 0]
    z = rng.binomial(1, 0.4, size=n)

    mu_1 = -1 + x + x ** 2
    mu_2 = 1 - 2 * x
    mu_x = np.where(z == 1, mu_1, mu_2)
    eps = rng.normal(size=n) * 1.5 * sig
    Y = mu_x + eps
    return X, mu_x, eps, Y

def gen_data_group(n, sig=1, dim=20, random_state=None):
    """Generates high-dimensional artificial data with a group structure.

    The group indicator ``G`` is obtained by cutting the standard-normal
    first column of ``X`` into four regions with probabilities
    ``(0.05, 0.35, 0.25, 0.35)``. The sparse linear covariate effect uses the
    next three columns of ``X``.
    
    Args:
        n (int): Number of samples to generate.
        sig (float, optional): Noise scaling factor. Defaults to 1.
        dim (int, optional): Dimensionality of the feature space. Defaults to 20.
        random_state (int or np.random.Generator, optional): Random seed or generator for reproducible samples.
        
    Returns:
        tuple: A tuple (X, mu_x, eps, Y) representing the generated data and components.
    """
    rng = _get_rng(random_state)

    if dim < 4:
        raise ValueError("dim must be at least 4 for group structure generation.")

    X = rng.normal(size=n*dim).reshape((n,dim)) # X matrix

    # Standard-normal quantiles at cumulative probabilities 0.05, 0.40, 0.65.
    group_cutoffs = np.array([-1.64485363, -0.25334710, 0.38532047])
    G = np.digitize(X[:, 0], group_cutoffs)
    X[:, 0] = G # replace the first column with group indicator

    lin_eff = (0.25 * X[:, 1] - 0.20 * X[:, 2] + 0.15 * X[:, 3])
    group_mu = np.array([5.0, 2.4, 1.0, -0.5])
    group_sigma = np.array([0.5, 3.0, 0.25, 1.0]) * sig

    mu_x = group_mu[G] + lin_eff
    sigma_x = group_sigma[G]
    eps = rng.normal(size=n) * sigma_x
    Y = mu_x + eps
    return X, mu_x, eps, Y

def gen_data_mix(n, sig=1, dim=20, random_state=None):
    """Generates artificial data using mixture models with x-dependent mixture probability. 
    
    Args:
        n (int): Number of samples to generate.
        sig (float, optional): Noise scaling factor. Defaults to 1.
        dim (int, optional): Dimensionality of the feature space. Defaults to 20.
        random_state (int or np.random.Generator, optional): Random seed or generator for reproducible samples.
        
    Returns:
        tuple: A tuple (X, mu_x, eps, Y) representing the generated data and components.
    """
    rng = _get_rng(random_state)

    X = rng.normal(size=n*dim).reshape((n,dim)) # X matrix
    pi_x = _expit(4 * (X[:, 0] - 0.25))
    z = rng.binomial(1, pi_x)
    mu_H, sigma_H = 2.7 + 0.5*X[:, 1] + 0.2*X[:, 2]**2, 3.0
    mu_L, sigma_L = 0.7 - 1.4*X[:, 4], 0.30
    #mu_H, sigma_H = 2.7 + 0.5*X[:, 0] + 0.2*X[:, 0]**2, 3.0
    #mu_L, sigma_L = 0.7 - 1.4*X[:, 0], 0.30

    mu_x = np.where(z == 1, mu_H, mu_L)
    sigma_x = np.where(z == 1, sigma_H, sigma_L) * sig
    eps = rng.normal(size=n) * sigma_x
    Y = mu_x + eps

    return X, mu_x, eps, Y

# BH procedure
def BH(pvals, q):
    """Applies the Benjamini-Hochberg (BH) procedure to a list of p-values.
    
    Args:
        pvals (array-like): List or array of p-values.
        q (float): The nominal False Discovery Rate (FDR) level.
        
    Returns:
        np.ndarray: The indices forming the rejection set.
    """
    pvals = np.asarray(pvals, dtype=float)
    ntest = pvals.size

    if ntest == 0:
        return np.array([], dtype=int)

    order = np.argsort(pvals, kind="mergesort")
    sorted_pvals = pvals[order]
    thresholds = q * np.arange(1, ntest + 1) / ntest
    selected = np.flatnonzero(sorted_pvals <= thresholds)

    if selected.size == 0:
        return np.array([], dtype=int)

    return order[: selected[-1] + 1]

# evaluation functions
def eval_MDR(L, R, sel):
    """Evaluates selection performance for risk and power in the MDR sense.
    
    Args:
        L (np.ndarray): The true loss corresponding to every instance.
        R (np.ndarray): The true rewards for each instance.
        sel (array-like): The selection set generated by the test procedure.
        
    Returns:
        tuple: (risk_acc, reward_acc) indicating the MDR risk and cumulative reward.
    """
    if len(sel) == 0:
        return 0, 0
    risk_acc = np.sum(L[sel]) / len(L)
    reward_acc = np.sum(R[sel])
    return risk_acc, reward_acc

def eval_SDR(L, R, sel):
    """Evaluates selection performance for risk and power in the SDR sense.
    
    Args:
        L (np.ndarray): The true loss corresponding to every instance.
        R (np.ndarray): The true rewards for each instance.
        sel (array-like): The selection set generated by the test procedure.
        
    Returns:
        tuple: (sdr, bin_power, reward) corresponding to SDR, binary power (equivalent to power in the binary loss case), and reward metrics.
    """
    if len(sel) == 0:
        return 0, 0, 0

    sdr = np.sum(L[sel]) / len(sel)
    # binary power is defined only for 0-1 loss
    true_rej = len(L) - np.sum(L) # number of zeros in L
    bin_power = (len(sel) - np.sum(L[sel])) / true_rej if true_rej != 0 else 0 # defined only for 0-1 loss
    # reward is more general, can be used for any loss function
    reward = np.sum(R[sel])
    return sdr, bin_power, reward
