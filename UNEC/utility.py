import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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


def loss_two_sided(Y, tau, a, b):
    """Calculates the smoothened two-sided indicator loss.

    The loss is the sum of the smooth lower-tail and upper-tail indicators:
    ``sigmoid(-tau * (Y - a)) + sigmoid(tau * (Y - b))``. It is close to
    zero when ``Y`` is inside ``(a, b)`` and close to one when ``Y`` is
    outside the interval.

    Args:
        Y (np.ndarray): The target values.
        tau (float): Hyperparameter for smoothing. Larger values make the
            loss closer to ``1{Y <= a or Y >= b}``. If ``tau=np.inf``, the
            exact two-sided indicator loss is returned.
        a (float or np.ndarray): Lower endpoint or observation-specific lower
            endpoints of the target interval.
        b (float or np.ndarray): Upper endpoint or observation-specific upper
            endpoints of the target interval. Every value must be larger than
            the corresponding value of ``a``.

    Returns:
        np.ndarray: The computed smoothened two-sided loss values.
    """
    Y = np.asarray(Y)
    a = np.asarray(a)
    b = np.asarray(b)
    if np.any(a >= b):
        raise ValueError("Every lower endpoint a must be smaller than b.")
    if tau == np.inf:
        return ((Y <= a) | (Y >= b)).astype(float)
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite number or np.inf.")
    lower_tail = _expit(-tau * (Y - a))
    upper_tail = _expit(tau * (Y - b))
    return lower_tail + upper_tail


def fit_mean_model(model_type="rf", random_state=None, min_samples_leaf=10):
    """Construct a mean-regression model used by the simulation scripts."""
    if model_type == "linear":
        return LinearRegression()
    if model_type == "rf":
        return RandomForestRegressor(
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
        )
    raise ValueError("model_type must be 'linear' or 'rf'.")

# Data generation function
def gen_data_Jin2023(n, sig=0.1, dim=20, random_state=None, sigma_type=2):
    """Generates artificial data using the data generation process in Jin and Candes (2023).
    
    Args:
        n (int): Number of samples to generate.
        sig (float): Noise scaling factor. Defaults to 0.1.
        dim (int, optional): Dimensionality of the feature space. Defaults to 20.
        random_state (int or np.random.Generator, optional): Random seed or generator for reproducible samples.
        sigma_type (int, optional): Conditional noise-scale construction:
            1 uses the constant ``1.5``; 2 uses ``(5.5 - |mu|) / 2``; and 3 uses ``0.25 * mu**2 * 1{|mu| < 2} + 0.5 * |mu| * 1{|mu| >= 1}``. 
            Each construction is multiplied by ``sig``. Defaults to 2.
        
    Returns:
        tuple: A tuple (X, mu_x, eps, Y) representing the generated data and components.
    """
    rng = _get_rng(random_state)

    X = rng.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
    mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
    abs_mu = np.abs(mu_x)
    if sigma_type == 1:
        sigma_x = np.full(n, 1.5) * sig
    elif sigma_type == 2:
        sigma_x = (5.5 - abs_mu) / 2 * sig
    elif sigma_type == 3:
        sigma_x = (0.25 * mu_x ** 2 * (abs_mu < 2) + 0.5 * abs_mu * (abs_mu >= 1)) * sig
    else:
        raise ValueError("sigma_type must be 1, 2, or 3.")

    eps = rng.normal(size=n) * sigma_x
    Y = mu_x + eps
    return X, mu_x, eps, Y

def gen_data_mix(n, sig=1, dim=20, random_state=None, mix_type=1):
    """Generate one of two two-component Gaussian-mixture models.
    
    Args:
        n (int): Number of samples to generate.
        sig (float, optional): Noise scaling factor. Defaults to 1.
        dim (int, optional): Dimensionality of the feature space. Defaults to 20.
        random_state (int or np.random.Generator, optional): Random seed or generator for reproducible samples.
        mix_type (int, optional): Mixture construction. Type 1 is the standard covariate-dependent mixture, and type 2 is the complex nonlinear heteroscedastic mixture. 
            Defaults to 1.
        
    Returns:
        tuple: A tuple (X, mu_x, eps, Y) representing the generated data and components.
    """
    rng = _get_rng(random_state)
    required_dim = {1: 5, 2: 6}
    if mix_type not in required_dim:
        raise ValueError("mix_type must be 1 or 2.")
    if dim < required_dim[mix_type]:
        raise ValueError(
            f"dim must be at least {required_dim[mix_type]} for mix_type={mix_type}."
        )

    X = rng.normal(size=n * dim).reshape((n, dim))

    if mix_type == 1:
        pi_x = _expit(4 * (X[:, 0] - 0.25))
        z = rng.binomial(1, pi_x)
        mu_H = 2.7 + 0.5 * X[:, 1] + 0.2 * X[:, 2] ** 2
        mu_L = 0.7 - 1.4 * X[:, 4]
        sigma_H, sigma_L = 3.0, 0.30
    else:  # mix_type == 2
        pi_x = _expit(2.0 * X[:, 0] + X[:, 1] * X[:, 2] - 0.5)
        z = rng.binomial(1, pi_x)
        mu_H = (2.7 + 0.5 * X[:, 1] + 0.3 * X[:, 2] ** 2 + 0.3 * X[:, 3] * X[:, 4])
        mu_L = 0.7 - 1.4 * X[:, 4] + 0.3 * np.sin(X[:, 5])
        sigma_H = 2.0 + _expit(X[:, 1] + X[:, 2])
        sigma_L = 0.25 + 0.20 * np.abs(X[:, 4])

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
