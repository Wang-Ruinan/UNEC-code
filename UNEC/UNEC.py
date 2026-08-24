import numpy as np
from scipy.special import ndtr
from .utility import BH

# p-values from Jin2023
def _as_index_array(sel):
    return np.asarray(sel, dtype=int)

def _as_1d_array(name, values):
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return arr

def _split_calib(Dcalib):
    if not isinstance(Dcalib, (tuple, list)) or len(Dcalib) != 2:
        raise ValueError("Dcalib must be a tuple or list of losses and scores (Lcalib, Scalib).")

    Lcalib = _as_1d_array("Lcalib", Dcalib[0])
    Scalib = _as_1d_array("Scalib", Dcalib[1])
    if len(Lcalib) == 0:
            raise ValueError("The calibration set cannot be empty.")
    if len(Lcalib) != len(Scalib):
        raise ValueError("The losses and scores (Lcalib, Scalib) must have the same length.")
    return Lcalib, Scalib

def _is_legacy_dtest(Dtest):
    if not isinstance(Dtest, (tuple, list)) or len(Dtest) != 2:
        return False
    if np.ndim(Dtest[1]) == 0:
        return False
    return Dtest[0] is None or np.ndim(Dtest[0]) > 0

def _get_stest(Dtest):
    if _is_legacy_dtest(Dtest):
        Dtest = Dtest[1]
    return _as_1d_array("Dtest", Dtest)

def _validate_binary_loss(Lcalib):
    if not np.all(np.isin(Lcalib, [0, 1])):
        raise ValueError("Conformal selection requires binary calibration losses in {0, 1}.")


def _validate_alpha(alpha):
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    return alpha

def CS(Dcalib, Dtest, alpha, mult_test=True, return_pvals=False):
    """Conformal Selection (CS) procedure for binary losses that controls the marginal deployment risk (MDR) or selective deployment risk (SDR).
    Here, MDR reduces to the average type-I error and SDR reduces to the usual false discovery rate (FDR).
    
    The function applies only when the loss function evaluates strictly to {0,1}.
    
    Args:
        Dcalib (tuple): A tuple containing losses and scores (Lcalib, Scalib) for the calibration set.
        Dtest (array-like): Test scores Stest. A legacy tuple/list (ignored, Stest) is also accepted.
        alpha (float): The target error margin.
        mult_test (bool): Whether to perform multiple testing correction using the Benjamini-Hochberg (BH) procedure. If False, MDR is controlled; otherwise SDR is controlled.
        return_pvals (bool): If True, returns the calculated p-values alongside the selected indices.
        
    Returns:
        Union[np.ndarray, tuple]: Selected indices, or (selected indices, p-values) if return_pvals is True.
    """
    alpha = _validate_alpha(alpha)
    Lcalib, Scalib = _split_calib(Dcalib)
    _validate_binary_loss(Lcalib)
    Stest = _get_stest(Dtest)
    Ncalib, Ntest = len(Scalib), len(Stest)

    calib_scores = 1000 * (Lcalib == 0) + Scalib
    test_scores = Stest
    
    pvals = np.zeros(Ntest)
    for j in range(Ntest):
        pvals[j] = (1 + np.sum(calib_scores <= test_scores[j])) / (Ncalib + 1)

    if mult_test:
        sel = BH(pvals, alpha)
    else:
        sel = np.flatnonzero(pvals <= alpha)

    if not return_pvals:
        return _as_index_array(sel)
    return sel, pvals


# UNEC
def _validate_delta(delta):
    delta = float(delta)
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be a non-negative number")
    return delta

def UNEC_SDR(Dcalib, Dtest, alpha, delta=1, return_info=False):
    """Select test samples with the Sheridan score calibration rule.

    Larger scores are treated as stronger evidence that ``Y > c_val``.

    Args:
        Dcalib (tuple) : A tuple ``(Lcalib, Scalib)`` containing calibration outcomes and scores.
        Dtest (array-like): Test scores.
        alpha (float): Target false discovery rate in ``(0, 1]``.
        delta (float, default=1): Finite-sample correction added to the cumulative false alarms.
        return_info (bool): If True, returns additional information including the threshold and the SDR curve.

    Returns:
        Union[np.ndarray, tuple]: Selected indices, or (selected indices, threshold, sdr_curve) if return_info is True.
    """

    alpha = _validate_alpha(alpha)
    delta = _validate_delta(delta)
    Lcalib, Scalib = _split_calib(Dcalib)
    Stest = _get_stest(Dtest)

    if not np.all(np.isfinite(Scalib)) or not np.all(np.isfinite(Stest)):
        raise ValueError("Calibration and test scores must be finite.")

    ## SDR control ##
    # Compute the cumulative selective discovery rate curve
    order = np.argsort(-Scalib, kind="mergesort") # from largest to smallest
    sorted_scores = Scalib[order]
    sorted_losses = Lcalib[order]

    sdr_curve = (np.cumsum(sorted_losses) + delta) / np.arange(1, len(Lcalib) + 1)

    valid = np.flatnonzero(sdr_curve <= alpha)
    threshold = np.inf if valid.size == 0 else sorted_scores[valid[-1]]
    selected = np.flatnonzero(Stest >= threshold)

    if return_info:
        return selected, threshold, sdr_curve
    return selected


def UNEC_MDR(Dcalib, Dtest, alpha, delta=1, return_info=False):
    """Select test samples using a calibration estimate of MDR.

    Larger scores are treated as indicating safer or more desirable samples.

    Args:
        Dcalib (tuple): ``(Lcalib, Scalib)`` containing calibration losses and scores.
        Dtest (array-like): Test scores.
        alpha (float): Target marginal deployment risk in ``(0, 1]``.
        delta (float): Finite-sample correction added to cumulative losses.
        return_info (bool): If True, also return the threshold and MDR curve.

    Returns:
        Union[np.ndarray, tuple]: Selected indices, or
        ``(selected indices, threshold, mdr_curve)`` when ``return_info=True``.
    """
    alpha = _validate_alpha(alpha)
    delta = _validate_delta(delta)
    Lcalib, Scalib = _split_calib(Dcalib)
    Stest = _get_stest(Dtest)

    if not np.all(np.isfinite(Lcalib)):
        raise ValueError("Calibration losses must be finite.")
    if not np.all(np.isfinite(Scalib)) or not np.all(np.isfinite(Stest)):
        raise ValueError("Calibration and test scores must be finite.")

    ## MDR control ##
    # Compute the cumulative marginal discovery rate curve
    order = np.argsort(-Scalib, kind="mergesort")
    sorted_scores = Scalib[order]
    sorted_losses = Lcalib[order]

    ncalib = len(Lcalib)
    mdr_curve = (np.cumsum(sorted_losses) + delta) / ncalib

    valid = np.flatnonzero(mdr_curve <= alpha)
    threshold = np.inf if valid.size == 0 else sorted_scores[valid[-1]]
    selected = np.flatnonzero(Stest >= threshold)

    if return_info:
        return selected, threshold, mdr_curve
    return selected

# Sheridan score function
def _validate_min_scale(min_scale):
    min_scale = float(min_scale)
    if not np.isfinite(min_scale) or min_scale <= 0:
        raise ValueError("min_scale must be a positive finite number.")
    return min_scale

def sheridan_score(X, mean_model, scale_model, c_val, min_scale=0.01, squared_scale=False):
    """Compute the Sheridan score ``(mu_hat(X) - c) / sigma_hat(X)``.

    Args:
        X (array-like): Feature matrix used for prediction.
        mean_model: Fitted model with a ``predict`` method for the mean.
        scale_model: Fitted model with a ``predict`` method for the scale or squared scale.
        c_val (float): Outcome thresholds.
        min_scale (float): Lower bound for predicted scales to avoid division by zero. Defaults to 0.01.
        squared_scale (bool): If True, ``scale_model`` is treated as predicting squared residuals and its predictions are square-rooted. Defaults to False.

    Returns:
        np.ndarray: One Sheridan score for each row of ``X``. Larger scores indicate stronger evidence that the outcome is above ``c_val``.
    """
    min_scale = _validate_min_scale(min_scale)

    pred_mu = np.asarray(mean_model.predict(X), dtype=float).reshape(-1)
    pred_scale = np.asarray(scale_model.predict(X), dtype=float).reshape(-1)

    if len(pred_mu) != len(pred_scale):
        raise ValueError("Mean and scale models must return the same number of predictions.")

    if squared_scale:
        pred_scale = np.sqrt(np.maximum(pred_scale, min_scale ** 2))
    else:
        pred_scale = np.maximum(pred_scale, min_scale)
    return (pred_mu - c_val) / pred_scale

def mixture_sheridan_score(mean_model, scale_model, X, c_val, min_scale=0.01):
    """Compute the Sheridan score from Gaussian-mixture predictions.

    Args:
        mean_model: Fitted model with a ``predict`` method for the mean.
        scale_model: Fitted model with a ``predict`` method for the scale.
        X (array-like): Feature matrix used for prediction.
        c_val (float): Outcome thresholds.
        min_scale (float): Lower bound for the aggregated conditional scale to avoid division by zero. Defaults to 0.01.

    Returns:
        np.ndarray: One Sheridan score for each sample. Larger scores indicate stronger evidence that the outcome is above ``c_val``.
    """
    min_scale = _validate_min_scale(min_scale)
    pred_mu = np.asarray(mean_model.predict(X, aggregate=True),dtype=float).reshape(-1)
    pred_scale = np.asarray(scale_model.predict(X),dtype=float).reshape(-1)

    if len(pred_mu) != len(pred_scale):
        raise ValueError("Mean and scale models must return the same number of predictions.")

    pred_scale = np.maximum(pred_scale,min_scale)
    return (pred_mu - c_val) / pred_scale

# Neyman-Pearson selection
def NP(Dcalib, Dtest, alpha, delta=1, return_info=False):
    """Select test samples using the NP-odds calibration rule from Qin et al.(2025).

    Smaller scores indicate lower predicted loss. For binary loss, the NP
    score can be constructed as P(Y <= c | X) / P(Y > c | X).

    Args:
        Dcalib (tuple): ``(Lcalib, Scalib)`` containing calibration losses and NP scores.
        Dtest (array-like): Test NP scores.
        alpha (float): Target selective deployment risk in ``(0, 1]``.
        delta (float): Finite-sample correction added to cumulative losses.
        return_info (bool): If True, also return the threshold and FDR curve.

    Returns:
        Union[np.ndarray, tuple]: Selected test indices, or
        ``(selected, threshold, fdr_curve)`` if ``return_info=True``.
    """
    alpha = _validate_alpha(alpha)
    delta = _validate_delta(delta)
    Lcalib, Scalib = _split_calib(Dcalib)
    Stest = _get_stest(Dtest)

    if not np.all(np.isfinite(Lcalib)):
        raise ValueError("Calibration losses must be finite.")
    if not np.all(np.isfinite(Scalib)) or not np.all(np.isfinite(Stest)):
        raise ValueError("Calibration and test scores must be finite.")

    # Smaller NP scores are selected first.
    order = np.argsort(Scalib, kind="mergesort") # from smallest to largest
    sorted_scores = Scalib[order]
    sorted_losses = Lcalib[order]

    fdr_curve = (np.cumsum(sorted_losses) + delta) / np.arange(1, len(Lcalib) + 1)

    valid = np.flatnonzero(fdr_curve <= alpha)
    threshold = -np.inf if valid.size == 0 else sorted_scores[valid[-1]]
    selected = np.flatnonzero(Stest <= threshold)

    if return_info:
        return selected, threshold, fdr_curve
    return selected

# Neyman-Pearson score
def _validate_prob_clip(prob_clip):
    prob_clip = float(prob_clip)
    if not np.isfinite(prob_clip) or prob_clip <= 0 or prob_clip >= 0.5:
        raise ValueError("prob_clip must be a finite number in (0, 0.5).")
    return prob_clip

def NP_score(X, mean_model, c_val, residual_scale, min_scale=0.01, prob_clip=1e-12):
    """Compute the Gaussian NP odds score.

    The score is
    ``P(Y <= c_val | X) / P(Y > c_val | X)`` under a Gaussian working
    model. Smaller scores indicate stronger evidence that the outcome is
    above ``c_val``.

    Args:
        X (array-like): Feature matrix used for prediction.
        mean_model: Fitted model with a ``predict`` method for the mean.
        c_val (float): Outcome threshold.
        residual_scale (float or array-like): Estimated residual standard deviation. It may be a common scalar or one value per row of ``X``.
        min_scale (float): Lower bound for the residual scale. Defaults to 0.01.
        prob_clip (float): Lower and upper probability clipping constant. Defaults to 1e-12.

    Returns:
        np.ndarray: One NP odds score for each row of ``X``.
    """
    min_scale = _validate_min_scale(min_scale)
    prob_clip = _validate_prob_clip(prob_clip)

    pred_mu = np.asarray(mean_model.predict(X), dtype=float).reshape(-1)
    if not np.all(np.isfinite(pred_mu)):
        raise ValueError("The mean model must return finite predictions.")

    pred_scale = np.asarray(residual_scale, dtype=float)
    if pred_scale.ndim == 0:
        pred_scale = np.full(len(pred_mu), float(pred_scale))
    else:
        pred_scale = pred_scale.reshape(-1)
        if len(pred_scale) != len(pred_mu):
            raise ValueError("residual_scale must be a scalar or have one value per row of X.")
    if not np.all(np.isfinite(pred_scale)):
        raise ValueError("residual_scale must contain only finite values.")

    pred_scale = np.maximum(pred_scale, min_scale)
    lower_tail_prob = ndtr((float(c_val) - pred_mu) / pred_scale)
    lower_tail_prob = np.clip(lower_tail_prob, prob_clip, 1 - prob_clip)
    return lower_tail_prob / (1 - lower_tail_prob)
