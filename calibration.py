"""
calibration.py — Post-training temperature scaling for MC Dropout CI.

Temperature scaling adjusts the width of uncertainty intervals
without touching model weights. It finds a scalar T on validation
data such that coverage matches the target (0.80 by default).

This is mathematically correct — it's the standard post-hoc
calibration method from Guo et al. (2017).
"""

import numpy as np
from scipy.optimize import minimize_scalar


def find_temperature(mc_std, y_true, mc_mean, target_coverage=0.80):
    """
    Find temperature T such that:
        coverage(mc_mean ± T * mc_std) ≈ target_coverage

    Args:
        mc_std:           (N,) epistemic std from MC Dropout
        y_true:           (N,) ground truth
        mc_mean:          (N,) predicted mean
        target_coverage:  float, target fraction to cover (default 0.80)

    Returns:
        T: float scalar — multiply mc_std by T at inference
    """
    def coverage_error(T):
        lower = mc_mean - T * mc_std
        upper = mc_mean + T * mc_std
        cov   = np.mean((y_true >= lower) & (y_true <= upper))
        return (cov - target_coverage) ** 2

    result = minimize_scalar(coverage_error, bounds=(0.1, 20.0), method="bounded")
    T      = float(result.x)
    lower  = mc_mean - T * mc_std
    upper  = mc_mean + T * mc_std
    actual = float(np.mean((y_true >= lower) & (y_true <= upper)))

    print(f"  Temperature T = {T:.4f}")
    print(f"  Coverage before: {np.mean((y_true >= mc_mean - mc_std) & (y_true <= mc_mean + mc_std)):.4f}")
    print(f"  Coverage after:  {actual:.4f}  (target: {target_coverage})")
    return T


def apply_temperature(mc_std, T):
    """Scale uncertainty by temperature at inference time."""
    return mc_std * T