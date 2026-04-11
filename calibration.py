"""
calibration.py — CrashGuard AI Post-Training Calibration

PURPOSE:
    Temperature scaling finds a scalar T on validation data such that
    coverage matches target (0.80) without retraining the model.

    This is the standard post-hoc calibration method.
    Reference: Guo et al. (2017), "On Calibration of Modern Neural Networks"

USAGE:
    Called automatically at end of train.py
    Saves T to calibration_temperature.pkl
    Loaded in evaluate.py and live monitor
"""

import numpy as np
import joblib
import os
from scipy.optimize import minimize_scalar


def find_temperature(mc_std, y_true, mc_mean, target_coverage=0.80):
    """
    Find temperature T such that:
        coverage(mc_mean ± T * mc_std) ≈ target_coverage

    Args:
        mc_std:          (N,) epistemic std from MC Dropout (inverse transformed)
        y_true:          (N,) ground truth (inverse transformed)
        mc_mean:         (N,) predicted mean (inverse transformed)
        target_coverage: float, default 0.80

    Returns:
        T: float — multiply mc_std by T at inference time
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

    before_lower = mc_mean - mc_std
    before_upper = mc_mean + mc_std
    before_cov   = float(np.mean((y_true >= before_lower) & (y_true <= before_upper)))

    print(f"  Temperature T       = {T:.4f}")
    print(f"  Coverage before T   = {before_cov:.4f}")
    print(f"  Coverage after  T   = {actual:.4f}  (target: {target_coverage})")

    if T > 10:
        print(f"  Warning: T={T:.2f} is high — MC Dropout uncertainty is near zero")
        print(f"  This means model is overconfident — dropout rate may need increase")

    return T


def apply_temperature(mc_std, T):
    """Scale uncertainty by temperature at inference time."""
    return mc_std * T


def save_temperature(T, path="calibration_temperature.pkl"):
    joblib.dump(T, path)
    print(f"  Temperature saved: {path}")


def load_temperature(path="calibration_temperature.pkl"):
    if os.path.exists(path):
        T = joblib.load(path)
        print(f"  Temperature loaded: T={T:.4f}")
        return T
    print(f"  No temperature file found at {path} — using T=1.0 (no scaling)")
    return 1.0
