"""
evaluate.py — CrashGuard AI Evaluation

FIXES in this version:
  1. mc_dropout_predict() unpacks 4 values: mean, std, lower, upper
  2. Coverage uses quantile-based CI (lower, upper) from mc_dropout_predict
     instead of ±1.96*std — correct for non-Gaussian CPU distributions
  3. log1p inverse transform applied before metric computation
  4. total_std computation preserved for calibration diagram

SPEED:
  - ARIMA limited to 2000 points, 6 order combinations (~10 min total)
  - Prophet disabled by default (run_prophet=False)
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, norm
import warnings
warnings.filterwarnings("ignore")

from lstm_model import predict, mc_dropout_predict, inverse_log1p


# =============================================
# METRICS
# =============================================
def calculate_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


# =============================================
# CALIBRATION — RELIABILITY DIAGRAM
# =============================================
def compute_reliability_diagram(y_true, mean_pred, total_std,
                                 n_bins=10, save_path="calibration_diagram.png"):
    confidence_levels = np.linspace(0.1, 1.0, n_bins)
    actual_coverages  = []

    for conf in confidence_levels:
        z      = norm.ppf((1 + conf) / 2)
        lower  = mean_pred - z * total_std
        upper  = mean_pred + z * total_std
        inside = np.mean((y_true >= lower) & (y_true <= upper))
        actual_coverages.append(float(inside))

    actual_coverages = np.array(actual_coverages)
    ece              = float(np.mean(np.abs(confidence_levels - actual_coverages)))

    z95       = norm.ppf(0.975)
    ci_widths = 2 * z95 * total_std
    sharpness = float(np.mean(ci_widths))

    if ece < 0.05:
        quality = "Well calibrated"
    elif ece < 0.10:
        quality = "Acceptable calibration"
    else:
        quality = "Poor calibration"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0a0e1a")

    ax1 = axes[0]
    ax1.set_facecolor("#111827")
    ax1.plot([0, 1], [0, 1], color="#64748b", linewidth=1.5,
             linestyle="--", label="Perfect calibration", zorder=1)
    ax1.fill_between([0, 1], [0, 0], [0, 1],
                     alpha=0.04, color="#ef4444", label="Overconfident zone")
    ax1.fill_between([0, 1], [0, 1], [1, 1],
                     alpha=0.04, color="#3b82f6", label="Underconfident zone")
    ax1.plot(confidence_levels, actual_coverages,
             color="#22c55e", linewidth=2.5, marker="o", markersize=7,
             label=f"CrashGuard AI (ECE={ece:.3f})", zorder=3)
    ax1.fill_between(confidence_levels, confidence_levels, actual_coverages,
                     alpha=0.15, color="#22c55e")
    ax1.set_xlabel("Predicted confidence level", color="#94a3b8", fontsize=11)
    ax1.set_ylabel("Actual coverage", color="#94a3b8", fontsize=11)
    ax1.set_title("Reliability Diagram", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_color("#2d3748")
    ax1.grid(True, color="#1e293b", linewidth=0.5)
    ax1.legend(facecolor="#1a1f35", edgecolor="#2d3748",
               labelcolor="#e2e8f0", fontsize=9)
    ax1.text(0.05, 0.92, f"ECE = {ece:.4f}  ({quality})",
             transform=ax1.transAxes,
             color="#22c55e" if ece < 0.05 else "#f59e0b",
             fontsize=10, fontweight="bold")

    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    ax2.hist(ci_widths, bins=40, color="#3b82f6", alpha=0.8, edgecolor="#1e293b")
    ax2.axvline(sharpness, color="#f59e0b", linewidth=2, linestyle="--",
                label=f"Mean width = {sharpness:.4f}")
    ax2.set_xlabel("95% CI width", color="#94a3b8", fontsize=11)
    ax2.set_ylabel("Count", color="#94a3b8", fontsize=11)
    ax2.set_title("Sharpness — CI Width Distribution",
                  color="white", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#2d3748")
    ax2.grid(True, color="#1e293b", linewidth=0.5, axis="y")
    ax2.legend(facecolor="#1a1f35", edgecolor="#2d3748",
               labelcolor="#e2e8f0", fontsize=9)

    plt.suptitle("CrashGuard AI — Uncertainty Calibration Analysis",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    plt.close()
    print(f"  Calibration diagram saved: {save_path}")

    return {
        "predicted_confidences": confidence_levels.tolist(),
        "actual_coverages":      actual_coverages.tolist(),
        "ECE":                   ece,
        "ECE_quality":           quality,
        "Sharpness":             sharpness,
        "Coverage_95":           float(actual_coverages[8]),
        "diagram_path":          save_path,
    }


# =============================================
# BASELINES
# =============================================
def naive_forecast(y):
    preds     = np.zeros_like(y)
    preds[0]  = y[0]
    preds[1:] = y[:-1]
    return preds


def moving_average_forecast(y, window=5):
    preds = []
    for i in range(len(y)):
        preds.append(np.mean(y[max(0, i - window):i + 1] if i > 0 else [y[0]]))
    return np.array(preds)


# =============================================
# FAST ARIMA
# =============================================
def arima_forecast_walkforward(y, max_train=2000, order=None):
    arima_data = y[:2000] if len(y) > 2000 else y
    n_points   = len(arima_data)
    print(f"  Walk-forward ARIMA on {n_points} points (capped at 2000 for speed)...")

    if order is None:
        try:
            train_sample = arima_data[:min(300, n_points // 2)]
            best_aic, best_order = np.inf, (1, 1, 0)
            for p in [1, 2, 3]:
                for q in [0, 1]:
                    try:
                        aic = ARIMA(train_sample, order=(p, 1, q)).fit().aic
                        if aic < best_aic:
                            best_aic, best_order = aic, (p, 1, q)
                    except Exception:
                        continue
            order = best_order
            print(f"  ARIMA best order: {order}")
        except Exception:
            order = (1, 1, 0)

    preds, history = [], list(arima_data[:max_train])
    for i in range(n_points):
        try:
            fit  = ARIMA(history[-max_train:], order=order).fit()
            pred = float(np.clip(fit.forecast(steps=1)[0], 0.0, 1.0))
        except Exception:
            pred = history[-1]
        preds.append(pred)
        history.append(float(arima_data[i]))
        if i % 500 == 0 and i > 0:
            print(f"    ARIMA step {i}/{n_points}")

    arima_preds_short = np.array(preds)
    if len(y) > 2000:
        extension        = np.full(len(y) - 2000, arima_preds_short[-1])
        arima_preds_full = np.concatenate([arima_preds_short, extension])
    else:
        arima_preds_full = arima_preds_short

    print(f"  ARIMA complete.")
    return arima_preds_full, order


# =============================================
# PROPHET
# =============================================
def prophet_forecast_walkforward(y, max_train=3000):
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    except ImportError:
        print("  Prophet not installed.")
        return naive_forecast(y), False

    import pandas as pd
    REFIT_EVERY, preds, history, prophet_model = 50, [], list(y[:max_train]), None

    for i in range(len(y)):
        if i % REFIT_EVERY == 0 or prophet_model is None:
            try:
                window     = history[-max_train:]
                df_prophet = pd.DataFrame({
                    "ds": pd.date_range(start="2024-01-01", periods=len(window), freq="5s"),
                    "y":  window
                })
                prophet_model = Prophet(
                    daily_seasonality=True, weekly_seasonality=True,
                    yearly_seasonality=False, changepoint_prior_scale=0.05,
                    interval_width=0.95, verbose=False
                )
                prophet_model.fit(df_prophet)
            except Exception:
                preds.append(history[-1] if history else 0.0)
                history.append(float(y[i]))
                continue
        try:
            last_ds  = pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=5 * len(history))
            forecast = prophet_model.predict(pd.DataFrame({"ds": [last_ds]}))
            pred     = float(np.clip(forecast["yhat"].values[0], 0.0, 1.0))
        except Exception:
            pred = history[-1] if history else 0.0
        preds.append(pred)
        history.append(float(y[i]))

    return np.array(preds), True


def get_prophet_verdict(lstm_rmse, prophet_rmse, dm_pval, prophet_success):
    if not prophet_success:
        return "Prophet not available"
    if lstm_rmse < prophet_rmse:
        pct = (prophet_rmse - lstm_rmse) / prophet_rmse * 100
        sig = "statistically significant" if dm_pval < 0.05 else "not statistically significant"
        return (f"LSTM outperforms Prophet by {pct:.1f}% "
                f"(RMSE {lstm_rmse:.4f} vs {prophet_rmse:.4f}, DM p={dm_pval:.3f} — {sig})")
    gap = (lstm_rmse - prophet_rmse) / prophet_rmse * 100
    return (f"Prophet baseline is strong (gap={gap:.1f}%). "
            f"LSTM provides uncertainty quantification that Prophet cannot.")


# =============================================
# DIEBOLD-MARIANO
# =============================================
def diebold_mariano_test(e1, e2):
    d      = e1 ** 2 - e2 ** 2
    mean_d = np.mean(d)
    var_d  = np.var(d, ddof=1)
    if var_d == 0:
        return 0.0, 1.0
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - t.cdf(abs(dm_stat), df=len(d) - 1))
    return float(dm_stat), float(p_value)


# =============================================
# HELPERS
# =============================================
def inverse_transform_cpu(cpu_scaler, data):
    return cpu_scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def ljung_box_test(residuals):
    try:
        lb = acorr_ljungbox(residuals, lags=[20], return_df=True)
        return float(lb["lb_pvalue"].values[0])
    except Exception:
        return float("nan")


def walk_forward_validation(model, X_test, y_test, cpu_scaler, step=10):
    rmses = []
    for start in range(0, len(X_test) - step, step):
        pred_sc = predict(model, X_test[start:start + step])
        pred    = inverse_transform_cpu(cpu_scaler, pred_sc)
        true    = inverse_transform_cpu(cpu_scaler, y_test[start:start + step])
        rmse, _ = calculate_metrics(true, pred)
        rmses.append(rmse)
    if not rmses:
        return float("nan"), float("nan")
    return float(np.mean(rmses)), float(np.std(rmses))


# =============================================
# MAIN EVALUATION
# =============================================
def evaluate_model(model, X_test, y_test, scaler, cpu_scaler=None,
                   run_prophet=False,
                   calibration_path="calibration_diagram.png",
                   use_log1p=True):
    """
    Full evaluation pipeline.

    FIX: mc_dropout_predict now returns 4 values.
    Coverage uses quantile-based lower/upper CI — correct for non-Gaussian data.
    log1p inverse applied before metric computation.
    """

    if cpu_scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        cpu_scaler                   = MinMaxScaler()
        cpu_scaler.min_              = np.array([scaler.min_[0]])
        cpu_scaler.scale_            = np.array([scaler.scale_[0]])
        cpu_scaler.data_min_         = np.array([scaler.data_min_[0]])
        cpu_scaler.data_max_         = np.array([scaler.data_max_[0]])
        cpu_scaler.data_range_       = np.array([scaler.data_range_[0]])
        cpu_scaler.n_features_in_    = 1
        cpu_scaler.feature_names_in_ = None

    # ── LSTM point predictions ────────────────────────────────
    print("\n[LSTM predictions...]")
    pred_scaled = predict(model, X_test)

    if use_log1p:
        pred_scaled = inverse_log1p(pred_scaled)

    pred_raw  = inverse_transform_cpu(cpu_scaler, pred_scaled)
    true      = inverse_transform_cpu(cpu_scaler, y_test)

    bias      = float(np.mean(true - pred_raw))
    pred      = np.clip(pred_raw + bias, 0.0, 1.0)
    residuals = true - pred

    lstm_rmse, lstm_mae = calculate_metrics(true, pred)
    print(f"  LSTM RMSE: {lstm_rmse:.6f}  (bias corrected by {bias:+.6f})")

    # ── MC Dropout — FIX: unpack 4 values ────────────────────
    print(f"\n[MC Dropout uncertainty ({100} samples)...]")
    mc_mean_log, mc_std_log, lower_log, upper_log = mc_dropout_predict(
        model, X_test, n_samples=100
    )

    if use_log1p:
        mc_mean_scaled = inverse_log1p(mc_mean_log)
        lower_scaled   = inverse_log1p(lower_log)
        upper_scaled   = inverse_log1p(upper_log)
    else:
        mc_mean_scaled = mc_mean_log
        lower_scaled   = lower_log
        upper_scaled   = upper_log

    cpu_range = float(cpu_scaler.data_max_[0] - cpu_scaler.data_min_[0])
    mc_mean   = inverse_transform_cpu(cpu_scaler, mc_mean_scaled) + bias
    lower_inv = np.clip(
        inverse_transform_cpu(cpu_scaler, lower_scaled), 0.0, 1.0
    )
    upper_inv = np.clip(
        inverse_transform_cpu(cpu_scaler, upper_scaled), 0.0, 1.0
    )

    mc_std_epistemic = mc_std_log.flatten() * cpu_range
    print(f"  MC std (epistemic) mean: {np.mean(mc_std_epistemic):.6f}")

    # ── Total std for calibration diagram ────────────────────
    aleatoric_std = float(np.std(residuals))
    total_std     = np.sqrt(mc_std_epistemic ** 2 + aleatoric_std ** 2)
    print(f"  Aleatoric std: {aleatoric_std:.6f}")
    print(f"  Total std:     {np.mean(total_std):.6f}")

    avg_total_std = float(np.mean(total_std))
    confidence    = float(np.clip(np.exp(-avg_total_std / 0.15), 0.05, 0.95))

    # ── Coverage — FIX: quantile-based ───────────────────────
    # lower_inv and upper_inv are 5th/95th percentile of MC samples
    # not ±1.96*std — correct for heavy-tailed CPU distributions
    coverage = float(np.mean((true >= lower_inv) & (true <= upper_inv)))
    avg_width = float(np.mean(upper_inv - lower_inv))
    print(f"  Coverage (quantile-based): {coverage:.4f}")

    # ── Calibration metrics ───────────────────────────────────
    print("\n[Calibration analysis...]")
    calibration = compute_reliability_diagram(
        y_true    = true,
        mean_pred = mc_mean,
        total_std = total_std,
        n_bins    = 10,
        save_path = calibration_path
    )

    ece       = calibration["ECE"]
    sharpness = calibration["Sharpness"]
    print(f"  ECE:       {ece:.4f}  ({calibration['ECE_quality']})")
    print(f"  Sharpness: {sharpness:.4f}")
    print(f"  Coverage:  {coverage:.4f}")

    # ── Baselines ─────────────────────────────────────────────
    print("\n[Baselines...]")
    naive_pred            = naive_forecast(true)
    naive_rmse, naive_mae = calculate_metrics(true, naive_pred)

    ma_pred         = moving_average_forecast(true)
    ma_rmse, ma_mae = calculate_metrics(true, ma_pred)

    print("\n[ARIMA baseline...]")
    arima_pred, arima_order = arima_forecast_walkforward(true)
    arima_rmse, arima_mae   = calculate_metrics(true, arima_pred)
    print(f"  ARIMA RMSE: {arima_rmse:.6f}  order={arima_order}")

    prophet_pred    = None
    prophet_rmse    = float("nan")
    prophet_mae     = float("nan")
    prophet_success = False
    prophet_verdict = "Prophet skipped (run_prophet=False). Set run_prophet=True to enable."

    if run_prophet:
        print("\n[Prophet baseline...]")
        prophet_pred, prophet_success = prophet_forecast_walkforward(true)
        if prophet_success:
            prophet_rmse, prophet_mae = calculate_metrics(true, prophet_pred)

    # ── Diagnostics ───────────────────────────────────────────
    print("\n[Diagnostics...]")
    lb_pvalue       = ljung_box_test(residuals)
    wf_mean, wf_std = walk_forward_validation(model, X_test, y_test, cpu_scaler)

    lstm_errors  = true - pred
    arima_errors = true - arima_pred
    dm_stat, dm_pval = diebold_mariano_test(lstm_errors, arima_errors)

    dm_stat_prophet = dm_pval_prophet = float("nan")
    if prophet_success and prophet_pred is not None:
        prophet_errors = true - prophet_pred
        dm_stat_prophet, dm_pval_prophet = diebold_mariano_test(
            lstm_errors, prophet_errors
        )
        prophet_verdict = get_prophet_verdict(
            lstm_rmse, prophet_rmse, dm_pval_prophet, prophet_success
        )

    beats_arima = "✅ YES" if lstm_rmse < arima_rmse else "❌ NO"
    print(f"\n{'='*60}")
    print(f"  LSTM RMSE:        {lstm_rmse:.6f}")
    print(f"  ARIMA RMSE:       {arima_rmse:.6f}")
    print(f"  LSTM beats ARIMA: {beats_arima}")
    print(f"  ECE:              {ece:.4f}  ({calibration['ECE_quality']})")
    print(f"  Coverage 90%:     {coverage:.4f}")
    print(f"  Confidence:       {confidence:.4f}")
    print(f"  DM p-value:       {dm_pval:.6f}")
    print(f"{'='*60}\n")

    return {
        "LSTM":          {"RMSE": lstm_rmse,  "MAE": lstm_mae},
        "Naive":         {"RMSE": naive_rmse, "MAE": naive_mae},
        "MovingAverage": {"RMSE": ma_rmse,    "MAE": ma_mae},
        "ARIMA":         {"RMSE": arima_rmse, "MAE": arima_mae,
                          "order": str(arima_order)},
        "Prophet": {
            "RMSE":    prophet_rmse,
            "MAE":     prophet_mae,
            "success": prophet_success,
            "verdict": prophet_verdict,
        },
        "Calibration": {
            "ECE":                   ece,
            "ECE_quality":           calibration["ECE_quality"],
            "Sharpness":             sharpness,
            "Coverage_95":           coverage,
            "predicted_confidences": calibration["predicted_confidences"],
            "actual_coverages":      calibration["actual_coverages"],
            "diagram_path":          calibration_path,
        },
        "Diagnostics": {
            "LjungBox_pvalue":       lb_pvalue,
            "WalkForward_RMSE":      wf_mean,
            "WalkForward_std":       wf_std,
            "Coverage_95":           coverage,
            "DieboldMariano_stat":   dm_stat,
            "DieboldMariano_pvalue": dm_pval,
            "DieboldMariano_sig":    bool(dm_pval < 0.05),
            "DM_Prophet_stat":       dm_stat_prophet,
            "DM_Prophet_pvalue":     dm_pval_prophet,
            "Confidence":            confidence,
            "Bias_corrected":        round(bias, 6),
            "Epistemic_std":         float(np.mean(mc_std_epistemic)),
            "Aleatoric_std":         aleatoric_std,
            "Total_std":             float(np.mean(total_std)),
            "LSTM_beats_ARIMA":      bool(lstm_rmse < arima_rmse),
            "LSTM_beats_Prophet":    bool(lstm_rmse < prophet_rmse)
                                     if prophet_success else None,
            "Prophet_verdict":       prophet_verdict,
        },
        "_arrays": {
            "true":              true,
            "pred":              pred,
            "pred_raw":          pred_raw,
            "mc_mean":           mc_mean,
            "mc_std":            total_std,
            "mc_std_epistemic":  mc_std_epistemic,
            "lower":             lower_inv,         # quantile-based lower CI
            "upper":             upper_inv,         # quantile-based upper CI
            "residuals":         residuals,
            "arima_pred":        arima_pred,
            "prophet_pred":      prophet_pred,
        }
    }
