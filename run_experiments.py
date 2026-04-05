# -*- coding: utf-8 -*-
"""
run_experiments.py -- CrashGuard AI 5-Seed Experiment Runner

Runs training 5 times with different random seeds.
Computes mean and std for each metric.
Saves results to experiments/results_5seed.json
Prints a clean results table for README and technical report.

Usage:
    python run_experiments.py

Output:
    experiments/results_5seed.json
    experiments/results_5seed_summary.txt
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

from lstm_model import build_lstm_model, train_model, mc_dropout_predict, set_seed
from preprocessing import prepare_data
from evaluate import (
    calculate_metrics,
    naive_forecast,
    moving_average_forecast,
    arima_forecast_walkforward,
    diebold_mariano_test,
    uncertainty_coverage,
    walk_forward_validation,
    inverse_transform_cpu,
)

# ============================================================
# CONFIG
# ============================================================
SEEDS            = [42, 123, 456, 789, 1337]
WINDOW_SIZE      = 60
FORECAST_HORIZON = 1
EPOCHS           = 50
BATCH_SIZE       = 256
LSTM_UNITS       = [128, 64]
DROPOUT_RATE     = 0.25
LEARNING_RATE    = 0.001
MC_SAMPLES       = 50
DATA_PATH        = "data/cpu_timeseries.csv"
OUTPUT_DIR       = "experiments"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA ONCE — same data for all seeds
# ============================================================
print("=" * 60)
print("CrashGuard AI -- 5-Seed Experiment Runner")
print("=" * 60)
print(f"\nSeeds: {SEEDS}")
print(f"Window: {WINDOW_SIZE} | Horizon: {FORECAST_HORIZON} | Epochs: {EPOCHS}")
print(f"LSTM units: {LSTM_UNITS} | Dropout: {DROPOUT_RATE}")
print(f"Batch: {BATCH_SIZE} | MC samples: {MC_SAMPLES}")
print()

print("[Loading data...]")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"  Rows: {len(df):,}")
assert len(df) >= 10_000, "Need 10K+ rows. Run generate_data.py first."

print("[Preparing data...]")
processed = prepare_data(
    df,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON
)

X_train    = processed["X_train"]
y_train    = processed["y_train"]
X_val      = processed["X_val"]
y_val      = processed["y_val"]
X_test     = processed["X_test"]
y_test     = processed["y_test"]
cpu_scaler = processed["cpu_scaler"]
scaler     = processed["scaler"]
num_features = X_train.shape[2]

print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"  Features: {num_features}")

# ============================================================
# RUN ARIMA ONCE — same for all seeds (deterministic baseline)
# ============================================================
print("\n[Running walk-forward ARIMA baseline (runs once)...]")
true_cpu = inverse_transform_cpu(cpu_scaler, y_test)
arima_pred, arima_order = arima_forecast_walkforward(true_cpu)
arima_rmse, arima_mae   = calculate_metrics(true_cpu, arima_pred)
naive_pred = naive_forecast(true_cpu)
naive_rmse, naive_mae   = calculate_metrics(true_cpu, naive_pred)
print(f"  ARIMA RMSE: {arima_rmse:.6f} (order={arima_order})")
print(f"  Naive RMSE: {naive_rmse:.6f}")


# ============================================================
# COLLECT RESULTS
# ============================================================
all_results = []

for run_idx, seed in enumerate(SEEDS):

    print("\n" + "=" * 60)
    print(f"RUN {run_idx + 1} / {len(SEEDS)}  |  Seed = {seed}")
    print("=" * 60)

    # Set seed everywhere
    set_seed(seed)

    # Build model
    model = build_lstm_model(
        window_size=WINDOW_SIZE,
        num_features=num_features,
        forecast_horizon=FORECAST_HORIZON,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )

    # Train
    history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    epochs_run       = len(history.history["loss"])
    final_train_loss = history.history["loss"][-1]
    final_val_loss   = history.history["val_loss"][-1]
    print(f"\n  Epochs run: {epochs_run}")
    print(f"  Train loss: {final_train_loss:.6f}")
    print(f"  Val loss:   {final_val_loss:.6f}")

    # ── Point predictions ───────────────────────────────────
    pred_scaled = model.predict(X_test, verbose=0)
    pred_raw    = inverse_transform_cpu(cpu_scaler, pred_scaled)
    true        = inverse_transform_cpu(cpu_scaler, y_test)

    # Bias correction
    bias = float(np.mean(true - pred_raw))
    pred = np.clip(pred_raw + bias, 0.0, 1.0)

    lstm_rmse, lstm_mae = calculate_metrics(true, pred)

    # ── MC Dropout ──────────────────────────────────────────
    mc_mean_sc, mc_std_sc = mc_dropout_predict(model, X_test, n_samples=MC_SAMPLES)
    mc_mean    = inverse_transform_cpu(cpu_scaler, mc_mean_sc) + bias
    cpu_range  = float(cpu_scaler.data_max_[0] - cpu_scaler.data_min_[0])
    mc_std     = mc_std_sc.flatten() * cpu_range
    coverage   = uncertainty_coverage(true, mc_mean, mc_std)

    # ── Walk-forward LSTM ───────────────────────────────────
    wf_rmse, wf_std = walk_forward_validation(
        model, X_test, y_test, cpu_scaler, step=10
    )

    # ── Diebold-Mariano ─────────────────────────────────────
    lstm_errors  = true - pred
    arima_errors = true - arima_pred
    dm_stat, dm_pval = diebold_mariano_test(lstm_errors, arima_errors)

    # ── Print run summary ───────────────────────────────────
    print(f"\n  RMSE:          {lstm_rmse:.6f}")
    print(f"  MAE:           {lstm_mae:.6f}")
    print(f"  Coverage:      {coverage:.4f}")
    print(f"  DM p-value:    {dm_pval:.6f}")
    print(f"  WF RMSE:       {wf_rmse:.6f}")
    print(f"  Beats ARIMA:   {lstm_rmse < arima_rmse}")

    run_result = {
        "seed":          seed,
        "run":           run_idx + 1,
        "RMSE":          round(lstm_rmse, 6),
        "MAE":           round(lstm_mae, 6),
        "Coverage":      round(coverage, 6),
        "DM_pvalue":     round(dm_pval, 6),
        "WF_RMSE":       round(wf_rmse, 6),
        "WF_std":        round(wf_std, 6),
        "DM_stat":       round(dm_stat, 6),
        "Bias":          round(bias, 6),
        "epochs_run":    epochs_run,
        "train_loss":    round(final_train_loss, 6),
        "val_loss":      round(final_val_loss, 6),
        "beats_ARIMA":   bool(lstm_rmse < arima_rmse),
    }
    all_results.append(run_result)

    # Save each run's model separately
    run_model_path = os.path.join(OUTPUT_DIR, f"model_seed{seed}.keras")
    try:
        model.save(run_model_path)
        print(f"  Model saved: {run_model_path}")
    except Exception as e:
        print(f"  Model save warning: {e}")

    # Also save as main model if this is best RMSE so far
    best_rmse_so_far = min(r["RMSE"] for r in all_results)
    if lstm_rmse <= best_rmse_so_far:
        joblib.dump(model, "saved_model.pkl")
        try:
            model.save("saved_model.keras")
        except Exception:
            pass
        print(f"  Best model updated (RMSE={lstm_rmse:.6f})")


# ============================================================
# COMPUTE MEAN AND STD
# ============================================================
print("\n" + "=" * 60)
print("AGGREGATING RESULTS ACROSS 5 SEEDS")
print("=" * 60)

metrics_to_aggregate = ["RMSE", "MAE", "Coverage", "DM_pvalue", "WF_RMSE"]

summary = {}
for metric in metrics_to_aggregate:
    values = [r[metric] for r in all_results]
    summary[metric] = {
        "mean":   round(float(np.mean(values)), 6),
        "std":    round(float(np.std(values)), 6),
        "min":    round(float(np.min(values)), 6),
        "max":    round(float(np.max(values)), 6),
        "values": values
    }

beats_arima_count = sum(1 for r in all_results if r["beats_ARIMA"])


# ============================================================
# PRINT RESULTS TABLE
# ============================================================
print("\n")
print("=" * 60)
print("RESULTS TABLE (for README and technical report)")
print("=" * 60)
print()

col1 = 20
col2 = 12
col3 = 12

header = f"{'Metric':<{col1}} {'Mean':<{col2}} {'Std':<{col3}}"
print(header)
print("-" * (col1 + col2 + col3))

metric_display_names = {
    "RMSE":      "RMSE",
    "MAE":       "MAE",
    "Coverage":  "Coverage 95%",
    "DM_pvalue": "DM p-value",
    "WF_RMSE":   "Walk-Forward RMSE",
}

for metric, display in metric_display_names.items():
    mean_val = summary[metric]["mean"]
    std_val  = summary[metric]["std"]
    print(f"{display:<{col1}} {mean_val:<{col2}.6f} {std_val:<{col3}.6f}")

print()
print(f"Baseline ARIMA RMSE:   {arima_rmse:.6f} (order={arima_order})")
print(f"Baseline Naive RMSE:   {naive_rmse:.6f}")
print(f"LSTM beats ARIMA:      {beats_arima_count}/{len(SEEDS)} runs")
print()

# ============================================================
# PRINT PER-RUN TABLE
# ============================================================
print("=" * 60)
print("PER-RUN BREAKDOWN")
print("=" * 60)
print()

run_header = f"{'Seed':<8} {'RMSE':<12} {'MAE':<12} {'Coverage':<12} {'DM p-val':<12} {'WF RMSE':<12} {'Beats ARIMA'}"
print(run_header)
print("-" * len(run_header))

for r in all_results:
    beats = "YES" if r["beats_ARIMA"] else "NO"
    print(
        f"{r['seed']:<8} "
        f"{r['RMSE']:<12.6f} "
        f"{r['MAE']:<12.6f} "
        f"{r['Coverage']:<12.4f} "
        f"{r['DM_pvalue']:<12.6f} "
        f"{r['WF_RMSE']:<12.6f} "
        f"{beats}"
    )

print()


# ============================================================
# SAVE JSON
# ============================================================
output = {
    "experiment_info": {
        "run_at":          datetime.now().isoformat(),
        "seeds":           SEEDS,
        "n_runs":          len(SEEDS),
        "window_size":     WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "epochs_max":      EPOCHS,
        "batch_size":      BATCH_SIZE,
        "lstm_units":      LSTM_UNITS,
        "dropout_rate":    DROPOUT_RATE,
        "learning_rate":   LEARNING_RATE,
        "mc_samples":      MC_SAMPLES,
        "train_rows":      len(df),
        "num_features":    num_features,
    },
    "baselines": {
        "ARIMA": {
            "RMSE":  round(arima_rmse, 6),
            "MAE":   round(arima_mae, 6),
            "order": str(arima_order),
            "method": "walk-forward"
        },
        "Naive": {
            "RMSE": round(naive_rmse, 6),
            "MAE":  round(naive_mae, 6),
        }
    },
    "summary": summary,
    "beats_arima": {
        "count": beats_arima_count,
        "total": len(SEEDS),
        "pct":   round(beats_arima_count / len(SEEDS) * 100, 1)
    },
    "per_run": all_results
}

json_path = os.path.join(OUTPUT_DIR, "results_5seed.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved: {json_path}")


# ============================================================
# SAVE SUMMARY TEXT (for README copy-paste)
# ============================================================
summary_lines = []
summary_lines.append("## CrashGuard AI -- Experimental Results (N=5 seeds)")
summary_lines.append("")
summary_lines.append("### Model Configuration")
summary_lines.append(f"- Architecture: LSTM ({LSTM_UNITS[0]}, {LSTM_UNITS[1]})")
summary_lines.append(f"- Window size: {WINDOW_SIZE} | Horizon: {FORECAST_HORIZON}")
summary_lines.append(f"- Dropout rate: {DROPOUT_RATE} | MC samples: {MC_SAMPLES}")
summary_lines.append(f"- Training rows: {len(df):,}")
summary_lines.append(f"- Features: {num_features}")
summary_lines.append("")
summary_lines.append("### Results Table")
summary_lines.append("")
summary_lines.append("| Metric | Mean | Std |")
summary_lines.append("|--------|------|-----|")
for metric, display in metric_display_names.items():
    m = summary[metric]["mean"]
    s = summary[metric]["std"]
    summary_lines.append(f"| {display} | {m:.6f} | {s:.6f} |")
summary_lines.append("")
summary_lines.append("### Baseline Comparison")
summary_lines.append("")
summary_lines.append("| Model | RMSE |")
summary_lines.append("|-------|------|")
summary_lines.append(f"| **LSTM (ours)** | **{summary['RMSE']['mean']:.6f} ± {summary['RMSE']['std']:.6f}** |")
summary_lines.append(f"| ARIMA {arima_order} | {arima_rmse:.6f} |")
summary_lines.append(f"| Naive | {naive_rmse:.6f} |")
summary_lines.append("")
summary_lines.append(f"LSTM beats ARIMA in {beats_arima_count}/{len(SEEDS)} runs.")
summary_lines.append("")
summary_lines.append(f"*Results are mean ± std over N={len(SEEDS)} independent runs with seeds {SEEDS}.*")

txt_path = os.path.join(OUTPUT_DIR, "results_5seed_summary.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"Summary saved: {txt_path}")


# ============================================================
# FINAL MESSAGE
# ============================================================
print()
print("=" * 60)
print("DONE")
print("=" * 60)
print()
print(f"Files saved:")
print(f"  {json_path}")
print(f"  {txt_path}")
print()
print(f"Best RMSE across 5 runs:  {summary['RMSE']['min']:.6f}")
print(f"Mean RMSE:                {summary['RMSE']['mean']:.6f} +/- {summary['RMSE']['std']:.6f}")
print(f"Mean Coverage:            {summary['Coverage']['mean']:.4f} +/- {summary['Coverage']['std']:.4f}")
print(f"LSTM beats ARIMA:         {beats_arima_count}/{len(SEEDS)} runs ({beats_arima_count/len(SEEDS)*100:.0f}%)")
print()
print("Copy the table from experiments/results_5seed_summary.txt")
print("into your README and technical report.")
print()
print("Best model saved as saved_model.keras + saved_model.pkl")