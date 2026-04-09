from dotenv import load_dotenv
load_dotenv()
from transformer_model import build_transformer_model
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import joblib
import time
import os
import tensorflow as tf
from datetime import datetime

from notifications import send_slack_alert, send_test_message, SLACK_WEBHOOK_URL
from preprocessing import prepare_data
from lstm_model import build_lstm_model, train_model, mc_dropout_predict, inverse_log1p
from tcn_model import build_tcn_model
from evaluate import evaluate_model
from optuna_tuning import run_optuna
from anomaly_detection import get_anomaly_summary
from live_monitor import get_system_stats, build_live_feature_window, get_current_cpu
from ablation_study import run_full_ablation, ABLATION_CONFIGS
from advanced_features import AlertSystem
from error_analysis import (
    segment_errors, get_worst_predictions,
    plot_error_by_region, plot_prediction_vs_actual_with_errors,
    plot_rolling_error, plot_error_distribution_by_region
)
from mlflow_tracker import log_experiment, get_all_runs, get_best_run
from multistep_forecast import (
    mc_multistep_forecast,
    plot_multistep_forecast, plot_step_uncertainty
)
from shap_explainer import (
    get_shap_explainer, compute_shap_values, get_feature_importance,
    get_temporal_shap, plot_feature_importance,
    plot_temporal_importance, plot_shap_waterfall
)
from xgboost_model import train_xgb, predict_xgb, ensemble_predict, save_xgb, load_xgb

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="CrashGuard AI — Predictive CPU Forecasting",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CrashGuard AI — Predictive CPU Observability Platform")
st.caption("LSTM + XGBoost Ensemble · MC Dropout uncertainty · SHAP explainability · MLflow tracking · Real-time alerting")

# ================= 10 TABS =================
(tab1, tab2, tab3, tab4, tab5,
 tab6, tab7, tab8, tab9, tab10) = st.tabs([
    "📁 Data & Training",
    "📊 Evaluation & Comparison",
    "🔴 Live Monitor",
    "🚨 Anomaly Detection",
    "🔍 SHAP Explainability",
    "📈 Multi-Step Forecast",
    "🧪 MLflow Tracking",
    "🔬 Ablation Study",
    "🔎 Error Analysis",
    "🚨 Alerts",
])

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("data/google_cluster_processed.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    return df

df = load_data()

# ================= INIT ALERT SYSTEM =================
if "alert_system" not in st.session_state:
    st.session_state["alert_system"] = AlertSystem()

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Configuration")
st.sidebar.divider()

st.sidebar.subheader("🔔 Slack Alerts")
slack_webhook = st.sidebar.text_input(
    "Slack Webhook URL",
    value=SLACK_WEBHOOK_URL,
    placeholder="https://hooks.slack.com/...",
    type="password"
)
if slack_webhook and st.sidebar.button("Test Slack"):
    sent = send_test_message(slack_webhook)
    if sent:
        st.sidebar.success("✅ Slack connected!")
    else:
        st.sidebar.error("❌ Failed — check URL")

st.sidebar.divider()
window_size      = st.sidebar.slider("Window Size",       30, 120, 60)
forecast_horizon = st.sidebar.slider("Forecast Horizon",   1,  10,  1)
epochs           = st.sidebar.slider("Epochs",            10, 100, 50)
model_type       = st.sidebar.selectbox("Model Architecture", ["LSTM", "TCN", "Transformer"])

st.sidebar.divider()
st.sidebar.subheader("⚡ Evaluation Speed")
run_prophet_option = st.sidebar.checkbox(
    "Run Prophet baseline (slow — 20-30 min)",
    value=False
)
st.sidebar.caption("✅ Fast mode: ARIMA capped at 2000 pts, 6 order combos.")

# ================= TAB 1: DATA & TRAINING =================
with tab1:
    st.subheader("📁 Dataset Preview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Avg CPU",    f"{df['cpu_usage'].mean():.4f}")
    col3.metric("Max CPU",    f"{df['cpu_usage'].max():.4f}")
    col4.metric("Min CPU",    f"{df['cpu_usage'].min():.4f}")

    st.dataframe(df.head(20), use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cpu_usage"],
        name="CPU Usage", line=dict(color="#ef4444", width=1)
    ))
    fig.update_layout(
        title="CPU Usage Over Time — Google Cluster Trace (60K rows)",
        height=350, template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🔧 Training Pipeline")

    if st.sidebar.button("Prepare Data"):
        with st.spinner("Preparing data..."):
            processed = prepare_data(
                df, window_size=window_size,
                forecast_horizon=forecast_horizon
            )
            st.session_state["processed"] = processed
        n_feat = processed["X_train"].shape[2]
        st.success(
            f"✅ Data prepared — {processed['X_train'].shape[0]:,} train · "
            f"{processed['X_val'].shape[0]:,} val · "
            f"{processed['X_test'].shape[0]:,} test · "
            f"{n_feat} features"
        )

    if st.sidebar.button("Run Hyperparameter Tuning"):
        processed = st.session_state.get("processed")
        if processed is None:
            st.warning("Prepare data first!")
        else:
            with st.spinner("Running Optuna (20 trials)..."):
                study = run_optuna(processed, trials=20)
            st.success("✅ Best params found!")
            st.json(study.best_params)
            st.session_state["best_params"] = study.best_params

    if st.sidebar.button("Train Model"):
        processed = st.session_state.get("processed")
        if processed is None:
            st.warning("Prepare data first!")
        else:
            num_features = processed["X_train"].shape[2]
            best_params  = st.session_state.get("best_params")
            if best_params is None:
                lstm_units, dropout, lr, batch = [128, 64], 0.30, 0.001, 256
            else:
                lstm_units = [best_params["units1"], best_params["units2"]]
                dropout    = best_params["dropout"]
                lr         = best_params["lr"]
                batch      = best_params["batch_size"]

            with st.spinner(f"Training {model_type}..."):
                if model_type == "LSTM":
                    model = build_lstm_model(
                        window_size=window_size, num_features=num_features,
                        forecast_horizon=forecast_horizon, lstm_units=lstm_units,
                        dropout_rate=dropout, learning_rate=lr,
                        weight_decay=1e-4
                    )
                elif model_type == "TCN":
                    model = build_tcn_model(
                        window_size=window_size, num_features=num_features,
                        forecast_horizon=forecast_horizon
                    )
                else:
                    model = build_transformer_model(
                        window_size=window_size, num_features=num_features,
                        forecast_horizon=forecast_horizon
                    )

                # FIX: log1p on targets before training
                y_train_log = np.log1p(processed["y_train"])
                y_val_log   = np.log1p(processed["y_val"])

                # FIX: train_model() no longer takes spike_threshold/spike_weight
                history = train_model(
                    model,
                    processed["X_train"], y_train_log,
                    processed["X_val"],   y_val_log,
                    epochs=epochs, batch_size=batch
                )

                # Train XGBoost ensemble component
                st.info("Training XGBoost ensemble component...")
                xgb_model = train_xgb(
                    processed["X_train"], y_train_log,
                    processed["X_val"],   y_val_log
                )
                save_xgb(xgb_model, "saved_xgb_model.pkl")

                joblib.dump(model, "saved_model.pkl")
                try:
                    model.save("saved_model.keras")
                except Exception:
                    pass

                st.session_state["model"]     = model
                st.session_state["xgb_model"] = xgb_model
                st.session_state["history"]   = history

            st.success(f"✅ {model_type} + XGBoost trained and saved!")

            hist = history.history
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=hist["loss"],     name="Train Loss", line=dict(color="#ef4444")))
            fig_loss.add_trace(go.Scatter(y=hist["val_loss"], name="Val Loss",   line=dict(color="#3b82f6")))
            fig_loss.update_layout(
                title="Training Loss Curve",
                template="plotly_dark", height=300,
                xaxis_title="Epoch", yaxis_title="Huber Loss"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    if st.sidebar.button("Evaluate Model"):
        model     = st.session_state.get("model")
        processed = st.session_state.get("processed")
        if model is None:
            st.warning("Train model first!")
        else:
            with st.spinner("Evaluating..."):
                results = evaluate_model(
                    model,
                    processed["X_test"], processed["y_test"],
                    processed["scaler"], processed.get("cpu_scaler"),
                    run_prophet=run_prophet_option,
                    use_log1p=True
                )
            st.session_state["results"] = results
            st.success("✅ Evaluation complete! Go to Tab 2 to see results.")

            try:
                best_params = st.session_state.get("best_params") or {}
                run_id = log_experiment(
                    model=model, model_type=model_type,
                    params={
                        "window_size": window_size,
                        "forecast_horizon": forecast_horizon,
                        "epochs": epochs,
                        **best_params
                    },
                    metrics=results,
                    window_size=window_size,
                    forecast_horizon=forecast_horizon,
                    feature_names=processed.get("feature_names", [])
                )
                st.info(f"📊 Logged to MLflow — Run ID: `{run_id}`")
            except Exception as e:
                st.warning(f"MLflow logging skipped: {e}")

    if st.sidebar.button("Visualize Forecast"):
        results   = st.session_state.get("results")
        processed = st.session_state.get("processed")
        if results and processed:
            arrays = results.get("_arrays")
            if arrays:
                y_true  = arrays["true"]
                y_pred  = arrays["pred"]
                mc_mean = arrays.get("mc_mean", y_pred)
                # FIX: use quantile-based lower/upper from _arrays
                # not ±1.96*std
                lower   = arrays.get("lower", np.zeros_like(y_pred))
                upper   = arrays.get("upper", np.ones_like(y_pred))
                n       = min(200, len(y_true))

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(y=y_true[:n], name="Actual",
                                            line=dict(color="white",   width=2)))
                fig_fc.add_trace(go.Scatter(y=y_pred[:n], name="Prediction",
                                            line=dict(color="#ef4444", width=2, dash="dot")))
                fig_fc.add_trace(go.Scatter(y=upper[:n], name="Upper 90% CI",
                                            line=dict(color="#f59e0b", dash="dot", width=1)))
                fig_fc.add_trace(go.Scatter(
                    y=lower[:n], name="Lower 90% CI",
                    line=dict(color="#f59e0b", dash="dot", width=1),
                    fill="tonexty", fillcolor="rgba(245,158,11,0.1)"
                ))
                fig_fc.update_layout(
                    title="Forecast vs Actual with 90% Uncertainty Bands (Quantile-based)",
                    template="plotly_dark", height=450
                )
                st.plotly_chart(fig_fc, use_container_width=True)

# ================= TAB 2: EVALUATION =================
with tab2:
    st.subheader("📊 Model Performance")
    results = st.session_state.get("results")

    if results is not None:

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LSTM RMSE",    f"{results['LSTM']['RMSE']:.6f}")
        c2.metric("Naive RMSE",   f"{results['Naive']['RMSE']:.6f}")
        c3.metric("ARIMA RMSE",   f"{results['ARIMA']['RMSE']:.6f}")
        c4.metric("Coverage 90%", f"{results['Diagnostics']['Coverage_95']:.3f}")

        if results["Diagnostics"].get("LSTM_beats_ARIMA"):
            pct = (results["ARIMA"]["RMSE"] - results["LSTM"]["RMSE"]) / results["ARIMA"]["RMSE"] * 100
            st.success(
                f"✅ LSTM beats ARIMA by {pct:.1f}% RMSE "
                f"({results['LSTM']['RMSE']:.6f} vs {results['ARIMA']['RMSE']:.6f})"
            )
        else:
            st.error("❌ LSTM does NOT beat ARIMA — needs more training or better features")

        models    = ["LSTM", "Naive", "MovingAverage", "ARIMA"]
        rmse_vals = [results["LSTM"]["RMSE"],         results["Naive"]["RMSE"],
                     results["MovingAverage"]["RMSE"], results["ARIMA"]["RMSE"]]
        mae_vals  = [results["LSTM"]["MAE"],          results["Naive"]["MAE"],
                     results["MovingAverage"]["MAE"],  results["ARIMA"]["MAE"]]
        colors    = ["#ef4444", "#6b7280", "#3b82f6", "#f59e0b"]

        prophet_data    = results.get("Prophet", {})
        prophet_success = prophet_data.get("success", False)
        prophet_rmse    = prophet_data.get("RMSE", None)
        if prophet_success and prophet_rmse and prophet_rmse == prophet_rmse:
            models.append("Prophet")
            rmse_vals.append(prophet_rmse)
            mae_vals.append(prophet_data.get("MAE", 0))
            colors.append("#8b5cf6")

        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Bar(
            x=models, y=rmse_vals, marker_color=colors,
            text=[f"{v:.6f}" for v in rmse_vals], textposition="auto"
        ))
        fig_rmse.update_layout(title="RMSE Comparison — All Models",
                               template="plotly_dark", height=380)
        st.plotly_chart(fig_rmse, use_container_width=True)

        if prophet_success:
            verdict = results["Diagnostics"].get("Prophet_verdict", "")
            if results["Diagnostics"].get("LSTM_beats_Prophet"):
                st.success(f"Prophet: {verdict}")
            else:
                st.info(f"Prophet: {verdict}")
        else:
            msg = results.get("Prophet", {}).get("verdict", "")
            if msg:
                st.info(f"ℹ️ {msg}")

        st.divider()
        st.subheader("📐 Uncertainty Calibration")

        calibration = results.get("Calibration", {})
        if calibration:
            ece       = calibration.get("ECE",         float("nan"))
            sharpness = calibration.get("Sharpness",   float("nan"))
            coverage  = calibration.get("Coverage_95", float("nan"))
            quality   = calibration.get("ECE_quality", "")

            cal1, cal2, cal3 = st.columns(3)
            cal1.metric("ECE", f"{ece:.4f}", delta=quality,
                        delta_color="normal" if ece < 0.05 else "off" if ece < 0.10 else "inverse")
            cal2.metric("Sharpness (Avg 95% CI Width)", f"{sharpness:.4f}")
            cal3.metric("Coverage @ 90% CI", f"{coverage:.4f}")

            if ece < 0.05:
                st.success(f"✅ Well calibrated — ECE = {ece:.4f}")
            elif ece < 0.10:
                st.warning(f"⚠️ Acceptable — ECE = {ece:.4f}")
            else:
                st.error(f"❌ Poor calibration — ECE = {ece:.4f}")

            diagram_path = calibration.get("diagram_path", "calibration_diagram.png")
            if os.path.exists(diagram_path):
                st.image(diagram_path, use_container_width=True)
            else:
                pred_conf = calibration.get("predicted_confidences", [])
                act_cov   = calibration.get("actual_coverages", [])
                if pred_conf and act_cov:
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Perfect calibration",
                        line=dict(color="#64748b", dash="dash", width=1.5)
                    ))
                    fig_cal.add_trace(go.Scatter(
                        x=pred_conf, y=act_cov, mode="lines+markers",
                        name=f"CrashGuard AI (ECE={ece:.3f})",
                        line=dict(color="#22c55e", width=2.5),
                        marker=dict(size=8)
                    ))
                    fig_cal.update_layout(
                        title="Reliability Diagram",
                        xaxis_title="Predicted confidence level",
                        yaxis_title="Actual coverage",
                        template="plotly_dark", height=400,
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)

        st.divider()
        st.subheader("📈 Statistical Diagnostics")
        diag = results["Diagnostics"]
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Ljung-Box p",  f"{diag['LjungBox_pvalue']:.4f}")
        d2.metric("WF RMSE",      f"{diag['WalkForward_RMSE']:.6f}")
        d3.metric("WF Std",       f"{diag['WalkForward_std']:.6f}")
        d4.metric("Coverage 90%", f"{diag['Coverage_95']:.4f}")
        d5.metric("DM p-value",   f"{diag['DieboldMariano_pvalue']:.6f}")

        if diag["DieboldMariano_pvalue"] < 0.05:
            st.success(
                f"✅ DM test: p = {diag['DieboldMariano_pvalue']:.4f} — "
                f"LSTM statistically significantly better than ARIMA"
            )
        else:
            st.warning(f"⚠️ DM p = {diag['DieboldMariano_pvalue']:.4f} — not yet significant")

        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(
            x=models, y=mae_vals, marker_color=colors,
            text=[f"{v:.6f}" for v in mae_vals], textposition="auto"
        ))
        fig_mae.update_layout(title="MAE Comparison — All Models",
                              template="plotly_dark", height=380)
        st.plotly_chart(fig_mae, use_container_width=True)

        arrays = results.get("_arrays")
        if arrays is not None:
            st.subheader("📉 Residual Analysis")
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(
                x=arrays["residuals"], nbinsx=50,
                marker_color="#3b82f6", name="Residuals"
            ))
            fig_res.update_layout(
                title="Residual Distribution",
                template="plotly_dark", height=300
            )
            st.plotly_chart(fig_res, use_container_width=True)

        export_df = pd.DataFrame({"Model": models, "RMSE": rmse_vals, "MAE": mae_vals})
        st.download_button(
            "⬇️ Download Results CSV",
            export_df.to_csv(index=False).encode(),
            "evaluation_results.csv", "text/csv"
        )

    else:
        st.info("Train and evaluate your model first (Tab 1).")

# ================= TAB 3: LIVE MONITOR =================
with tab3:
    st.subheader("🔴 Live System CPU Monitor")
    auto_refresh = st.toggle("Enable Auto-Refresh (every 2s)", value=False, key="live_refresh")

    stats = get_system_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPU Usage",   f"{stats['cpu_total']}%")
    col2.metric("Memory Used", f"{stats['memory_used_pct']}%")
    col3.metric("Free Memory", f"{stats['memory_available_gb']:.1f} GB")
    col4.metric("CPU Cores",   stats['cpu_core_count'])

    if stats["cpu_per_core"]:
        fig_cores = go.Figure()
        fig_cores.add_trace(go.Bar(
            x=[f"Core {i}" for i in range(len(stats["cpu_per_core"]))],
            y=stats["cpu_per_core"], marker_color="#ef4444"
        ))
        fig_cores.update_layout(
            title="Per-Core CPU Usage (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_dark", height=300
        )
        st.plotly_chart(fig_cores, use_container_width=True)

    model     = st.session_state.get("model")
    processed = st.session_state.get("processed")

    if model is not None and processed is not None:
        st.subheader("🧠 Live LSTM + XGBoost Ensemble Prediction")

        expected_features = processed["X_train"].shape[2]
        expected_window   = processed["X_train"].shape[1]

        if "live_history" not in st.session_state:
            st.session_state["live_history"] = []

        current_cpu = get_current_cpu()
        st.session_state["live_history"].append(current_cpu)

        if len(st.session_state["live_history"]) >= expected_window:
            try:
                raw_window  = build_live_feature_window(
                    st.session_state["live_history"],
                    window_size=expected_window
                )
                live_window = raw_window.reshape(1, expected_window, expected_features)

                if np.isnan(live_window).any():
                    st.warning("⚠️ NaN in live window — using last known prediction.")
                    live_pred = st.session_state.get("last_live_pred", current_cpu)
                else:
                    X_tensor = tf.constant(live_window, dtype=tf.float32)

                    # FIX: mc_dropout_predict returns 4 values
                    mc_mean_log, mc_std_log, lower_log, upper_log = mc_dropout_predict(
                        model, live_window, n_samples=100
                    )

                    mc_mean_live = float(np.clip(inverse_log1p(mc_mean_log)[0][0], 0.0, 1.0))
                    lower_live   = float(np.clip(inverse_log1p(lower_log)[0][0], 0.0, 1.0))
                    upper_live   = float(np.clip(inverse_log1p(upper_log)[0][0], 0.0, 1.0))

                    cpu_scaler = processed.get("cpu_scaler")
                    if cpu_scaler is not None:
                        mc_mean_live = float(np.clip(
                            cpu_scaler.inverse_transform([[mc_mean_live]])[0][0], 0.0, 1.0
                        ))
                        lower_live = float(np.clip(
                            cpu_scaler.inverse_transform([[lower_live]])[0][0], 0.0, 1.0
                        ))
                        upper_live = float(np.clip(
                            cpu_scaler.inverse_transform([[upper_live]])[0][0], 0.0, 1.0
                        ))

                    live_pred  = mc_mean_live
                    ci_width   = upper_live - lower_live
                    confidence = float(np.clip(1.0 - (ci_width / 0.5), 0.05, 0.95))
                    st.session_state["last_live_pred"] = live_pred

                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Current CPU",    f"{current_cpu:.4f}")
                    col_b.metric("Predicted Next", f"{live_pred:.4f}")
                    col_c.metric("90% CI",         f"[{lower_live:.3f}, {upper_live:.3f}]")
                    col_d.metric("Confidence",     f"{confidence:.2f}")

                    if live_pred > 0.75:
                        st.error(f"🔴 SPIKE PREDICTED — {live_pred:.1%} exceeds 75% threshold")
                    else:
                        st.success("🟢 System stable — no spike predicted")

                fig_live = go.Figure()
                history_display = st.session_state["live_history"][-100:]
                fig_live.add_trace(go.Scatter(
                    y=history_display, name="Live CPU",
                    line=dict(color="#ef4444", width=2)
                ))
                if not np.isnan(live_window).any():
                    fig_live.add_trace(go.Scatter(
                        x=[len(history_display)],
                        y=[live_pred],
                        mode="markers", name="Next Prediction",
                        marker=dict(color="#f59e0b", size=14, symbol="diamond")
                    ))
                fig_live.update_layout(
                    title="Live CPU History + Next Prediction",
                    template="plotly_dark", height=300
                )
                st.plotly_chart(fig_live, use_container_width=True)

            except Exception as e:
                last_known = st.session_state.get("last_live_pred", current_cpu)
                st.warning(f"⚠️ Prediction error — using last known {last_known:.4f}: {e}")
        else:
            remaining = expected_window - len(st.session_state["live_history"])
            st.info(f"Collecting data — need {remaining} more readings.")
            st.progress(len(st.session_state["live_history"]) / expected_window)

    else:
        st.info("💡 Train your model first (Tab 1), then live prediction will appear here.")

    if auto_refresh:
        time.sleep(1)
        st.rerun()

# ================= TAB 4: ANOMALY DETECTION =================
with tab4:
    st.subheader("🚨 CPU Anomaly Detection")
    st.caption("Isolation Forest + Z-Score dual detection on real CPU telemetry")

    sample_size = st.slider(
        "Sample size", 100,
        min(len(df), 2000), min(500, len(df)), step=100
    )

    if st.button("Run Anomaly Detection"):
        with st.spinner("Detecting anomalies..."):
            cpu_sample = df["cpu_usage"].values[:sample_size]
            summary    = get_anomaly_summary(cpu_sample)
        st.session_state["anomaly_summary"] = summary
        st.session_state["anomaly_cpu"]     = cpu_sample

    summary    = st.session_state.get("anomaly_summary")
    cpu_sample = st.session_state.get("anomaly_cpu")

    if summary is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Points",    f"{summary['total_points']:,}")
        col2.metric("Anomalies Found", f"{summary['anomaly_count']:,}")
        col3.metric("Anomaly %",       f"{summary['anomaly_pct']}%")
        col4.metric("IF Flags",        f"{summary.get('if_flags', 0):,}")

        anomaly_mask = summary["combined_flags"]
        normal_idx   = np.where(~anomaly_mask)[0]
        anomaly_idx  = np.where(anomaly_mask)[0]

        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=normal_idx, y=cpu_sample[normal_idx],
            mode="lines", name="Normal", line=dict(color="#3b82f6", width=1)
        ))
        fig_anom.add_trace(go.Scatter(
            x=anomaly_idx, y=cpu_sample[anomaly_idx],
            mode="markers", name="Anomaly",
            marker=dict(color="#ef4444", size=6, symbol="x")
        ))
        fig_anom.update_layout(
            title="CPU Usage — Anomalies Highlighted",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig_anom, use_container_width=True)
    else:
        st.info("Click 'Run Anomaly Detection' to start.")

# ================= TAB 5: SHAP EXPLAINABILITY =================
with tab5:
    st.subheader("🔍 SHAP Model Explainability")

    model     = st.session_state.get("model")
    processed = st.session_state.get("processed")

    if model is None or processed is None:
        st.info("Train the model first (Tab 1).")
    else:
        if st.button("Run SHAP Analysis"):
            with st.spinner("Computing SHAP values (~1-2 min)..."):
                explainer   = get_shap_explainer(model, processed["X_train"])
                shap_values, X_flat = compute_shap_values(
                    explainer, processed["X_test"], n_samples=10
                )
                st.session_state["shap_values"] = shap_values
            st.success("✅ SHAP analysis complete!")

        shap_values = st.session_state.get("shap_values")
        if shap_values is not None:
            feature_names = processed["feature_names"]
            n_features    = len(feature_names)

            st.subheader("1️⃣ Global Feature Importance")
            importance_df = get_feature_importance(shap_values, feature_names, window_size)
            fig_imp = plot_feature_importance(importance_df)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.dataframe(importance_df, use_container_width=True)

            st.subheader("2️⃣ Temporal Importance")
            temporal = get_temporal_shap(shap_values, window_size, n_features)
            fig_temp = plot_temporal_importance(temporal)
            st.plotly_chart(fig_temp, use_container_width=True)

            st.subheader("3️⃣ Single Prediction Waterfall")
            sample_idx = st.slider("Select test sample", 0, len(shap_values) - 1, 0)
            fig_wf     = plot_shap_waterfall(
                np.array(shap_values)[sample_idx], feature_names, window_size
            )
            st.plotly_chart(fig_wf, use_container_width=True)

# ================= TAB 6: MULTI-STEP FORECAST =================
with tab6:
    st.subheader("📈 Multi-Step Forecasting with MC Dropout")

    model     = st.session_state.get("model")
    processed = st.session_state.get("processed")

    if model is None or processed is None:
        st.info("Train the model first (Tab 1).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            n_steps    = st.slider("Forecast Steps",      1, 300, 5)
        with col2:
            mc_samples = st.slider("MC Dropout Samples", 10,  50, 30)

        if st.button("Run Multi-Step Forecast"):
            with st.spinner(f"Forecasting {n_steps} steps ahead..."):
                cpu_scaler  = processed.get("cpu_scaler")
                X_seed      = processed["X_test"][-1:].copy()
                mean_preds, std_preds = mc_multistep_forecast(
                    model, X_seed, n_steps, cpu_scaler, n_samples=mc_samples
                )
                last_known_scaled = processed["X_test"][-60:, -1, 0]
                last_known        = cpu_scaler.inverse_transform(
                    last_known_scaled.reshape(-1, 1)
                ).flatten()
                st.session_state["ms_mean"]       = mean_preds
                st.session_state["ms_std"]        = std_preds
                st.session_state["ms_last_known"] = last_known
                st.session_state["ms_n_steps"]    = n_steps
            st.success(f"✅ {n_steps}-step forecast complete!")

        mean_preds  = st.session_state.get("ms_mean")
        std_preds   = st.session_state.get("ms_std")
        last_known  = st.session_state.get("ms_last_known")
        saved_steps = st.session_state.get("ms_n_steps", n_steps)

        if mean_preds is not None:
            cols = st.columns(min(saved_steps, 5))
            for i in range(min(saved_steps, 5)):
                cols[i].metric(f"t+{i+1}", f"{mean_preds[i]:.4f}", f"±{std_preds[i]:.4f}")
            fig_ms  = plot_multistep_forecast(mean_preds, std_preds, last_known, saved_steps)
            st.plotly_chart(fig_ms, use_container_width=True)
            fig_unc = plot_step_uncertainty(mean_preds, std_preds)
            st.plotly_chart(fig_unc, use_container_width=True)

# ================= TAB 7: MLFLOW =================
with tab7:
    st.subheader("🧪 MLflow Experiment Tracking")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Experiment Runs"):
            st.session_state["mlflow_runs"] = get_all_runs()
            st.session_state["mlflow_best"] = get_best_run()
    with col2:
        if st.button("Open MLflow UI"):
            st.info("Run in terminal: `mlflow ui` → open http://localhost:5000")

    best = st.session_state.get("mlflow_best") or get_best_run()
    if best is not None:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Model",        best["model_type"])
        b2.metric("Best RMSE",    f"{best['lstm_rmse']:.6f}")
        b3.metric("Best MAE",     f"{best['lstm_mae']:.6f}")
        b4.metric("Coverage 90%",
                  f"{best['coverage_95']:.4f}"
                  if best["coverage_95"] == best["coverage_95"] else "N/A")
        if best["params"]:
            st.json(best["params"])

    runs_df = st.session_state.get("mlflow_runs") or get_all_runs()
    if runs_df is not None and not runs_df.empty:
        st.dataframe(runs_df, use_container_width=True)

# ================= TAB 8: ABLATION =================
with tab8:
    st.subheader("🔬 Ablation Study")
    st.caption("Trains 4 LSTM models — each adding one feature group — to measure each group's contribution")

    exp_data = {
        "Experiment": list(ABLATION_CONFIGS.keys()),
        "Features Added": [
            "cpu_usage only",
            "+ hour_sin, hour_cos, dow_sin, dow_cos",
            "+ lag1, lag5, lag10",
            "+ roll_mean_10, roll_std_10"
        ],
        "Total Features": [1, 5, 8, 10]
    }
    st.dataframe(pd.DataFrame(exp_data), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        ablation_epochs = st.slider("Epochs per Experiment", 10, 50, 20)
    with col2:
        st.info("4 experiments × ~30s each ≈ 2 minutes total")

    if st.button("🔬 Run Ablation Study"):
        with st.spinner("Training 4 models..."):
            ablation_results = run_full_ablation(
                df, window_size=window_size,
                forecast_horizon=forecast_horizon,
                epochs=ablation_epochs
            )
            st.session_state["ablation_results"] = ablation_results
        st.success("✅ Ablation study complete!")

    ablation_results = st.session_state.get("ablation_results")
    if ablation_results is not None:
        st.dataframe(ablation_results, use_container_width=True)
        valid = ablation_results.dropna(subset=["RMSE"])
        if not valid.empty:
            colors_abl = ["#6b7280", "#3b82f6", "#f59e0b", "#22c55e"]
            fig_abl = go.Figure()
            fig_abl.add_trace(go.Bar(
                x=[f"Exp {chr(65+i)}" for i in range(len(valid))],
                y=valid["RMSE"].tolist(),
                marker_color=colors_abl[:len(valid)],
                text=[f"{v:.6f}" for v in valid["RMSE"]],
                textposition="auto"
            ))
            fig_abl.update_layout(
                title="Ablation Study — RMSE by Feature Group",
                template="plotly_dark", height=400
            )
            st.plotly_chart(fig_abl, use_container_width=True)
    else:
        st.info("Click 'Run Ablation Study' to start.")

# ================= TAB 9: ERROR ANALYSIS =================
with tab9:
    st.subheader("🔎 Error Analysis by CPU Load Region")

    results   = st.session_state.get("results")
    processed = st.session_state.get("processed")

    if results is None or processed is None:
        st.info("Train and evaluate your model first (Tab 1).")
    else:
        arrays = results.get("_arrays")
        if arrays is None:
            st.warning("Re-evaluate model to generate error arrays.")
        else:
            y_true          = arrays["true"]
            y_pred          = arrays["pred"]
            segment_results = segment_errors(y_true, y_pred)

            cols = st.columns(4)
            for i, (region, info) in enumerate(segment_results.items()):
                with cols[i % 4]:
                    st.metric(
                        f"{region} (n={info['Count']})",
                        f"RMSE: {info['RMSE']:.6f}",
                        f"{info['Pct']}% of data"
                    )

            st.plotly_chart(plot_error_by_region(segment_results),                    use_container_width=True)
            st.plotly_chart(plot_prediction_vs_actual_with_errors(y_true, y_pred, segment_results), use_container_width=True)
            st.plotly_chart(plot_rolling_error(y_true, y_pred),                       use_container_width=True)
            st.plotly_chart(plot_error_distribution_by_region(y_true, y_pred, segment_results),     use_container_width=True)

            worst_df = get_worst_predictions(y_true, y_pred, n=10)
            st.subheader("Top 10 Worst Predictions")
            st.dataframe(worst_df, use_container_width=True)
            st.download_button(
                "⬇️ Download Error Analysis CSV",
                worst_df.to_csv(index=False).encode(),
                "error_analysis.csv", "text/csv"
            )

# ================= TAB 10: ALERTS =================
with tab10:
    st.subheader("🚨 Real-Time Alert System")

    alert_system = st.session_state["alert_system"]
    model        = st.session_state.get("model")

    col1, col2 = st.columns(2)
    with col1:
        high_thresh = st.slider("High Alert Threshold",   0.5, 0.95, 0.8, 0.05)
    with col2:
        med_thresh  = st.slider("Medium Alert Threshold", 0.3, 0.7,  0.6, 0.05)

    alert_system.high_threshold   = high_thresh
    alert_system.medium_threshold = med_thresh

    if model is not None:
        import psutil as _psutil
        current_cpu = _psutil.cpu_percent(interval=0.1) / 100.0

        col_a, col_b = st.columns(2)
        with col_a:
            test_cpu = st.slider("Simulate Predicted CPU", 0.0, 1.0, current_cpu, 0.01)
        with col_b:
            test_uncertainty = st.slider("Simulate Uncertainty", 0.0, 0.2, 0.03, 0.01)

        if st.button("🚨 Check Alert"):
            alert = alert_system.check_prediction(test_cpu, current_cpu, test_uncertainty)
            msg   = f"{alert['emoji']} **{alert['level']}** — {alert['message']}\n\n**Action:** {alert['action']}"
            if alert["level"] == "HIGH":
                st.error(msg)
                sent = send_slack_alert(
                    alert["level"], test_cpu, current_cpu, alert["action"]
                )
                st.success("✅ Slack alert sent!") if sent else st.warning(
                    "⚠️ Slack not configured"
                )
            elif alert["level"] == "MEDIUM":
                st.warning(msg)
            else:
                st.success(msg)
    else:
        st.info("Train your model first (Tab 1).")

    summary = alert_system.get_alert_summary()
    if summary["total"] > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Alerts", summary["total"])
        c2.metric("🔴 High",      summary["high"])
        c3.metric("🟡 Medium",    summary["medium"])
        c4.metric("🟢 Normal",    summary["normal"])
