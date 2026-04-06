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
from datetime import datetime

from notifications import send_slack_alert, send_test_message, SLACK_WEBHOOK_URL
from preprocessing import prepare_data
from lstm_model import build_lstm_model, train_model, mc_dropout_predict
from tcn_model import build_tcn_model
from evaluate import evaluate_model
from optuna_tuning import run_optuna
from anomaly_detection import get_anomaly_summary
from live_monitor import get_system_stats, build_live_feature_window, get_current_cpu
from pdf_report import generate_report
from ablation_study import run_full_ablation, ABLATION_CONFIGS
from unique_features import (
    NLInsightEngine, ExplainableAlertSystem,
    CarbonAwareScheduler, FederatedLearningSimulator
)
from advanced_features import (
    AlertSystem, AutoRetrainer, ConfidenceDecisionEngine,
    OnlineLearner, MultiServerMonitor
)
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

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Real-Time CPU Workload Forecasting",
    page_icon="🖥️",
    layout="wide"
)

st.title("🖥️ Real-Time CPU Workload Forecasting System")
st.caption("Research-grade LSTM/TCN/Transformer forecasting with MLflow tracking, anomaly detection, SHAP explainability, multi-step forecasting & live monitoring")

# ================= TABS =================
(tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9,
 tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17,
 tab18, tab19, tab20) = st.tabs([
    "📁 Data & Training",
    "📊 Evaluation & Comparison",
    "🔴 Live Monitor",
    "🚨 Anomaly Detection",
    "🔍 SHAP Explainability",
    "📈 Multi-Step Forecast",
    "🧪 MLflow Tracking",
    "🔬 Ablation Study",
    "🔎 Error Analysis",
    "🏗️ Architecture",
    "🚨 Alerts",
    "🔄 Auto Retrain",
    "🎯 Confidence",
    "🧠 Online Learning",
    "🖥️ Multi-Server",
    "💬 AI Insights",
    "🔍 Explainable Alerts",
    "🌱 Carbon-Aware",
    "🤝 Federated Learning",
    "🔮 Predict"
])

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("data/google_cluster_processed.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    return df

df = load_data()

# ================= INIT ADVANCED FEATURES =================
if "alert_system" not in st.session_state:
    st.session_state["alert_system"] = AlertSystem()
if "auto_retrainer" not in st.session_state:
    st.session_state["auto_retrainer"] = AutoRetrainer()
if "confidence_engine" not in st.session_state:
    st.session_state["confidence_engine"] = ConfidenceDecisionEngine()
if "online_learner" not in st.session_state:
    st.session_state["online_learner"] = OnlineLearner()
if "multi_server" not in st.session_state:
    st.session_state["multi_server"] = MultiServerMonitor(n_servers=5)
if "nl_engine" not in st.session_state:
    st.session_state["nl_engine"] = NLInsightEngine()
if "expl_alert" not in st.session_state:
    st.session_state["expl_alert"] = ExplainableAlertSystem()
if "carbon_scheduler" not in st.session_state:
    st.session_state["carbon_scheduler"] = CarbonAwareScheduler()
if "federated" not in st.session_state:
    st.session_state["federated"] = FederatedLearningSimulator(n_clients=3)

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Training Configuration")
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
window_size = st.sidebar.slider("Window Size", 30, 120, 60)   # FIX: default 60 to match training
forecast_horizon = st.sidebar.slider("Forecast Horizon", 1, 300, 1)
epochs = st.sidebar.slider("Epochs", 10, 100, 50)             # FIX: default 50 for better training
model_type = st.sidebar.selectbox("Model Architecture", ["LSTM", "TCN", "Transformer"])

# ================= SIDEBAR — EVALUATION SPEED OPTIONS =================
st.sidebar.divider()
st.sidebar.subheader("⚡ Evaluation Speed")
run_prophet_option = st.sidebar.checkbox(
    "Run Prophet baseline (slow — 20-30 min)",
    value=False,
    help="Uncheck for fast evaluation under 10 minutes. Prophet is optional."
)
st.sidebar.caption(
    "✅ Fast mode: ARIMA capped at 2000 pts, 6 order combos. "
    "Evaluation completes in ~10 min."
)

# ================= TAB 1: DATA & TRAINING =================
with tab1:
    st.subheader("📁 Dataset Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Avg CPU Usage", f"{df['cpu_usage'].mean():.4f}")
    col3.metric("Max CPU Usage", f"{df['cpu_usage'].max():.4f}")
    st.dataframe(df.head(20))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cpu_usage"],
        name="CPU Usage", line=dict(color="#ef4444", width=1)
    ))
    fig.update_layout(title="CPU Usage Over Time", height=350, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    if st.sidebar.button("Prepare Data"):
        with st.spinner("Preparing data..."):
            processed = prepare_data(
                df, window_size=window_size,
                forecast_horizon=forecast_horizon
            )
            st.session_state["processed"] = processed
        st.success(f"✅ Data prepared — {processed['X_train'].shape[0]:,} train samples, "
                   f"{processed['X_train'].shape[2]} features per timestep")

    if st.sidebar.button("Run Hyperparameter Tuning"):
        processed = st.session_state.get("processed")
        if processed is None:
            st.warning("Prepare data first!")
        else:
            with st.spinner("Running Optuna (20 trials)..."):
                study = run_optuna(processed, trials=20)
            st.json(study.best_params)
            st.session_state["best_params"] = study.best_params

    if st.sidebar.button("Train Model"):
        processed = st.session_state.get("processed")
        if processed is None:
            st.warning("Prepare data first!")
        else:
            num_features = processed["X_train"].shape[2]
            st.info(f"Training with {num_features} features, window={window_size}")
            best_params = st.session_state.get("best_params")
            if best_params is None:
                lstm_units, dropout, lr, batch = [128, 64], 0.25, 0.001, 256
            else:
                lstm_units = [best_params["units1"], best_params["units2"]]
                dropout = best_params["dropout"]
                lr = best_params["lr"]
                batch = best_params["batch_size"]

            with st.spinner(f"Training {model_type} model..."):
                if model_type == "LSTM":
                    model = build_lstm_model(
                        window_size=window_size, num_features=num_features,
                        forecast_horizon=forecast_horizon, lstm_units=lstm_units,
                        dropout_rate=dropout, learning_rate=lr
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

                history = train_model(
                    model,
                    processed["X_train"], processed["y_train"],
                    processed["X_val"], processed["y_val"],
                    epochs=epochs, batch_size=batch
                )
                joblib.dump(model, "saved_model.pkl")
                try:
                    model.save("saved_model.keras")
                except Exception:
                    pass
                st.session_state["model"] = model
                st.session_state["history"] = history

            st.success(f"✅ {model_type} model trained and saved! Features: {num_features}")

            hist = history.history
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=hist["loss"], name="Train Loss", line=dict(color="#ef4444")))
            fig_loss.add_trace(go.Scatter(y=hist["val_loss"], name="Val Loss", line=dict(color="#3b82f6")))
            fig_loss.update_layout(title="Training Loss Curve", template="plotly_dark", height=300)
            st.plotly_chart(fig_loss, use_container_width=True)

    if st.sidebar.button("Evaluate Model"):
        model = st.session_state.get("model")
        processed = st.session_state.get("processed")
        if model is None:
            st.warning("Train model first!")
        else:
            if run_prophet_option:
                st.info("⏳ Running with Prophet — expect 30-60 minutes...")
            else:
                st.info("⚡ Fast mode — expect ~10 minutes (Prophet skipped, ARIMA capped at 2000 pts)...")

            with st.spinner("Evaluating... (check terminal for progress)"):
                results = evaluate_model(
                    model,
                    processed["X_test"], processed["y_test"],
                    processed["scaler"], processed.get("cpu_scaler"),
                    run_prophet=run_prophet_option
                )
            st.session_state["results"] = results
            st.success("✅ Evaluation complete!")

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
        results = st.session_state.get("results")
        processed = st.session_state.get("processed")
        if results and processed:
            arrays = results.get("_arrays")
            if arrays:
                y_true = arrays["true"]
                y_pred = arrays["pred"]
                mc_mean = arrays.get("mc_mean", y_pred)
                mc_std = arrays.get("mc_std", np.zeros_like(y_pred))
                n = min(200, len(y_true))
                upper = mc_mean[:n] + 1.96 * mc_std[:n]
                lower = mc_mean[:n] - 1.96 * mc_std[:n]
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(y=y_true[:n], name="Actual", line=dict(color="white", width=2)))
                fig_fc.add_trace(go.Scatter(y=y_pred[:n], name="Prediction", line=dict(color="#ef4444", width=2, dash="dot")))
                fig_fc.add_trace(go.Scatter(y=upper, name="Upper CI", line=dict(color="#f59e0b", dash="dot", width=1)))
                fig_fc.add_trace(go.Scatter(
                    y=lower, name="Lower CI",
                    line=dict(color="#f59e0b", dash="dot", width=1),
                    fill="tonexty", fillcolor="rgba(245,158,11,0.1)"
                ))
                fig_fc.update_layout(title="Forecast with Uncertainty Bands", template="plotly_dark", height=450)
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
        c4.metric("Coverage 95%", f"{results['Diagnostics']['Coverage_95']:.3f}")

        if results["Diagnostics"].get("LSTM_beats_ARIMA"):
            pct = (results["ARIMA"]["RMSE"] - results["LSTM"]["RMSE"]) / results["ARIMA"]["RMSE"] * 100
            st.success(f"✅ LSTM beats ARIMA by {pct:.1f}% RMSE ({results['LSTM']['RMSE']:.6f} vs {results['ARIMA']['RMSE']:.6f})")
        else:
            st.error(f"❌ LSTM does NOT beat ARIMA — needs more training or better data")

        models    = ["LSTM", "Naive", "MovingAverage", "ARIMA"]
        rmse_vals = [results["LSTM"]["RMSE"], results["Naive"]["RMSE"],
                     results["MovingAverage"]["RMSE"], results["ARIMA"]["RMSE"]]
        mae_vals  = [results["LSTM"]["MAE"],  results["Naive"]["MAE"],
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
        fig_rmse.update_layout(title="RMSE Comparison — All Models", template="plotly_dark", height=380)
        st.plotly_chart(fig_rmse, use_container_width=True)

        if prophet_success:
            verdict = results["Diagnostics"].get("Prophet_verdict", "")
            beats   = results["Diagnostics"].get("LSTM_beats_Prophet")
            if beats:
                st.success(f"Prophet: {verdict}")
            else:
                st.info(f"Prophet: {verdict}")
        else:
            prophet_msg = results.get("Prophet", {}).get("verdict", "")
            if prophet_msg:
                st.info(f"ℹ️ {prophet_msg}")

        # ── CALIBRATION SECTION ───────────────────────────────────
        st.divider()
        st.subheader("📐 Uncertainty Calibration")
        st.caption("Reliability diagram, ECE, and sharpness — proves uncertainty bands are statistically valid")

        calibration = results.get("Calibration", {})

        if calibration:
            ece       = calibration.get("ECE", float("nan"))
            sharpness = calibration.get("Sharpness", float("nan"))
            coverage  = calibration.get("Coverage_95", float("nan"))
            quality   = calibration.get("ECE_quality", "")

            cal1, cal2, cal3 = st.columns(3)

            ece_color = (
                "normal" if ece < 0.05 else
                "off"    if ece < 0.10 else
                "inverse"
            )
            cal1.metric(
                label="ECE (Expected Calibration Error)",
                value=f"{ece:.4f}",
                delta=quality,
                delta_color=ece_color,
                help="Lower is better. < 0.05 = well calibrated."
            )
            cal2.metric(
                label="Sharpness (Avg 95% CI Width)",
                value=f"{sharpness:.4f}",
                help="Narrower is better IF the model is calibrated."
            )
            cal3.metric(
                label="Coverage @ 95% CI",
                value=f"{coverage:.4f}",
                help="Fraction of true values inside 95% CI. Target: ~0.95."
            )

            if ece < 0.05:
                st.success(f"✅ Well calibrated — ECE = {ece:.4f} < 0.05. Uncertainty bands accurately reflect prediction confidence.")
            elif ece < 0.10:
                st.warning(f"⚠️ Acceptable calibration — ECE = {ece:.4f}. Uncertainty is reasonable but could be improved.")
            else:
                st.error(f"❌ Poor calibration — ECE = {ece:.4f} > 0.10. Consider more MC samples or longer training.")

            diagram_path = calibration.get("diagram_path", "calibration_diagram.png")
            if os.path.exists(diagram_path):
                # FIX: use_column_width deprecated → use_container_width
                st.image(
                    diagram_path,
                    caption="Reliability diagram (left) and CI width distribution (right). Perfect calibration = diagonal line.",
                    use_container_width=True
                )
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
        else:
            st.info("Calibration metrics not found. Re-evaluate your model.")

        st.divider()
        st.subheader("📈 Statistical Diagnostics")
        diag = results["Diagnostics"]
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Ljung-Box p",  f"{diag['LjungBox_pvalue']:.4f}")
        d2.metric("WF RMSE",      f"{diag['WalkForward_RMSE']:.6f}")
        d3.metric("WF Std",       f"{diag['WalkForward_std']:.6f}")
        d4.metric("Coverage 95%", f"{diag['Coverage_95']:.4f}")
        d5.metric("DM p-value",   f"{diag['DieboldMariano_pvalue']:.6f}")

        if diag["DieboldMariano_pvalue"] < 0.05:
            st.success(
                f"✅ Diebold-Mariano test significant (p={diag['DieboldMariano_pvalue']:.4f} < 0.05) — "
                f"LSTM is statistically significantly better than ARIMA"
            )
        else:
            st.warning(
                f"⚠️ Diebold-Mariano p={diag['DieboldMariano_pvalue']:.4f} — "
                f"difference not statistically significant yet"
            )

        arima_order = results["ARIMA"].get("order", "unknown")
        st.info(
            f"ℹ️ ARIMA evaluated in fast mode: 2000 points, order={arima_order}. "
            f"Full dataset ARIMA would take 3+ hours — not needed for demo."
        )

        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(
            x=models, y=mae_vals, marker_color=colors,
            text=[f"{v:.6f}" for v in mae_vals], textposition="auto"
        ))
        fig_mae.update_layout(title="MAE Comparison — All Models", template="plotly_dark", height=380)
        st.plotly_chart(fig_mae, use_container_width=True)

        arrays = results.get("_arrays")
        if arrays is not None:
            st.subheader("📉 Residual Analysis")
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(x=arrays["residuals"], nbinsx=50, marker_color="#3b82f6", name="Residuals"))
            fig_res.update_layout(title="Residual Error Distribution", template="plotly_dark", height=300)
            st.plotly_chart(fig_res, use_container_width=True)

        export_df = pd.DataFrame({"Model": models, "RMSE": rmse_vals, "MAE": mae_vals})
        st.download_button(
            "⬇️ Download Results CSV",
            export_df.to_csv(index=False).encode(),
            "evaluation_results.csv", "text/csv"
        )

    else:
        st.info("Train and evaluate your model first.")

# ================= TAB 3: LIVE MONITOR =================
with tab3:
    st.subheader("🔴 Live System CPU Monitor")
    auto_refresh = st.toggle("Enable Auto-Refresh (every 2s)", value=False, key="live_refresh")
    stats = get_system_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPU Usage", f"{stats['cpu_total']}%")
    col2.metric("Memory Used", f"{stats['memory_used_pct']}%")
    col3.metric("Free Memory", f"{stats['memory_available_gb']:.1f} GB")
    col4.metric("CPU Cores", stats['cpu_core_count'])

    if stats["cpu_per_core"]:
        fig_cores = go.Figure()
        fig_cores.add_trace(go.Bar(
            x=[f"Core {i}" for i in range(len(stats["cpu_per_core"]))],
            y=stats["cpu_per_core"], marker_color="#ef4444"
        ))
        fig_cores.update_layout(title="Per-Core CPU Usage (%)", yaxis=dict(range=[0, 100]), template="plotly_dark", height=300)
        st.plotly_chart(fig_cores, use_container_width=True)

    model = st.session_state.get("model")
    processed = st.session_state.get("processed")

    if model is not None and processed is not None:
        st.subheader("Live LSTM Prediction")

        # Get expected feature count from trained model
        expected_features = processed["X_train"].shape[2]
        expected_window   = processed["X_train"].shape[1]

        if "live_history" not in st.session_state:
            st.session_state["live_history"] = []

        current_cpu = get_current_cpu()
        st.session_state["live_history"].append(current_cpu)

        if len(st.session_state["live_history"]) >= expected_window:
            try:
                # FIX: Build live window and validate shape before prediction
                live_window = build_live_feature_window(
                    st.session_state["live_history"],
                    window_size=expected_window
                )

                # Convert to numpy array and validate shape
                live_window = np.array(live_window)

                # FIX: Shape validation — catch mismatch before it crashes
                if live_window.shape != (expected_window, expected_features):
                    st.error(
                        f"⚠️ Feature mismatch: live_window has shape {live_window.shape}, "
                        f"but model expects ({expected_window}, {expected_features}). "
                        f"Check that live_monitor.py builds {expected_features} features."
                    )
                    st.info(
                        f"💡 Fix: Open live_monitor.py and ensure build_live_feature_window() "
                        f"returns exactly {expected_features} features per timestep."
                    )
                else:
                    # Reshape to (1, window, features) for model input
                    live_window = live_window.reshape(1, expected_window, expected_features)

                    # FIX: NaN check before prediction
                    if np.isnan(live_window).any():
                        st.warning("⚠️ NaN detected in live window — using last valid prediction.")
                        live_pred = st.session_state.get("last_live_pred", current_cpu)
                    else:
                        # FIX: Use MC Dropout for live prediction (mean of 30 samples)
                        import tensorflow as tf
                        X_tensor = tf.constant(live_window, dtype=tf.float32)
                        mc_preds_live = np.array([
                            model(X_tensor, training=True).numpy()[0][0]
                            for _ in range(30)
                        ])
                        mc_mean_live = float(np.clip(np.mean(mc_preds_live), 0.0, 1.0))
                        mc_std_live  = float(np.std(mc_preds_live))

                        # Inverse transform if cpu_scaler available
                        cpu_scaler = processed.get("cpu_scaler")
                        if cpu_scaler is not None:
                            mc_mean_live = float(np.clip(
                                cpu_scaler.inverse_transform([[mc_mean_live]])[0][0], 0.0, 1.0
                            ))
                            mc_std_live = mc_std_live * (cpu_scaler.data_max_[0] - cpu_scaler.data_min_[0])

                        live_pred = mc_mean_live
                        st.session_state["last_live_pred"] = live_pred

                        # 95% CI bounds
                        ci_lower = float(np.clip(live_pred - 1.96 * mc_std_live, 0.0, 1.0))
                        ci_upper = float(np.clip(live_pred + 1.96 * mc_std_live, 0.0, 1.0))

                        # Confidence: narrower CI = more confident
                        ci_width = ci_upper - ci_lower
                        confidence = float(np.clip(1.0 - (ci_width / 0.5), 0.05, 0.95))

                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("Current CPU", f"{current_cpu:.4f}")
                        col_b.metric("Predicted Next", f"{live_pred:.4f}")
                        col_c.metric("95% CI", f"[{ci_lower:.3f}, {ci_upper:.3f}]")
                        col_d.metric("Confidence", f"{confidence:.2f}")

                        # Spike alert
                        spike_threshold = 0.75
                        if live_pred > spike_threshold:
                            st.error(f"🔴 SPIKE PREDICTED — CPU forecast {live_pred:.1%} exceeds threshold {spike_threshold:.0%}")
                        else:
                            st.success(f"🟢 System stable — no spike predicted")

                    fig_live = go.Figure()
                    history_display = st.session_state["live_history"][-100:]
                    fig_live.add_trace(go.Scatter(
                        y=history_display, name="Live CPU",
                        line=dict(color="#ef4444", width=2)
                    ))
                    if not np.isnan(live_window).any():
                        # Show prediction point at end
                        fig_live.add_trace(go.Scatter(
                            x=[len(history_display)],
                            y=[live_pred],
                            mode="markers",
                            name="Prediction",
                            marker=dict(color="#f59e0b", size=12, symbol="diamond")
                        ))
                    fig_live.update_layout(title="Live CPU History + Prediction", template="plotly_dark", height=300)
                    st.plotly_chart(fig_live, use_container_width=True)

            except Exception as e:
                # FIX: Fallback — never crash the dashboard
                last_known = st.session_state.get("last_live_pred", current_cpu)
                st.warning(f"⚠️ Prediction error (using last known value: {last_known:.4f}): {e}")
        else:
            remaining = expected_window - len(st.session_state["live_history"])
            st.info(f"Collecting data — need {remaining} more readings before prediction starts.")
            st.progress(len(st.session_state["live_history"]) / expected_window)

    elif model is None:
        st.info("💡 Train your model first (Tab 1), then live prediction will appear here.")

    if auto_refresh:
        time.sleep(1)
        st.rerun()

# ================= TAB 4: ANOMALY DETECTION =================
with tab4:
    st.subheader("🚨 CPU Anomaly Detection")
    st.caption("Isolation Forest + Z-Score dual detection")
    sample_size = st.slider("Sample size", 100, min(len(df), 2000), min(500, len(df)), step=100)
    if st.button("Run Anomaly Detection"):
        with st.spinner("Detecting anomalies..."):
            cpu_sample = df["cpu_usage"].values[:sample_size]
            summary = get_anomaly_summary(cpu_sample)
        st.session_state["anomaly_summary"] = summary
        st.session_state["anomaly_cpu"] = cpu_sample

    summary = st.session_state.get("anomaly_summary")
    cpu_sample = st.session_state.get("anomaly_cpu")

    if summary is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Points", f"{summary['total_points']:,}")
        col2.metric("Anomalies Found", f"{summary['anomaly_count']:,}")
        col3.metric("Anomaly %", f"{summary['anomaly_pct']}%")
        col4.metric("Isolation Forest Flags", f"{summary.get('if_flags', 0):,}")
        if 'zscore_flags' in summary:
            st.metric("Z-Score Flags", f"{summary['zscore_flags']:,}")

        anomaly_mask = summary["combined_flags"]
        normal_idx = np.where(~anomaly_mask)[0]
        anomaly_idx = np.where(anomaly_mask)[0]

        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(x=normal_idx, y=cpu_sample[normal_idx], mode="lines", name="Normal", line=dict(color="#3b82f6", width=1)))
        fig_anom.add_trace(go.Scatter(x=anomaly_idx, y=cpu_sample[anomaly_idx], mode="markers", name="Anomaly", marker=dict(color="#ef4444", size=6, symbol="x")))
        fig_anom.update_layout(title="CPU Usage with Anomalies Highlighted", template="plotly_dark", height=400)
        st.plotly_chart(fig_anom, use_container_width=True)

        anomaly_vals = cpu_sample[anomaly_idx]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=anomaly_vals, nbinsx=20, marker_color="#ef4444", name="Anomaly Values"))
        fig_dist.update_layout(title="Anomaly Value Distribution", template="plotly_dark", height=300)
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Click 'Run Anomaly Detection' to start.")

# ================= TAB 5: SHAP EXPLAINABILITY =================
with tab5:
    st.subheader("🔍 SHAP Model Explainability")
    model = st.session_state.get("model")
    processed = st.session_state.get("processed")
    if model is None or processed is None:
        st.info("Train the model first.")
    else:
        if st.button("Run SHAP Analysis"):
            with st.spinner("Computing SHAP values (takes 1-2 min)..."):
                explainer = get_shap_explainer(model, processed["X_train"])
                shap_values, X_flat = compute_shap_values(explainer, processed["X_test"], n_samples=10)
                st.session_state["shap_values"] = shap_values
            st.success("✅ SHAP analysis complete!")

        shap_values = st.session_state.get("shap_values")
        if shap_values is not None:
            feature_names = processed["feature_names"]
            n_features = len(feature_names)

            st.subheader("1️⃣ Feature Importance")
            importance_df = get_feature_importance(shap_values, feature_names, window_size)
            fig_imp = plot_feature_importance(importance_df)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.dataframe(importance_df, use_container_width=True)

            st.subheader("2️⃣ Temporal Importance")
            temporal = get_temporal_shap(shap_values, window_size, n_features)
            fig_temp = plot_temporal_importance(temporal)
            st.plotly_chart(fig_temp, use_container_width=True)

            st.subheader("3️⃣ Single Prediction Explanation")
            sample_idx = st.slider("Select test sample", 0, len(shap_values) - 1, 0)
            fig_wf = plot_shap_waterfall(np.array(shap_values)[sample_idx], feature_names, window_size)
            st.plotly_chart(fig_wf, use_container_width=True)

# ================= TAB 6: MULTI-STEP FORECAST =================
with tab6:
    st.subheader("📈 Multi-Step Forecasting")
    model = st.session_state.get("model")
    processed = st.session_state.get("processed")
    if model is None or processed is None:
        st.info("Train the model first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            n_steps = st.slider("Forecast Steps", 1, 300, 5)
        with col2:
            mc_samples = st.slider("MC Dropout Samples", 10, 50, 30)

        if st.button("Run Multi-Step Forecast"):
            with st.spinner(f"Forecasting {n_steps} steps ahead..."):
                cpu_scaler = processed.get("cpu_scaler")
                X_seed = processed["X_test"][-1:].copy()
                mean_preds, std_preds = mc_multistep_forecast(model, X_seed, n_steps, cpu_scaler, n_samples=mc_samples)
                last_known_scaled = processed["X_test"][-60:, -1, 0]
                last_known = cpu_scaler.inverse_transform(last_known_scaled.reshape(-1, 1)).flatten()
                st.session_state["ms_mean"] = mean_preds
                st.session_state["ms_std"] = std_preds
                st.session_state["ms_last_known"] = last_known
                st.session_state["ms_n_steps"] = n_steps
            st.success(f"✅ {n_steps}-step forecast complete!")

        mean_preds = st.session_state.get("ms_mean")
        std_preds = st.session_state.get("ms_std")
        last_known = st.session_state.get("ms_last_known")
        saved_steps = st.session_state.get("ms_n_steps", n_steps)

        if mean_preds is not None:
            cols = st.columns(min(saved_steps, 5))
            for i in range(min(saved_steps, 5)):
                cols[i].metric(f"t+{i+1}", f"{mean_preds[i]:.4f}", f"±{std_preds[i]:.4f}")
            fig_ms = plot_multistep_forecast(mean_preds, std_preds, last_known, saved_steps)
            st.plotly_chart(fig_ms, use_container_width=True)
            fig_unc = plot_step_uncertainty(mean_preds, std_preds)
            st.plotly_chart(fig_unc, use_container_width=True)

# ================= TAB 7: MLFLOW TRACKING =================
with tab7:
    st.subheader("🧪 MLflow Experiment Tracking")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Experiment Runs"):
            st.session_state["mlflow_runs"] = get_all_runs()
            st.session_state["mlflow_best"] = get_best_run()
    with col2:
        if st.button("Open MLflow UI"):
            st.info("Run in terminal: `mlflow ui` → then open http://localhost:5000")

    best = st.session_state.get("mlflow_best") or get_best_run()
    if best is not None:
        st.subheader("🏆 Best Run")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Model", best["model_type"])
        b2.metric("Best RMSE", f"{best['lstm_rmse']:.6f}")
        b3.metric("Best MAE", f"{best['lstm_mae']:.6f}")
        b4.metric("Coverage 95%", f"{best['coverage_95']:.4f}" if best['coverage_95'] == best['coverage_95'] else "N/A")
        if best["params"]:
            st.json(best["params"])

    runs_df = st.session_state.get("mlflow_runs") or get_all_runs()
    if runs_df is not None and not runs_df.empty:
        st.dataframe(runs_df, use_container_width=True)
        if "LSTM RMSE" in runs_df.columns and runs_df["LSTM RMSE"].notna().any():
            fig_mlflow = go.Figure()
            fig_mlflow.add_trace(go.Bar(
                x=runs_df["Run Name"], y=runs_df["LSTM RMSE"],
                marker_color="#ef4444",
                text=[f"{v:.6f}" for v in runs_df["LSTM RMSE"]],
                textposition="auto"
            ))
            fig_mlflow.update_layout(title="RMSE Across All Runs", template="plotly_dark", height=380)
            st.plotly_chart(fig_mlflow, use_container_width=True)
        st.download_button(
            "⬇️ Download Runs CSV",
            runs_df.to_csv(index=False).encode(),
            "mlflow_runs.csv", "text/csv"
        )

# ================= TAB 8: ABLATION STUDY =================
with tab8:
    st.subheader("🔬 Ablation Study")
    st.caption("Systematically measure the contribution of each feature group to model performance")
    st.info("We train 4 versions of LSTM — each adding one more feature group. This shows exactly which features improve performance.")

    exp_data = {
        "Experiment": list(ABLATION_CONFIGS.keys()),
        "Features Added": ["cpu_usage only", "+ hour_sin, hour_cos, dow_sin, dow_cos", "+ lag1, lag5, lag10", "+ roll_mean_10, roll_std_10"],
        "Total Features": [1, 5, 8, 10]
    }
    st.dataframe(pd.DataFrame(exp_data), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        ablation_epochs = st.slider("Training Epochs per Experiment", 10, 50, 20)
    with col2:
        st.info("4 experiments × ~30s each ≈ 2 minutes total")

    if st.button("🔬 Run Ablation Study"):
        with st.spinner("Running 4 experiments..."):
            progress = st.progress(0)
            ablation_results = run_full_ablation(df, window_size=window_size, forecast_horizon=forecast_horizon, epochs=ablation_epochs)
            progress.progress(100)
            st.session_state["ablation_results"] = ablation_results
        st.success("✅ Ablation study complete!")

    ablation_results = st.session_state.get("ablation_results")
    if ablation_results is not None:
        st.subheader("📊 Results Table")
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
            fig_abl.update_layout(title="Ablation Study — RMSE by Feature Group", template="plotly_dark", height=400)
            st.plotly_chart(fig_abl, use_container_width=True)

            if len(valid) > 1:
                baseline_rmse = valid["RMSE"].iloc[0]
                improvements = [(baseline_rmse - r) / baseline_rmse * 100 for r in valid["RMSE"]]
                fig_imp_abl = go.Figure()
                fig_imp_abl.add_trace(go.Scatter(
                    x=[f"Exp {chr(65+i)}" for i in range(len(valid))],
                    y=improvements, mode="lines+markers",
                    line=dict(color="#22c55e", width=3), marker=dict(size=10)
                ))
                fig_imp_abl.add_hline(y=0, line_dash="dash", line_color="#6b7280", annotation_text="Baseline")
                fig_imp_abl.update_layout(title="RMSE Improvement Over Baseline (%)", template="plotly_dark", height=350)
                st.plotly_chart(fig_imp_abl, use_container_width=True)

            best_exp = valid.loc[valid["RMSE"].idxmin(), "Experiment"]
            best_rmse = valid["RMSE"].min()
            worst_rmse = valid["RMSE"].max()
            total_improvement = (worst_rmse - best_rmse) / worst_rmse * 100
            st.success(f"🔑 Best: {best_exp} with RMSE = {best_rmse:.6f}. Feature engineering improved RMSE by {total_improvement:.1f}%")
            st.download_button("⬇️ Download Ablation CSV", ablation_results.to_csv(index=False).encode(), "ablation_results.csv", "text/csv")
    else:
        st.info("Click 'Run Ablation Study' to start.")

# ================= TAB 9: ERROR ANALYSIS =================
with tab9:
    st.subheader("🔎 Error Analysis")
    results = st.session_state.get("results")
    processed = st.session_state.get("processed")

    if results is None or processed is None:
        st.info("Train and evaluate your model first.")
    else:
        arrays = results.get("_arrays")
        if arrays is None:
            st.warning("Re-evaluate your model to generate error arrays.")
        else:
            y_true = arrays["true"]
            y_pred = arrays["pred"]
            segment_results = segment_errors(y_true, y_pred)

            st.subheader("Error Summary by Region")
            cols = st.columns(4)
            for i, (region, info) in enumerate(segment_results.items()):
                with cols[i % 4]:
                    st.metric(f"{region} (n={info['Count']})", f"RMSE: {info['RMSE']:.6f}", f"{info['Pct']}% of data")

            fig1 = plot_error_by_region(segment_results)
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = plot_prediction_vs_actual_with_errors(y_true, y_pred, segment_results)
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = plot_rolling_error(y_true, y_pred)
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = plot_error_distribution_by_region(y_true, y_pred, segment_results)
            st.plotly_chart(fig4, use_container_width=True)

            worst_df = get_worst_predictions(y_true, y_pred, n=10)
            st.subheader("Top 10 Worst Predictions")
            st.dataframe(worst_df, use_container_width=True)

            spike_rmse = segment_results["Spike"]["RMSE"]
            normal_rmse = segment_results["Normal"]["RMSE"]
            if spike_rmse > 0 and normal_rmse > 0:
                spike_factor = spike_rmse / normal_rmse
                st.error(f"⚠️ Model makes {spike_factor:.1f}x larger errors during spikes. Expected behavior — sudden load changes are hard to predict.")

            st.download_button("⬇️ Download Error Analysis CSV", worst_df.to_csv(index=False).encode(), "error_analysis.csv", "text/csv")

# ================= TAB 10: ARCHITECTURE =================
with tab10:
    st.subheader("🏗️ System Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📥 Input Pipeline**")
        st.markdown("- psutil live CPU data\n- google_cluster_processed.csv (60K rows)\n- 1-second sampling interval")
        st.markdown("**⚙️ Preprocessing**")
        st.markdown("- Linear interpolation\n- Leakage-safe MinMaxScaler\n- 12 engineered features\n- Sliding window (60 steps)")
    with col2:
        st.markdown("**🧠 Models**")
        st.markdown("- LSTM (Optuna tuned)\n- TCN (causal convolution)\n- Transformer (multi-head attention)\n- MC Dropout uncertainty (Gal & Ghahramani 2016)")
        st.markdown("**📊 Evaluation**")
        st.markdown("- RMSE, MAE\n- Walk-Forward validation\n- Ljung-Box, Diebold-Mariano\n- ECE calibration\n- Reliability diagram")
    with col3:
        st.markdown("**🚀 Outputs**")
        st.markdown("- Live monitoring dashboard\n- Anomaly detection\n- SHAP explainability\n- Multi-step forecast\n- Ablation study\n- Error analysis\n- Docker deployment\n- FastAPI REST API")
    st.divider()
    st.markdown("**🛠️ Tech Stack**")
    cols = st.columns(5)
    cols[0].info("TensorFlow/Keras")
    cols[1].info("Streamlit + Plotly")
    cols[2].info("Optuna + SHAP")
    cols[3].info("MLflow + Docker")
    cols[4].info("FastAPI + psutil")

# ================= TAB 11: ALERTS =================
with tab11:
    st.subheader("🚨 Real-Time Alert System")
    alert_system = st.session_state["alert_system"]
    model = st.session_state.get("model")

    col1, col2 = st.columns(2)
    with col1:
        high_thresh = st.slider("High Alert Threshold", 0.5, 0.95, 0.8, 0.05)
    with col2:
        med_thresh = st.slider("Medium Alert Threshold", 0.3, 0.7, 0.6, 0.05)

    alert_system.high_threshold = high_thresh
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
            if alert["level"] == "HIGH":
                st.error(f"{alert['emoji']} **{alert['level']}** — {alert['message']}\n\n**Action:** {alert['action']}")
                sent = send_slack_alert(alert["level"], test_cpu, current_cpu, alert["action"])
                if sent:
                    st.success("✅ Slack alert sent!")
                else:
                    st.warning("⚠️ Slack not configured — check webhook URL in sidebar")
            elif alert["level"] == "MEDIUM":
                st.warning(f"{alert['emoji']} **{alert['level']}** — {alert['message']}\n\n**Action:** {alert['action']}")
            else:
                st.success(f"{alert['emoji']} **{alert['level']}** — {alert['message']}\n\n**Action:** {alert['action']}")

    summary = alert_system.get_alert_summary()
    if summary["total"] > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Alerts", summary["total"])
        c2.metric("🔴 High", summary["high"])
        c3.metric("🟡 Medium", summary["medium"])
        c4.metric("🟢 Normal", summary["normal"])
        if alert_system.alert_history:
            hist_df = pd.DataFrame(alert_system.alert_history)[["timestamp", "level", "predicted", "current", "action"]]
            st.dataframe(hist_df.tail(10), use_container_width=True)

# ================= TAB 12: AUTO RETRAIN =================
with tab12:
    st.subheader("🔄 Auto Retraining System")
    retrainer = st.session_state["auto_retrainer"]
    status = retrainer.get_buffer_status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buffer Size", f"{status['buffer_size']:,}")
    c2.metric("Threshold", f"{status['threshold']:,}")
    c3.metric("Buffer Full", f"{status['pct_full']}%")
    c4.metric("Hours Since Retrain", f"{status['hours_since_retrain']}h")
    st.progress(min(status['pct_full'] / 100, 1.0))

    col1, col2 = st.columns(2)
    with col1:
        n_new = st.slider("Simulate new CPU readings", 10, 500, 100)
    with col2:
        if st.button("➕ Add New Data to Buffer"):
            import psutil as _psutil
            for _ in range(n_new):
                cpu = _psutil.cpu_percent(interval=0) / 100.0
                cpu += np.random.normal(0, 0.05)
                cpu = float(np.clip(cpu, 0, 1))
                retrainer.add_new_data(cpu)
            st.success(f"✅ Added {n_new} new readings!")
            st.rerun()

    should, reasons = retrainer.should_retrain()
    if should:
        st.warning("⚠️ Retraining recommended!\n\n" + "\n".join(f"- {r}" for r in reasons))
    else:
        st.success("✅ Model is up to date. No retraining needed.")

    hist = retrainer.get_retrain_history()
    if not hist.empty:
        st.dataframe(hist, use_container_width=True)

# ================= TAB 13: CONFIDENCE =================
with tab13:
    st.subheader("🎯 Confidence-Based Decision Engine")
    engine = st.session_state["confidence_engine"]
    col1, col2, col3 = st.columns(3)
    with col1:
        pred_cpu = st.slider("Predicted CPU", 0.0, 1.0, 0.75, 0.01)
    with col2:
        uncertainty = st.slider("Uncertainty (std)", 0.0, 0.2, 0.03, 0.005)
    with col3:
        curr_cpu_val = st.slider("Current CPU", 0.0, 1.0, 0.5, 0.01)

    if st.button("🎯 Make Decision"):
        decision = engine.make_decision(pred_cpu, uncertainty, curr_cpu_val)
        if decision["decision"] in ["AUTO-SCALE UP"]:
            st.error(f"{decision['icon']} **{decision['decision']}**\n\nAction: {decision['action']}\n\nConfidence: {decision['confidence_pct']}% | {decision['confidence_level']}")
        elif decision["decision"] in ["HUMAN REVIEW", "WAIT & OBSERVE"]:
            st.warning(f"{decision['icon']} **{decision['decision']}**\n\nAction: {decision['action']}\n\nConfidence: {decision['confidence_pct']}% | {decision['confidence_level']}")
        else:
            st.success(f"{decision['icon']} **{decision['decision']}**\n\nAction: {decision['action']}\n\nConfidence: {decision['confidence_pct']}% | {decision['confidence_level']}")

    stats = engine.get_decision_stats()
    if stats and stats["total_decisions"] > 0:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", stats["total_decisions"])
        c2.metric("⚡ Scale Up", stats["auto_scale_up"])
        c3.metric("💤 Scale Down", stats["auto_scale_down"])
        c4.metric("✅ Maintain", stats["maintain"])
        c5.metric("👀 Human Review", stats["human_review"])
        fig_pie = go.Figure(go.Pie(
            labels=["Auto Scale Up", "Auto Scale Down", "Maintain", "Human Review"],
            values=[stats["auto_scale_up"], stats["auto_scale_down"], stats["maintain"], stats["human_review"]],
            marker_colors=["#ef4444", "#3b82f6", "#22c55e", "#f59e0b"]
        ))
        fig_pie.update_layout(title="Decision Distribution", template="plotly_dark", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

# ================= TAB 14: ONLINE LEARNING =================
with tab14:
    st.subheader("🧠 Online Learning")
    learner = st.session_state["online_learner"]
    model = st.session_state.get("model")
    processed = st.session_state.get("processed")
    status = learner.get_status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Updates", status["total_updates"])
    c2.metric("Buffer Size", status["buffer_size"])
    c3.metric("Update Interval", status["update_interval"])
    c4.metric("Last Loss", f"{status['last_loss']:.6f}" if status["last_loss"] else "N/A")

    if model is None or processed is None:
        st.info("Train your model first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            n_online_samples = st.slider("New samples to learn from", 10, 200, 50)
        with col2:
            online_lr = st.select_slider("Learning Rate", options=[0.00001, 0.0001, 0.001], value=0.0001)
            learner.learning_rate = online_lr

        if st.button("🧠 Run Online Update"):
            with st.spinner("Updating model..."):
                idx = np.random.choice(len(processed["X_test"]), min(n_online_samples, len(processed["X_test"])), replace=False)
                X_new = processed["X_test"][idx]
                y_new = processed["y_test"][idx]
                success, result = learner.update_model(model, X_new, y_new)
                if success:
                    st.success(f"✅ Model updated! Loss: {result:.6f}")
                    st.session_state["model"] = model
                else:
                    st.error(f"Update failed: {result}")

        curve = learner.get_learning_curve()
        if not curve.empty:
            fig_online = go.Figure()
            fig_online.add_trace(go.Scatter(x=curve["update_number"], y=curve["loss"], mode="lines+markers", line=dict(color="#22c55e", width=2), name="Training Loss"))
            fig_online.update_layout(title="Loss After Each Online Update", template="plotly_dark", height=350)
            st.plotly_chart(fig_online, use_container_width=True)

# ================= TAB 15: MULTI-SERVER =================
with tab15:
    st.subheader("🖥️ Multi-Server Monitoring")
    monitor = st.session_state["multi_server"]
    model = st.session_state.get("model")
    processed = st.session_state.get("processed")
    auto_refresh_ms = st.toggle("Enable Auto-Refresh", value=False, key="multiserver_refresh")

    if model is not None and processed is not None:
        readings = monitor.predict_all_servers(model, processed, window_size)
    else:
        readings = monitor.get_server_readings()

    summary = monitor.get_fleet_summary(readings)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg CPU", f"{summary['avg_cpu']:.1%}")
    c2.metric("Max CPU", f"{summary['max_cpu']:.1%}")
    c3.metric("Min CPU", f"{summary['min_cpu']:.1%}")
    c4.metric("🔴 Critical", summary["critical_servers"])
    c5.metric("🟡 Warning", summary["warning_servers"])

    cols = st.columns(len(readings))
    for i, reading in enumerate(readings):
        with cols[i]:
            if reading["status"] == "Critical":
                st.error(f"**{reading['name']}**\n\n{reading['cpu_usage']:.1%}\n\n{reading['role']}")
            elif reading["status"] == "Warning":
                st.warning(f"**{reading['name']}**\n\n{reading['cpu_usage']:.1%}\n\n{reading['role']}")
            else:
                st.success(f"**{reading['name']}**\n\n{reading['cpu_usage']:.1%}\n\n{reading['role']}")

    fig_servers = go.Figure()
    colors_srv = ["#ef4444", "#3b82f6", "#f59e0b", "#22c55e", "#8b5cf6"]
    for i, reading in enumerate(readings):
        history = list(monitor.history[f"Server_{i+1}"])
        if history:
            fig_servers.add_trace(go.Scatter(y=history, name=reading["name"], line=dict(color=colors_srv[i % len(colors_srv)], width=2)))
    fig_servers.update_layout(title="CPU History — All Servers", template="plotly_dark", height=400)
    st.plotly_chart(fig_servers, use_container_width=True)
    server_df = pd.DataFrame(readings)[["name", "role", "cpu_usage", "status", "timestamp"]]
    st.dataframe(server_df, use_container_width=True)
    if auto_refresh_ms:
        time.sleep(2)
        st.rerun()

# ================= TAB 16: AI INSIGHTS =================
with tab16:
    st.subheader("💬 AI-Powered Natural Language Insights")
    nl_engine = st.session_state["nl_engine"]
    results = st.session_state.get("results")
    ablation_results = st.session_state.get("ablation_results")

    if results is None:
        st.info("Train and evaluate your model first.")
    else:
        if st.button("💬 Generate AI Insights"):
            with st.spinner("Analyzing results..."):
                insights = nl_engine.generate_insights(results=results, ablation_results=ablation_results)
                summary_nl = nl_engine.generate_summary_card(results)
                st.session_state["nl_insights"] = insights
                st.session_state["nl_summary"] = summary_nl

        insights = st.session_state.get("nl_insights")
        summary_nl = st.session_state.get("nl_summary")
        if insights:
            st.subheader("📋 Executive Summary")
            st.info(summary_nl)
            st.divider()
            for title, text in insights:
                with st.expander(f"**{title}**", expanded=True):
                    st.write(text)
            full_report = f"Executive Summary\n{'='*50}\n{summary_nl}\n\n"
            for title, text in insights:
                full_report += f"\n{title}\n{'-'*30}\n{text}\n"
            st.download_button("⬇️ Download Insights Report", full_report.encode(), "ai_insights_report.txt", "text/plain")

# ================= TAB 17: EXPLAINABLE ALERTS =================
with tab17:
    st.subheader("🔍 Explainable Alerts")
    expl_alert = st.session_state["expl_alert"]
    shap_values = st.session_state.get("shap_values")
    processed = st.session_state.get("processed")

    col1, col2 = st.columns(2)
    with col1:
        alert_thresh = st.slider("Alert Threshold", 0.5, 0.95, 0.8, 0.05)
        expl_alert.high_threshold = alert_thresh
    with col2:
        test_pred = st.slider("Simulate Predicted CPU", 0.0, 1.0, 0.85, 0.01)

    if st.button("🔍 Generate Explainable Alert"):
        feature_names = processed.get("feature_names", []) if processed else []
        w_size = window_size if processed else 60
        alert = expl_alert.explain_alert(predicted_cpu=test_pred, shap_values=shap_values, feature_names=feature_names, window_size=w_size)
        if alert:
            st.error(f"⚠️ **ALERT FIRED** — Predicted CPU: {test_pred:.1%}")
            st.write(alert["summary"])
            if alert["explanation"]:
                for feat in alert["explanation"]:
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Feature", feat["feature"])
                    col_b.metric("Contribution", f"{feat['contribution']:+.6f}")
                    col_c.metric("Direction", feat["direction"])
        else:
            st.success(f"✅ No alert — CPU {test_pred:.1%} is below threshold {alert_thresh:.1%}")

    if expl_alert.alert_log:
        for alert in expl_alert.alert_log[-5:]:
            with st.expander(f"🔴 Alert at {alert['timestamp']} — CPU: {alert['predicted_cpu']:.1%}"):
                st.write(alert["summary"])
                if alert["explanation"]:
                    st.dataframe(pd.DataFrame(alert["explanation"]), use_container_width=True)

# ================= TAB 18: CARBON-AWARE =================
with tab18:
    st.subheader("🌱 Carbon-Aware Workload Scheduling")
    scheduler = st.session_state["carbon_scheduler"]
    hour = datetime.now().hour
    carbon = scheduler.get_carbon_intensity(hour)
    renewable = scheduler.get_renewable_percentage(hour)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Hour", f"{hour:02d}:00")
    col2.metric("Carbon Intensity", f"{carbon:.0f} gCO2/kWh")
    col3.metric("Renewable Energy", f"{renewable}%")
    col4.metric("Carbon Saved", f"{scheduler.carbon_savings:.3f} kg")
    timeline = scheduler.get_carbon_timeline()
    fig_carbon = go.Figure()
    fig_carbon.add_trace(go.Scatter(x=timeline["Hour"], y=timeline["Carbon Intensity (gCO2/kWh)"], name="Carbon Intensity", line=dict(color="#ef4444", width=2), fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"))
    fig_carbon.add_trace(go.Scatter(x=timeline["Hour"], y=timeline["Renewable %"] * 3, name="Renewable % (scaled)", line=dict(color="#22c55e", width=2, dash="dot")))
    fig_carbon.add_vline(x=hour, line_color="#f59e0b", annotation_text="Now", line_dash="dash")
    fig_carbon.update_layout(title="24-Hour Carbon Intensity Forecast", template="plotly_dark", height=380)
    st.plotly_chart(fig_carbon, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        job_type = st.selectbox("Job Type", ["batch", "realtime"])
    with col2:
        can_delay = st.checkbox("Can be delayed?", value=True)
    with col3:
        predicted_cpu_val = st.slider("Predicted CPU", 0.0, 1.0, 0.5, 0.05)

    if st.button("🌱 Get Carbon-Aware Schedule"):
        result = scheduler.schedule_workload(predicted_cpu=predicted_cpu_val, job_type=job_type, can_delay=can_delay)
        decision = result["decision"]
        if decision["action"].startswith("RUN"):
            st.success(f"{decision['icon']} **{decision['action']}** — {decision['reason']}")
        elif decision["action"].startswith("DELAY"):
            st.warning(f"{decision['icon']} **{decision['action']}** — {decision['reason']}")
        else:
            st.error(f"{decision['icon']} **{decision['action']}** — {decision['reason']}")
        col_a, col_b = st.columns(2)
        col_a.metric("Carbon Intensity Now", f"{result['carbon_intensity']} gCO2/kWh")
        col_b.metric("Renewable Energy Now", f"{result['renewable_pct']}%")

# ================= TAB 19: FEDERATED LEARNING =================
with tab19:
    st.subheader("🤝 Federated Learning Simulation")
    federated = st.session_state["federated"]
    model = st.session_state.get("model")
    processed = st.session_state.get("processed")
    privacy = federated.get_privacy_summary()
    st.info(f"🔒 **Privacy Guarantee:** {privacy['data_shared']} | **Privacy Level:** {privacy['privacy_level']}")

    if model is None or processed is None:
        st.warning("Train your model first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            n_rounds = st.slider("Federated Rounds", 1, 5, 2)
        with col2:
            st.metric("Simulated Data Centers", federated.n_clients)
            for name in federated.client_names:
                st.caption(f"• {name}")

        if st.button("🤝 Run Federated Learning"):
            progress = st.progress(0)
            for round_num in range(1, n_rounds + 1):
                with st.spinner(f"Round {round_num}/{n_rounds}..."):
                    round_result, client_results = federated.run_federated_round(model, processed, round_num)
                    progress.progress(round_num / n_rounds)
                    st.write(f"**Round {round_num}:** Loss = {round_result['global_loss']:.6f} | Clients: {round_result['successful_clients']}/{round_result['n_clients']}")
            st.success("✅ Federated learning complete!")
            st.session_state["model"] = model

        hist = federated.get_round_history()
        if not hist.empty:
            st.dataframe(hist, use_container_width=True)
            if len(hist) > 1:
                fig_fed = go.Figure()
                fig_fed.add_trace(go.Scatter(x=hist["Round"], y=hist["Global Loss"], mode="lines+markers", line=dict(color="#22c55e", width=2), marker=dict(size=10), name="Global Loss"))
                fig_fed.update_layout(title="Federated Learning — Global Loss Per Round", template="plotly_dark", height=350)
                st.plotly_chart(fig_fed, use_container_width=True)

# ================= TAB 20: PREDICT =================
with tab20:
    st.subheader("🔮 Real-Time CPU Prediction")
    model = st.session_state.get("model")
    processed = st.session_state.get("processed")

    if model is None:
        st.info("Train the model first.")
    else:
        # FIX: Get actual feature count and window from trained model — not hardcoded
        expected_features = processed["X_train"].shape[2] if processed else 12
        expected_window   = processed["X_train"].shape[1] if processed else 60

        st.caption(f"Model expects: {expected_features} features × {expected_window} timesteps")

        col1, col2 = st.columns(2)
        with col1:
            cpu_input = st.number_input("CPU Usage (0-1)", 0.0, 1.0, 0.5, step=0.01)
            hour_input = st.slider("Hour of Day", 0, 23, 12)
            day_input = st.slider("Day of Week", 0, 6, 3)
        with col2:
            hour_sin = np.sin(2 * np.pi * hour_input / 24)
            hour_cos = np.cos(2 * np.pi * hour_input / 24)
            dow_sin  = np.sin(2 * np.pi * day_input / 7)
            dow_cos  = np.cos(2 * np.pi * day_input / 7)
            st.json({
                "cpu_input": cpu_input,
                "hour_sin": round(hour_sin, 4),
                "hour_cos": round(hour_cos, 4),
                "dow_sin":  round(dow_sin,  4),
                "dow_cos":  round(dow_cos,  4)
            })

        if st.button("🔮 Predict"):
            try:
                # FIX: Build feature vector dynamically to match expected_features
                # Base features always present (matches preprocessing.py order):
                # cpu_usage, hour_sin, hour_cos, dow_sin, dow_cos,
                # lag1, lag2, lag3, rolling_mean, rolling_std, + any extras
                base_features = [
                    cpu_input,      # cpu_usage
                    hour_sin,       # hour_sin
                    hour_cos,       # hour_cos
                    dow_sin,        # dow_sin
                    dow_cos,        # dow_cos
                    cpu_input,      # lag1 (approximate with current value)
                    cpu_input,      # lag2
                    cpu_input,      # lag3
                    cpu_input,      # rolling_mean (approximate)
                    0.01,           # rolling_std (approximate small value)
                    cpu_input,      # lag5 (if present)
                    cpu_input,      # lag10 (if present)
                ]

                # Use exactly as many features as the model expects
                features = np.array(base_features[:expected_features])

                # Pad with zeros if somehow fewer features than expected
                if len(features) < expected_features:
                    features = np.pad(features, (0, expected_features - len(features)))

                # Build window: tile single feature vector across all timesteps
                sample = np.tile(features, (expected_window, 1)).reshape(1, expected_window, expected_features)

                # FIX: Use MC Dropout for uncertainty-aware prediction
                import tensorflow as tf
                X_tensor = tf.constant(sample, dtype=tf.float32)
                mc_preds_tab20 = np.array([
                    model(X_tensor, training=True).numpy()[0][0]
                    for _ in range(30)
                ])
                raw_pred   = float(np.mean(mc_preds_tab20))
                raw_std    = float(np.std(mc_preds_tab20))

                cpu_scaler = processed.get("cpu_scaler") if processed else None
                if cpu_scaler is not None:
                    pred_inv = float(np.clip(cpu_scaler.inverse_transform([[raw_pred]])[0][0], 0.0, 1.0))
                    std_inv  = raw_std * (cpu_scaler.data_max_[0] - cpu_scaler.data_min_[0])
                else:
                    pred_inv = float(np.clip(raw_pred, 0.0, 1.0))
                    std_inv  = raw_std

                ci_lower = float(np.clip(pred_inv - 1.96 * std_inv, 0.0, 1.0))
                ci_upper = float(np.clip(pred_inv + 1.96 * std_inv, 0.0, 1.0))

                if pred_inv > 0.8:
                    st.error(f"🔴 Predicted CPU: {pred_inv:.4f} — HIGH LOAD | 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                elif pred_inv > 0.5:
                    st.warning(f"🟡 Predicted CPU: {pred_inv:.4f} — MODERATE | 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                else:
                    st.success(f"🟢 Predicted CPU: {pred_inv:.4f} — NORMAL | 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=pred_inv * 100,
                    title={"text": "Predicted CPU %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ef4444"},
                        "steps": [
                            {"range": [0, 50],  "color": "#22c55e"},
                            {"range": [50, 80], "color": "#f59e0b"},
                            {"range": [80, 100],"color": "#ef4444"}
                        ]
                    }
                ))
                fig_gauge.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(f"Expected input shape: (1, {expected_window}, {expected_features})")