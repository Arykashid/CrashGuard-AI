"""
lstm_model.py — CrashGuard AI Core Model

FIXES vs previous version:
  1. build_lstm_model default dropout_rate: 0.60 → 0.30
     (0.60 killed 60% of neurons each MC pass → enormous variance
      → std_inv was huge → confidence formula gave ~0.19
      → 0.30 gives meaningful uncertainty without destroying signal)

  2. train_model spike_weight default: 8.0 → 12.0
     (combined with dynamic threshold from train.py, 12x gives
      better spike recall without destabilising normal prediction)

  3. train_model spike_threshold default: 0.75 → 0.80
     (train.py now passes the correct dynamic threshold computed
      from actual data distribution — this default is a safe fallback)

  4. mc_dropout_predict uses training=True correctly — unchanged, already good

  5. All other architecture choices unchanged:
     - LayerNorm ✅ (correct for MC Dropout)
     - Huber loss ✅ (robust to spike outliers)
     - clipnorm=1.0 ✅ (prevents gradient explosion)
     - EarlyStopping patience=20 ✅
"""

import tensorflow as tf
import numpy as np
import random
import os
import json
import joblib
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =============================================
# REPRODUCIBILITY
# =============================================
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# =============================================
# MC DROPOUT — ALWAYS ACTIVE
# =============================================
@tf.keras.utils.register_keras_serializable()
class MCDropout(tf.keras.layers.Layer):
    """
    Monte Carlo Dropout — always active at inference.

    Standard Dropout: disabled when training=False (Keras default)
    MCDropout:        uses tf.nn.dropout directly → ALWAYS active

    This is what makes uncertainty quantification work.
    Each of the N forward passes uses a different random dropout mask
    → different predictions → std across N passes = epistemic uncertainty.

    Reference: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"
    """

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        # tf.nn.dropout is unconditional — ignores training flag
        return tf.nn.dropout(inputs, rate=self.rate)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


# =============================================
# MODEL BUILDER
# =============================================
def build_lstm_model(
    window_size,
    num_features,
    forecast_horizon=1,
    lstm_units=(128, 64),
    dropout_rate=0.30,       # FIX: was 0.60 → caused confidence ~0.19
    learning_rate=0.001
):
    """
    Stacked LSTM with LayerNorm, Huber loss, MCDropout.

    dropout_rate=0.30 rationale:
        - 0.60 killed 60% of neurons each MC pass
        - std across passes was enormous → noisy uncertainty
        - 0.30 gives meaningful stochasticity without destroying signal
        - 0.25-0.35 is the standard range for MC Dropout in practice

    Expected after fixes (60K rows, batch=256, 15 features):
        RMSE:       0.10–0.14
        Coverage:   0.85–0.93
        Confidence: 0.50–0.75
    """
    set_seed()
    model = Sequential()

    # First LSTM block
    model.add(LSTM(
        lstm_units[0],
        input_shape=(window_size, num_features),
        return_sequences=len(lstm_units) > 1
    ))
    model.add(LayerNormalization())
    model.add(MCDropout(dropout_rate))

    # Middle blocks (if more than 2 layers)
    for units in lstm_units[1:-1]:
        model.add(LSTM(units, return_sequences=True))
        model.add(LayerNormalization())
        model.add(MCDropout(dropout_rate))

    # Final LSTM block
    if len(lstm_units) > 1:
        model.add(LSTM(lstm_units[-1]))
        model.add(LayerNormalization())
        model.add(MCDropout(dropout_rate))

    model.add(Dense(forecast_horizon))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
        loss="huber",
        metrics=["mae"]
    )
    return model


# =============================================
# TRAINING — WITH SPIKE WEIGHTING
# =============================================
def train_model(
    model, X_train, y_train,
    X_val, y_val,
    epochs=80,
    batch_size=256,           # default 256 — train.py should pass this explicitly
    spike_threshold=0.80,     # fallback default — train.py passes dynamic threshold
    spike_weight=12.0         # FIX: was 8.0 → 12x for stronger spike learning
):
    """
    Train LSTM with spike-aware sample weighting.

    IMPORTANT: train.py computes spike_threshold dynamically from actual
    data distribution and passes it here. The default 0.80 is only a
    fallback for direct calls (e.g., ablation study, quick tests).

    spike_weight=12.0:
        Spikes are ~3-8% of data. Without weighting, 1 spike point
        vs 20 normal points → model ignores spikes.
        With weight=12: 1 spike = 12 normal points in loss.
        This forces the model to pay attention to spike patterns.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=10, min_lr=1e-6, verbose=1
        )
    ]

    # ── Spike-aware sample weights ──────────────────────────
    y_flat = y_train.flatten()
    sample_weights = np.where(
        y_flat > spike_threshold,
        spike_weight,
        1.0
    )

    spike_count  = int((y_flat > spike_threshold).sum())
    normal_count = len(y_flat) - spike_count
    spike_pct    = spike_count / len(y_flat) * 100

    print(f"\n  Spike weighting:")
    print(f"  Threshold: {spike_threshold:.3f} (scaled)")
    print(f"  Spikes  : {spike_count:,} ({spike_pct:.1f}%) → weight {spike_weight}x")
    print(f"  Normal  : {normal_count:,} ({100-spike_pct:.1f}%) → weight 1x")

    if spike_count == 0:
        print(f"\n  ⚠️  WARNING: 0 spikes found at threshold {spike_threshold:.3f}")
        print(f"  This means spike weighting is OFF — model learns the mean only")
        print(f"  Fix: check train.py — it should compute dynamic threshold from data")
        sample_weights = None

    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        sample_weight=sample_weights,
        callbacks=callbacks,
        verbose=1
    )


# =============================================
# MC DROPOUT PREDICT
# =============================================
def mc_dropout_predict(model, X, n_samples=50):
    """
    MC Dropout inference — N stochastic forward passes.

    Uses model(X, training=True), NOT model.predict(X).
    model.predict() disables dropout internally even with MCDropout layers.
    model(X, training=True) keeps the call graph active → MCDropout fires.

    Returns:
        mean: shape (N, horizon) — average prediction
        std:  shape (N, horizon) — uncertainty (epistemic)
    """
    X_tensor = tf.constant(X, dtype=tf.float32)
    preds = np.array([
        model(X_tensor, training=True).numpy()
        for _ in range(n_samples)
    ])
    return preds.mean(axis=0), preds.std(axis=0)


# =============================================
# STANDARD PREDICT (no uncertainty)
# =============================================
def predict(model, X):
    """Point prediction — MC Dropout still fires (MCDropout is always-on)."""
    return model.predict(X, verbose=0)


# =============================================
# SAVE / LOAD
# =============================================
def save_model(model, path="saved_model"):
    try:
        model.save(f"{path}.keras")
        joblib.dump(model, f"{path}.pkl")
        print(f"Saved: {path}.keras + {path}.pkl")
    except Exception as e:
        print(f"Save warning: {e}")


def load_model(path="saved_model"):
    keras_path = f"{path}.keras"
    pkl_path   = f"{path}.pkl"
    if os.path.exists(keras_path):
        return tf.keras.models.load_model(
            keras_path, custom_objects={"MCDropout": MCDropout}
        )
    elif os.path.exists(pkl_path):
        with tf.keras.utils.custom_object_scope({"MCDropout": MCDropout}):
            return joblib.load(pkl_path)
    return None


# =============================================
# EXPERIMENT METADATA
# =============================================
def save_experiment_metadata(params, save_dir="experiments"):
    os.makedirs(save_dir, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Metadata saved: {path}")


# =============================================
# SELF-TEST
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("MCDropout Self-Test")
    print("=" * 60)

    WINDOW, FEATURES = 10, 15   # Updated to 15 features
    model = build_lstm_model(
        window_size=WINDOW,
        num_features=FEATURES,
        lstm_units=(32, 16),
        dropout_rate=0.30,      # FIX: was 0.60
    )
    model.summary()

    X_test = np.random.randn(5, WINDOW, FEATURES).astype(np.float32)
    mean, std = mc_dropout_predict(model, X_test, n_samples=30)

    print(f"\nMC mean shape: {mean.shape}")
    print(f"MC std  shape: {std.shape}")
    print(f"MC std  values: {std.flatten()}")
    print(f"MC std  mean:   {std.mean():.6f}")

    assert std.mean() > 0, (
        "FAIL: MC Dropout produced std=0 — dropout is NOT active at inference!"
    )
    print("\n[PASS] MC samples have std > 0 — dropout is active at inference.")

    # Spike weighting test
    print("\n--- Spike Weighting Test ---")
    X_train_test = np.random.randn(100, WINDOW, FEATURES).astype(np.float32)
    y_train_test = np.random.rand(100, 1).astype(np.float32)
    y_train_test[10:15] = 0.9
    X_val_test = np.random.randn(20, WINDOW, FEATURES).astype(np.float32)
    y_val_test = np.random.rand(20, 1).astype(np.float32)

    history = train_model(
        model, X_train_test, y_train_test,
        X_val_test, y_val_test,
        epochs=3, batch_size=32,
        spike_threshold=0.75, spike_weight=12.0
    )
    print("[PASS] train_model with spike weighting runs without error.")

    # Save/load test
    TEMP_PATH = "__mc_test_model"
    model.save(f"{TEMP_PATH}.keras")
    loaded = tf.keras.models.load_model(
        f"{TEMP_PATH}.keras", custom_objects={"MCDropout": MCDropout}
    )
    mean2, std2 = mc_dropout_predict(loaded, X_test, n_samples=30)
    assert std2.mean() > 0, (
        "FAIL: Loaded model has std=0 — MCDropout lost during save/load!"
    )
    print("[PASS] Save/load round-trip preserves MCDropout behavior.")

    os.remove(f"{TEMP_PATH}.keras")
    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
