"""
Temporal Convolution Network (TCN) Model
Research-grade architecture for time-series forecasting
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling1D
)


def build_tcn_model(
    window_size,
    num_features,
    forecast_horizon,
    filters=64,
    kernel_size=3,
    dropout_rate=0.2,
    learning_rate=0.001
):

    model = Sequential()

    # First convolution layer
    model.add(
        Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            activation="relu",
            input_shape=(window_size, num_features)
        )
    )

    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Second convolution layer
    model.add(
        Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            activation="relu"
        )
    )

    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Global pooling
    model.add(GlobalAveragePooling1D())

    model.add(Dense(32, activation="relu"))

    # Output layer
    model.add(Dense(forecast_horizon))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model