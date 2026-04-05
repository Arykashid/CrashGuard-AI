import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    Dropout,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model


# ---------------- TRANSFORMER BLOCK ----------------
def transformer_block(x, head_size, num_heads, ff_dim, dropout):

    # Attention
    attention = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(x, x)

    attention = Dropout(dropout)(attention)
    x = LayerNormalization(epsilon=1e-6)(x + attention)

    # Feed Forward
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(x.shape[-1])(ff)

    x = LayerNormalization(epsilon=1e-6)(x + ff)

    return x


# ---------------- BUILD TRANSFORMER MODEL ----------------
def build_transformer_model(
    window_size,
    num_features,
    forecast_horizon,
    head_size=64,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout=0.2,
):

    inputs = Input(shape=(window_size, num_features))

    x = inputs

    for _ in range(num_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(forecast_horizon)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model