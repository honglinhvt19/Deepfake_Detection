import tensorflow as tf
import keras
from keras.layers import Layer, MultiHeadAttention, Dropout, LayerNormalization, Dense

@keras.saving.register_keras_serializable(package="Custom")
class Transformer(Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, use_spatial_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.use_spatial_attention = use_spatial_attention
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.ffn = keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(head_size * num_heads),
        ])
        self.dropout2 = Dropout(dropout)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        if self.use_spatial_attention:
            self.spatial_att = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
            self.spatial_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        x = inputs 

        # ---------------- Spatial Attention ----------------
        if self.use_spatial_attention:
            x_t = tf.transpose(x, [0, 2, 1])              # [B, D, T]
            x_t = self.spatial_att(x_t, x_t, training=training)
            x_t = self.spatial_norm(x_t + tf.transpose(x, [0, 2, 1]))
            x = tf.transpose(x_t, [0, 2, 1])              # [B, T, D]

        # ---------------- Temporal Attention ----------------
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = self.norm2(out1 + ffn_output)
        return ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({"use_spatial_attention": self.use_spatial_attention})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
