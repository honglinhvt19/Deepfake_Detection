import tensorflow as tf
import keras
from keras.layers import Layer, MultiHeadAttention, Dropout, LayerNormalization, Dense

@keras.saving.register_keras_serializable(package="Custom")
class Transformer(Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, use_spatial_attention=True, name="Transformer", **kwargs):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_spatial_attention = use_spatial_attention

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=0.25, name="temporal_attention")
        self.dropout1 = Dropout(0.25, name="dropout_temporal")
        self.norm1 = LayerNormalization(epsilon=1e-6, name="norm_temporal")

        self.ffn = keras.Sequential([
            Dense(ff_dim, activation="relu", name="ffn_dense1"),
            Dense(head_size * num_heads, name="ffn_dense2"),
        ])
        self.dropout2 = Dropout(0.25, name="dropout_ffn")
        self.norm2 = LayerNormalization(epsilon=1e-6, name="norm_ffn")

        if self.use_spatial_attention:
            self.spatial_att = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=0.25, name="spatial_attention")
            self.spatial_norm = LayerNormalization(epsilon=1e-6, name="norm_spatial")

    def build(self, input_shape):
        _, t, d = input_shape
        if d != self.head_size * self.num_heads:
            raise ValueError(
                f"Embedding dim ({d}) phải bằng head_size * num_heads ({self.head_size * self.num_heads})"
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = inputs 

        # # ---------------- Spatial Attention ----------------
        # if self.use_spatial_attention:
        #     x_spatial = self.spatial_att(x, x, training=training)
        #     x = self.spatial_norm(x + x_spatial)

        # ---------------- Temporal Attention ----------------
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = tf.cast(x, tf.float32)
        attn_output = tf.cast(attn_output, tf.float32)
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = tf.cast(ffn_output, tf.float32)
        ffn_output = self.norm2(out1 + ffn_output)
        return ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_size": self.head_size,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "use_spatial_attention": self.use_spatial_attention})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
