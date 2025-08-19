import tensorflow as tf
import keras
from keras.layers import Concatenate, Dense, BatchNormalization, Layer

@keras.saving.register_keras_serializable(package="Custom")
class Fusion(Layer):
    def __init__(self, embed_dims=256, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.concat = Concatenate(axis=-1)
        self.bn_projection = BatchNormalization()
        self.feature_projection = None

    def build(self, input_shape):
        d1 = input_shape[0][-1]
        d2 = input_shape[1][-1]
        in_dim = d1 + d2

        self.feature_projection = Dense(
            self.embed_dims,
            name="feature_projection"
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        xception_features, efficientnet_features = inputs  # [B, T, D1], [B, T, D2]
        combined_features = self.concat([xception_features, efficientnet_features])  # [B, T, D1+D2]
        combined_features = self.feature_projection(combined_features)              # [B, T, embed_dim]
        combined_features = self.bn_projection(combined_features, training=training)
        return combined_features
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.embed_dims)
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dims": self.embed_dims})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    