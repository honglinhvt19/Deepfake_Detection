import tensorflow as tf
import keras
from keras.layers import Concatenate, Dense, BatchNormalization, Layer
from .feature_extractor import FeatureExtractor

@keras.saving.register_keras_serializable(package="Custom")
class Fusion(Layer):
    def __init__(self, embed_dims=512, **kwargs):
        super(Fusion, self).__init__(**kwargs)
        self.embed_dims = embed_dims
        self.concat = Concatenate(axis=-1)
        self.feature_projection = Dense(self.embed_dims, kernel_initializer='he_normal', dtype='float32')
        self.bn_projection = BatchNormalization()

    def call(self, inputs, training=False):
        xception_features, efficientnet_features = inputs
        combined_features = self.concat([xception_features, efficientnet_features]) # [batch_size, 2048 + 1280]
        combined_features = self.feature_projection(combined_features) # [batch_size, embed_dim]
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
    