import tensorflow as tf
from keras.layers import Concatenate, Dense, BatchNormalization
from keras.models import Model
from .feature_extractor import FeatureExtractor

class Fusion(Model):
    def __init__(self, embed_dims=512):
        super(Fusion, self).__init__()

        self.embed_dims = embed_dims
        self.concatenate = Concatenate()
        self.feature_projection = Dense(embed_dims, activation='relu', kernel_initializer='he_normal')
        self.bn_projection = BatchNormalization()

    def call(self, xception_features, efficientnet_features, training=False):
        combined_features = self.concatenate([xception_features, efficientnet_features]) # [batch_size, 2048 + 1280]
        combined_features = self.feature_projection(combined_features) # [batch_size, embed_dim]
        combined_features = self.bn_projection(combined_features, training=training)

        return combined_features
    