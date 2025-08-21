import tensorflow as tf
import keras
from keras.layers import Concatenate, Dense, BatchNormalization, Layer
from keras.regularizers import l1

@keras.saving.register_keras_serializable(package="Custom")
class Fusion(Layer):
    def __init__(self, embed_dims=256, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.concat = Concatenate(axis=-1)
        self.bn_projection = BatchNormalization()
        self.feature_projection = None
        self.selection = None

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(f"Expected 2 inputs, got {len(input_shape) if isinstance(input_shape, (list, tuple)) else 1}")

        d1 = input_shape[0][-1]
        d2 = input_shape[1][-1]

        if d1 is None or d2 is None:
            raise ValueError(f"Input dimensions must be fully defined. Got d1={d1}, d2={d2}")

        total_dims = d1 + d2
        self.feature_projection = Dense(self.embed_dims, name="feature_projection")

        projection_input_shape = (input_shape[0][0], input_shape[0][1], total_dims)
        self.feature_projection.build(projection_input_shape)

        self.selection = Dense(self.embed_dims, activation='relu', kernel_regularizer=l1(0.01), name="feature_selection")
        selection_input_shape = (input_shape[0][0], input_shape[0][1], self.embed_dims)
        self.selection.build(selection_input_shape)

        bn_input_shape = (input_shape[0][0], input_shape[0][1], self.embed_dims)
        self.bn_projection.build(bn_input_shape)

        super().build(input_shape)

    def call(self, inputs, training=False):
        xception_features, efficientnet_features = inputs  # [B, T, D1], [B, T, D2]
        combined_features = self.concat([xception_features, efficientnet_features])  # [B, T, D1+D2]
        combined_features = self.feature_projection(combined_features)              # [B, T, embed_dim]
        combined_features = self.selection(combined_features)                       # Feature selection with L1
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

