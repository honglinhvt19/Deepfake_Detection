import tensorflow as tf
import keras
from keras.layers import Dense, LayerNormalization, Dropout, MultiHeadAttention, Layer
from .feature_extractor import FeatureExtractor
from .fusion import Fusion

@keras.saving.register_keras_serializable(package="Custom")
class Transformer(Layer):
    def __init__(self, num_classes=1, num_frames=4, embed_dims=256, num_heads=8,
                  ff_dim=1024, num_transformer_layers=3, dropout_rate=0.1, use_spatial_attention=True, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.num_transformer_layers = num_transformer_layers
        self.use_spatial_attention = use_spatial_attention
        self.temporal_transformer_layers = []
        self.spatial_attention = None
        self.spatial_norm = None
        self.global_pool = None
        self.classifier = None

    def build(self, input_shape):    
        self.temporal_transformer_layers = []
        for _ in range(self.num_transformer_layers):
            attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dims // self.num_heads)
            norm1 = LayerNormalization(epsilon=1e-6)
            dense1 = Dense(self.ff_dim, activation='relu', kernel_initializer='he_normal')
            dropout1 = Dropout(self.dropout_rate)
            dense2 = Dense(self.embed_dims, kernel_initializer='he_normal')
            dropout2 = Dropout(self.dropout_rate)
            norm2 = LayerNormalization(epsilon=1e-6)
            self.temporal_transformer_layers.append([attention, norm1, dense1, dropout1, dense2, dropout2, norm2])
        
        if self.use_spatial_attention:
            self.spatial_attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dims // self.num_heads)
            self.spatial_norm = LayerNormalization(epsilon=1e-6)

        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = Dense(self.num_classes, activation='sigmoid', kernel_initializer='he_normal', dtype='float32')
        super().build(input_shape)

    def call(self, inputs, training=False):
        # inputs shape: [batch_size, num_frames, embed_dims]
        x = inputs

        if self.use_spatial_attention:
            x = tf.transpose(x, [0, 2, 1])  # [batch_size, embed_dims, num_frames]
            x = self.spatial_attention(x, x, training=training)
            x = self.spatial_norm(x+x, training=training)
            x = tf.transpose(x, [0, 2, 1])

        for layer in self.temporal_transformer_layers:
            attn_output = layer[0](x, x, training=training)
            x = layer[1](x + attn_output, training=training)
            ff_output = layer[2](x, training=training)
            ff_output = layer[3](ff_output, training=training)
            ff_output = layer[4](ff_output, training=training)
            x = layer[5](x + ff_output, training=training)

        x = self.global_pool(x)  # [batch_size, embed_dims]
        x = self.classifier(x)  # [batch_size, num_classes]

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_frames': self.num_frames,
            'embed_dims': self.embed_dims,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_transformer_layers': self.num_transformer_layers,
            'dropout_rate': self.dropout_rate,
            'use_spatial_attention': self.use_spatial_attention
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    