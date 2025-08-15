import tensorflow as tf
from keras.models import Model
from .feature_extractor import FeatureExtractor
from .fusion import Fusion
from .transformer import Transformer

class ModelBuilder(Model):
    def __init__(self, num_classes=1, num_frames=4, embed_dims=256, num_heads=8,
                 ff_dim=1024, num_transformer_layers=3, dropout_rate=0.1, use_spatial_attention=True, **kwargs):
        super(ModelBuilder, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_layers = num_transformer_layers
        self.dropout_rate = dropout_rate
        self.use_spatial_attention = use_spatial_attention

        self.feature_extractor = FeatureExtractor()
        self.fusion = Fusion(embed_dims=self.embed_dims)
        self.transformer = Transformer(
            num_classes=self.num_classes,
            num_frames=self.num_frames,
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_layers=self.num_transformer_layers,
            dropout_rate=self.dropout_rate,
            use_spatial_attention=self.use_spatial_attention
        )

    def call(self, inputs, training=None):
        xception_features, efficientnet_features = self.feature_extractor(inputs, training=training)
        fusion_features = self.fusion([xception_features, efficientnet_features], training=training)
        outputs = self.transformer(fusion_features, training=training)

        return outputs
    
    def create_model(self):
        inputs = tf.keras.Input(shape=(self.num_frames, 224, 224, 3), name='input_videos')
        outputs = self.call(inputs)
        return Model(inputs=inputs, outputs=outputs, name='DeepfakeDetectionModel')
    
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

