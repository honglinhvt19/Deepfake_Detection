import tensorflow as tf
from keras.models import Model
from .feature_extractor import FeatureExtractor
from .fusion import Fusion
from .transformer import Transformer

class ModelBuilder():
    def __init__(self, num_classes=1, num_frames=4, embed_dims=256, num_heads=8,
                 ff_dim=1024, num_transformer_layers=3, dropout_rate=0.1, use_spatial_attention=True):
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_layers = num_transformer_layers
        self.dropout_rate = dropout_rate
        self.use_spatial_attention = use_spatial_attention

    def build(self):
        inputs = tf.keras.Input(shape=(self.num_frames, 299, 299, 3))

        feature_extractor = FeatureExtractor()
        xception_features, efficientnet_features = feature_extractor(inputs)

        fusion = Fusion(embed_dims=self.embed_dims)
        fusion_features = fusion(xception_features, efficientnet_features)

        transformer = Transformer(num_classes=self.num_classes, num_frames=self.num_frames,
                                  embed_dims=self.embed_dims, num_heads=self.num_heads,
                                  ff_dim=self.ff_dim, num_transformer_layers=self.num_transformer_layers,
                                  dropout_rate=self.dropout_rate,
                                  use_spatial_attention=self.use_spatial_attention)
        
        outputs = transformer(fusion_features)

        return Model(inputs=inputs, outputs=outputs, name='TransformerModel')
