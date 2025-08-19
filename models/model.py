import tensorflow as tf
import keras
from keras.models import Model
from models.feature_extractor import FeatureExtractor
from models.fusion import Fusion
from models.transformer import Transformer

class ModelBuilder(Model):
    def __init__(self, num_classes=1, num_frames=8, embed_dims=256, num_heads=8,
                 ff_dim=1024, num_transformer_layers=3, dropout_rate=0.1, use_spatial_attention=True,
                 freeze_ratio=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.use_spartial_attention = use_spatial_attention

        self.feature_extractor = FeatureExtractor(freeze_ratio=freeze_ratio)
        self.fusion = Fusion(embed_dims=embed_dims)

        self.transformer_layers = [
            Transformer(embed_dims//num_heads,
                        num_heads, ff_dim,
                        dropout=dropout_rate,
                        use_spatial_attention=use_spatial_attention)
            for _ in range(num_transformer_layers)
        ]

        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(dropout_rate)

        self.fc = keras.layers.Dense(self.num_classes, activation="sigmoid")

    def call(self, inputs, training=False):
        xcep_feat, eff_feat = self.feature_extractor(inputs, training=training)
        fused = self.fusion([xcep_feat, eff_feat], training=training)

        x = fused
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        x = self.pooling(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)

        return x

    def create_model(self):
        inputs = keras.Input(shape=(self.num_frames, 224, 224, 3))
        outputs = self.call(inputs)
        return Model(inputs=inputs, outputs=outputs, name="DeepfakeDetectionModel")
