import tensorflow as tf
from tensorflow import keras
from keras.layers import TimeDistributed, GlobalAveragePooling2D

class FeatureExtractor(keras.layers.Layer):
    def __init__(self, freeze_ratio=1.0, **kwargs):
        super().__init__(**kwargs)

        # Backbone 1: Xception
        base_xcep = keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            pooling=None,
            input_shape=(224, 224, 3)
        )
        self.xception = keras.Sequential([
            base_xcep,
            GlobalAveragePooling2D()
        ])

        # Backbone 2: EfficientNetB0
        base_eff = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling=None,
            input_shape=(224, 224, 3)
        )
        self.efficientnet = keras.Sequential([
            base_eff,
            GlobalAveragePooling2D()
        ])

        # Wrap bằng TimeDistributed để áp dụng cho từng frame
        self.xception_td = TimeDistributed(self.xception)
        self.efficientnet_td = TimeDistributed(self.efficientnet)

        # Freeze tỷ lệ backbone (theo freeze_ratio)
        self._freeze_backbones(freeze_ratio)

    def _freeze_backbones(self, freeze_ratio):
        def freeze_layers(model, ratio):
            n_layers = len(model.layers[0].layers)  # lấy backbone gốc
            n_freeze = int(n_layers * ratio)
            for i, layer in enumerate(model.layers[0].layers):
                layer.trainable = False if i < n_freeze else True

        freeze_layers(self.xception_td, freeze_ratio)
        freeze_layers(self.efficientnet_td, freeze_ratio)

    def call(self, inputs, training=False):
        # inputs: [B, T, H, W, C]
        xcep_feat = self.xception_td(inputs, training=training)       # [B, T, D1]
        eff_feat = self.efficientnet_td(inputs, training=training)    # [B, T, D2]
        return xcep_feat, eff_feat

    def get_config(self):
        config = super().get_config()
        config.update({
            "freeze_ratio": self.xception_td.layers[0].trainable,
            "xception_td": self.xception_td.get_config(),
            "efficientnet_td": self.efficientnet_td.get_config()
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)