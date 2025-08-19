import tensorflow as tf
import keras
from keras.layers import Layer, GlobalAveragePooling2D
from keras.applications import EfficientNetB0, Xception

@keras.saving.register_keras_serializable(package="Custom")
class FeatureExtractor(Layer):
    def __init__(self, freeze_ratio=1.0, **kwargs):
        super().__init__(**kwargs)
        self.freeze_ratio = freeze_ratio

        self.eff = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg")
        self.xcep = Xception(include_top=False, weights="imagenet", pooling="avg")

        for model in [self.eff, self.xcep]:
            num_layers = len(model.layers)
            freeze_until = int(num_layers * self.freeze_ratio)
            for layer in model.layers[:freeze_until]:
                layer.trainable = False
            for layer in model.layers[freeze_until:]:
                layer.trainable = True

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        num_frames = tf.shape(inputs)[1]
        inp_flat = tf.reshape(inputs, [-1, 224, 224, 3])  # [B*T, H, W, C]

        eff_feat = self.eff(inp_flat, training=training)   # (B*T, 1280)
        xcep_feat = self.xcep(inp_flat, training=training) # (B*T, 2048)

        eff_feat = tf.reshape(eff_feat, [batch_size, num_frames, -1])
        xcep_feat = tf.reshape(xcep_feat, [batch_size, num_frames, -1])
        return xcep_feat, eff_feat
    
    def get_config(self):
        config = super().get_config()
        config.update({"freeze_ratio": self.freeze_ratio})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
