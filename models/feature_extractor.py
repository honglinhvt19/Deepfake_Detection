import tensorflow as tf
import keras

class FeatureExtractor(keras.layers.Layer):
    def __init__(self, freeze_ratio=1.0, input_size=(224,224),  **kwargs):
        super().__init__(**kwargs)
        self.freeze_ratio = freeze_ratio
        self.input_size = input_size

        self.xcep_backbone = keras.applications.Xception(
            include_top=False, weights="imagenet", pooling="avg",
            input_shape=(299,299,3)
        )

        self.eff_backbone = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", pooling="avg",
            input_shape=(224,224,3)
        )

        self._apply_freeze_ratio(self.xcep_backbone, self.freeze_ratio)
        self._apply_freeze_ratio(self.eff_backbone, self.freeze_ratio)

    def _apply_freeze_ratio(self, model, ratio):
        n = len(model.layers)
        k = int(n * ratio)
        for i, layer in enumerate(model.layers):
            layer.trainable = False if i < k else True

    def call(self, inputs, training=False):
        shape = tf.shape(inputs)
        b, t = shape[0], shape[1]

        x = tf.reshape(inputs, (-1, shape[2], shape[3], 3))

        x_xcep = tf.image.resize(x, (299, 299), method=tf.image.ResizeMethod.BICUBIC)
        x_eff = tf.image.resize(x, (224, 224), method=tf.image.ResizeMethod.BICUBIC)

        xcep = self.xcep_backbone(x_xcep, training=training)
        eff  = self.eff_backbone(x_eff, training=training)

        xcep = tf.reshape(xcep, (b, t, xcep.shape[-1]))
        eff  = tf.reshape(eff,  (b, t, eff.shape[-1]))

        xcep.set_shape([None, None, 2048])
        eff.set_shape([None, None, 1280])

        return xcep, eff

    def get_config(self):
        config = super().get_config()
        config.update({
            "freeze_ratio": self.freeze_ratio,
            "input_size": self.input_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)