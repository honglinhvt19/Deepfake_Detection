import tensorflow as tf
import keras
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Input
from keras.models import Model
from keras.applications import EfficientNetB0
from keras.initializers import RandomNormal, Constant
import math
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@keras.saving.register_keras_serializable(package="Custom")
class EfficientNet(Model):
    def __init__(self, num_classes=1000, freeze_layers=100, pretrained=True, **kwargs):
        super(EfficientNet, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.freeze_layers = freeze_layers

        self.base_model = EfficientNetB0(include_top=False,
                                        weights='imagenet' if pretrained else None,
                                        input_shape=(224, 224, 3)
                                        )
        
        for layer in self.base_model.layers[:freeze_layers]:
            layer.trainable = False
        for layer in self.base_model.layers[freeze_layers:]:
            layer.trainable = True

        self.global_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(1024, activation='relu', kernel_initializer='he_normal')
        self.bn1 = BatchNormalization()
        self.fc = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal')

        for layer in self.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.filters
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=math.sqrt(2. / n))
            elif isinstance(layer, BatchNormalization):
                layer.gamma_initializer = Constant(1.0)
                layer.beta_initializer = Constant(0.0)
            elif isinstance(layer, Dense):
                n = layer.units
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=math.sqrt(2. / n))
        
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.global_pool(x)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.fc(x)

        return x
    
    def extract_features(self, inputs, training=False):
        features = self.base_model(inputs, training=training)
        features = self.global_pool(features)
        # features = self.dense1(features)
        # features = self.bn1(features, training=training)

        return features
    
    def effecientnet(pretrained=True, **kwargs):
        model = EfficientNet(pretrained, **kwargs)

        return model
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes, 'freeze_layers': self.freeze_layers, 'pretrained': self.pretrained})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
         