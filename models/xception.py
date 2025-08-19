import tensorflow as tf
import keras
from keras.layers import SeparableConv2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, ReLU, Add, BatchNormalization, Input, Dense
from keras.models import Model
from keras.initializers import RandomNormal, Constant
import math

class block(Model):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True, **kwargs):
        super(block, self).__init__(**kwargs)
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.reps = reps
        self.strides = strides
        self.start_with_relu = start_with_relu
        self.grow_first = grow_first

        if in_filters != out_filters or strides != 1:
            self.skip = [
                Conv2D(out_filters, 1, strides=strides, padding='valid', use_bias=False, 
                       kernel_initializer='he_normal'),
                BatchNormalization()
            ]
        else:
            self.skip = None

        layers = []
        filters = in_filters
        if grow_first:
            layers.append(ReLU())
            layers.append(SeparableConv2D(out_filters, 3, strides=1, padding='same',
                                          use_bias=False, depthwise_initializer='he_normal',
                                          pointwise_initializer='he_normal'))
            layers.append(BatchNormalization())
            filters = out_filters

        for _ in range(reps-1):
            layers.append(ReLU())
            layers.append(SeparableConv2D(filters, 3, strides=1, padding='same',
                                          use_bias=False, depthwise_initializer='he_normal',
                                          pointwise_initializer='he_normal'))
            layers.append(BatchNormalization())

        if not grow_first:
            layers.append(ReLU())
            layers.append(SeparableConv2D(out_filters, 3, strides=1, padding='same',
                                          use_bias=False, depthwise_initializer='he_normal',
                                          pointwise_initializer='he_normal'))
            layers.append(BatchNormalization())

        if not start_with_relu:
            layers = layers[1:]
        else:
            layers[0] = ReLU()

        if strides != 1:
            layers.append(MaxPooling2D(pool_size=3, strides=strides, padding='same'))
        
        self.main_branch = layers

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_filters': self.in_filters,
            'out_filters': self.out_filters,
            'reps': self.reps,
            'strides': self.strides,
            'start_with_relu': self.start_with_relu,
            'grow_first': self.grow_first
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.main_branch:
            x = layer(x, training=training)
        
        if self.skip is not None:
            skip = inputs
            for layer in self.skip:
                skip = layer(skip, training=training)
        else:
            skip = inputs

        x = Add()([x, skip])
        return x
    
class Xception(Model):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.num_classes = num_classes

        self.conv1 = Conv2D(32, 3, strides=2, padding='valid',
                            use_bias=False,
                            kernel_initializer='he_normal')
        self.bn1 = BatchNormalization()
        self.relu = ReLU()

        self.conv2 = Conv2D(64, 3, strides=1, padding='same',
                            use_bias=False,
                            kernel_initializer='he_normal')
        self.bn2 = BatchNormalization()

        self.block1 = block(64, 128, 2, strides=2, start_with_relu=False, grow_first=True)
        self.block2 = block(128, 256, 2, strides=2, start_with_relu=True, grow_first=True)
        self.block3 = block(256, 728, 2, strides=2, start_with_relu=True, grow_first=True)

        self.block4 = block(728, 728, 3, strides=1, start_with_relu=True, grow_first=True)
        self.block5 = block(728, 728, 3, strides=1, start_with_relu=True, grow_first=True)
        self.block6 = block(728, 728, 3, strides=1, start_with_relu=True, grow_first=True)
        self.block7 = block(728, 728, 3, strides=1, start_with_relu=True, grow_first=True)

        self.block8 = block(728, 728, 3, strides=1, start_with_relu=True, grow_first=True)
        self.block9 = block(728, 728, 3, strides=1, start_with_relu=True, grow_first=True)
        self.block10 = block(728, 728, 3, strides=1, start_with_relu=True, grow_first=True)
        self.block11 = block(728, 728, 3, strides=1, start_with_relu=True, grow_first=True)

        self.block12 = block(728, 1024, 2, strides=2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2D(1536, 3, strides=1, padding='same',
                            use_bias=False,
                            depthwise_initializer='he_normal',
                            pointwise_initializer='he_normal')
        self.bn3 = BatchNormalization()

        self.conv4 = SeparableConv2D(2048, 3, strides=1, padding='same',
                            use_bias=False,
                            depthwise_initializer='he_normal',
                            pointwise_initializer='he_normal')
        self.bn4 = BatchNormalization()

        self.global_pool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, kernel_initializer='he_normal')

        for layer in self.layers:
            if isinstance(layer, (Conv2D, SeparableConv2D)):
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.filters
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=math.sqrt(2. / n))
            elif isinstance(layer, BatchNormalization):
                layer.gamma_initializer = Constant(1.0)
                layer.beta_initializer = Constant(0.0)
            elif isinstance(layer, Dense):
                n = layer.units
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=math.sqrt(2. / n))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)

        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)

        x = self.block8(x, training=training)
        x = self.block9(x, training=training)
        x = self.block10(x, training=training)
        x = self.block11(x, training=training)

        x = self.block12(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.relu(x)

        x = self.global_pool(x)
        x = self.fc(x)

        return x
    
    def extract_features(self, inputs, training=False):
        features = self.conv1(inputs)
        features = self.bn1(features, training=training)
        features = self.relu(features)

        features = self.conv2(features)
        features = self.bn2(features, training=training)
        features = self.relu(features)

        features = self.block1(features, training=training)
        features = self.block2(features, training=training)
        features = self.block3(features, training=training)

        features = self.block4(features, training=training)
        features = self.block5(features, training=training)
        features = self.block6(features, training=training)
        features = self.block7(features, training=training)

        features = self.block8(features, training=training)
        features = self.block9(features, training=training)
        features = self.block10(features, training=training)
        features = self.block11(features, training=training)

        features = self.block12(features, training=training)

        features = self.conv3(features)
        features = self.bn3(features, training=training)
        features = self.relu(features)

        features = self.conv4(features)
        features = self.bn4(features, training=training)
        features = self.relu(features)

        features = self.global_pool(features)

        return features
        
    def xception(pretrained=False, **kwargs):
        model = Xception(**kwargs)
        if pretrained:
            pass
        return model
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

