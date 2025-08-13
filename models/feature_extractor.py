import tensorflow as tf
from keras.layers import Input, Layer
from .xception import Xception
from .efficientnet import EfficientNet

class FeatureExtractor(Layer):
    def __init__(self, num_classes=1000, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.xception = None
        self.efficientnet = None

    def build(self, input_shape):
        self.xception = Xception(num_classes=self.num_classes)
        self.efficientnet = EfficientNet(num_classes=self.num_classes, pretrained=self.pretrained)
        super(FeatureExtractor, self).build(input_shape)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        num_frames = tf.shape(inputs)[1]
        
        input_flat = tf.reshape(inputs, [-1, 224, 224, 3]) # [batch_size * num_frames, 224, 224, 3]
        
        xception_features = self.xception.extract_features(input_flat, training=training)
        efficientnet_features = self.efficientnet.extract_features(input_flat, training=training)

        xception_features = tf.reshape(xception_features, [batch_size, num_frames, 2048]) # [batch_size, num_frames, 2048]
        efficientnet_features = tf.reshape(efficientnet_features, [batch_size, num_frames, 1280])

        return xception_features, efficientnet_features
    
    
    