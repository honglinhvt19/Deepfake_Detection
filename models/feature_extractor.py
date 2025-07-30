import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from .xception import Xception
from .efficientnet import EfficientNet

class FeatureExtractor(Model):
    def __init__(self, num_classes=1000, pretrained=True):
        super(FeatureExtractor, self).__init__()
        
        self.num_classes = num_classes

        self.xception = Xception(num_classes=num_classes)
        self.efficientnet = EfficientNet(num_classes=num_classes)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        num_frames = tf.shape(inputs)[1]
        
        input_flat = tf.reshape(inputs, [-1, 299, 299, 3]) # [batch_size * num_frames, 299, 299, 3]
        
        xception_features = self.xception.extract_features(input_flat, training=training)
        efficientnet_features = self.efficientnet.extract_features(input_flat, training=training)

        xception_features = tf.reshape(xception_features, [batch_size, num_frames, 2048]) # [batch_size, num_frames, 2048]
        efficientnet_features = tf.reshape(efficientnet_features, [batch_size, num_frames, 1280])

        return xception_features, efficientnet_features
    
