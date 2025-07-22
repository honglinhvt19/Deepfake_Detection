import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from xception import Xception
from efficientnet import EfficientNet

class FeatureExtractor(Model):
    def __init__(self, num_classes=1000, pretrained=True):
        super(FeatureExtractor, self).__init__()
        
        self.num_classes = num_classes

        self.xception = Xception(num_classes=num_classes)
        self.efficientnet = EfficientNet(num_classes=num_classes)

    def call(self, inputs, training=False):
        xception_features = self.xception.extract_features(inputs, training=training)
        efficientnet_features = self.efficientnet.extract_features(inputs, training=training)

        return xception_features, efficientnet_features
    
