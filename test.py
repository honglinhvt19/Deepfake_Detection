import tensorflow as tf
from keras.models import load_model
from models.feature_extractor import FeatureExtractor
from models.fusion import Fusion
from models.transformer import Transformer
from models.model import ModelBuilder

custom_objects = {
    "FeatureExtractor": FeatureExtractor,
    "Fusion": Fusion,
    "Transformer": Transformer,
    "ModelBuilder": ModelBuilder
}
# load model
model = load_model("checkpoints\model_23-0.9995.keras", custom_objects=custom_objects)   # hoặc model.keras

# In kiến trúc
model.summary()

# Xuất chi tiết hơn
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file="model.png")
