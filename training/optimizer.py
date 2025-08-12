import os
import tensorflow as tf

def get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        # fallback
        return tf.keras.optimizers.Adam(learning_rate=lr)

def set_submodel_trainable(model: tf.keras.Model, target_class: type, trainable: bool) -> bool:
    found = False
    
    for layer in getattr(model, 'layers', []):
        # 1. Nếu là instance của target_class
        if isinstance(layer, target_class):
            layer.trainable = trainable
            # set cho các sublayers nếu có
            for sub in getattr(layer, 'layers', []):
                sub.trainable = trainable
            found = True
        # 2. Nếu có attribute 'efficientnet' là instance của target_class
        if hasattr(layer, 'efficientnet'):
            eff = getattr(layer, 'efficientnet')
            if isinstance(eff, target_class):
                eff.trainable = trainable
                # set cho các sublayers nếu có
                for sub in getattr(eff, 'layers', []):
                    sub.trainable = trainable
                found = True
        # 3. Đệ quy vào các sublayers
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            if set_submodel_trainable(layer, target_class, trainable):
                found = True
    return found