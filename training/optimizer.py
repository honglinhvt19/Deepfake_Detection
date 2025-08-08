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

def set_submodel_trainable(model: tf.keras.Model, target_class_name: str, trainable: bool):
    found = False
    # traverse model and nested layers
    for layer in model.layers:
        # If layer itself is a model-like object
        cls_name = layer.__class__.__name__
        if cls_name == target_class_name:
            layer.trainable = trainable
            # also set for its inner layers if present
            try:
                for sub in layer.layers:
                    sub.trainable = trainable
            except Exception:
                pass
            found = True
        # if layer has sublayers (functional or nested), recurse
        if hasattr(layer, "layers") and len(layer.layers) > 0:
            if set_submodel_trainable(layer, target_class_name, trainable):
                found = True
    return found