import tensorflow as tf
import os
import re
from keras.models import load_model

def create_checkpoint_callback(checkpoint_dir, monitor='val_loss', mode='min'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model_{epoch:02d}_{val_loss:.4f}.h5")
    print(f"Saving checkpoints to: {checkpoint_dir}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor=monitor,
        save_best_only=False,
        mode=mode,
        save_weights_only=False,
        verbose=1
    )
    return checkpoint_callback

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Can't find checkpoint at {checkpoint_path}")
        return model, 0
    
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.h5')]
    if not checkpoint_files:
        print(f"Can't find checkpoint in {checkpoint_path}")
        return model, 0
    
    latest_checkpoint = None
    latest_epoch = -1
    for f in checkpoint_files:
        match = re.match(r"model_(\d+)_[\d.]+.weights.h5", f)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = os.path.join(checkpoint_path, f)

    if latest_checkpoint:
        try:
            custom_objects = {}
            for obj in getattr(model, "submodules", []):
                if isinstance(obj, (tf.keras.layers.Layer, tf.keras.Model)):
                    cls = obj.__class__
                    mod = getattr(cls, "__module__", "")
                    if not (mod.startswith("keras") or mod.startswith("tensorflow") or mod.startswith("tf_keras")):
                        custom_objects[cls.__name__] = cls

            loaded_model = load_model(latest_checkpoint, custom_objects=custom_objects)
            print(f"Loaded model from {latest_checkpoint} at epoch {latest_epoch}")
            return loaded_model, latest_epoch
        except Exception as e:
            print(f"Error: Can't load model {latest_checkpoint}: {e}")
            return model, 0
    else:
        print(f"Can't find checkpoint in {checkpoint_path}")
        return model, 0
