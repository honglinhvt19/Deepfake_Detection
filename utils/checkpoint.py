import tensorflow as tf
import os
import re
from keras.saving import custom_object_scope
from keras.models import load_model

from models.feature_extractor import FeatureExtractor
from models.fusion import Fusion
from models.transformer import Transformer
from models.model import ModelBuilder

def create_checkpoint_callback(checkpoint_dir, monitor='roc_auc', mode='max'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{{epoch:02d}}-{{{monitor}:.4f}}.keras")
    print(f"Saving checkpoints to: {checkpoint_dir}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        save_weights_only=False,
        verbose=1
    )
    return checkpoint_callback

def load_checkpoint(model, checkpoint_path, monitor='roc_auc'):
    if not os.path.exists(checkpoint_path):
        print(f"Can't find checkpoint at {checkpoint_path}")
        checkpoint_callback = create_checkpoint_callback(checkpoint_path, monitor=monitor)
        return model, 0, float("inf"), checkpoint_callback
    
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.keras')]
    if not checkpoint_files:
        print(f"Can't find checkpoint in {checkpoint_path}")
        checkpoint_callback = create_checkpoint_callback(checkpoint_path, monitor=monitor)
        return model, 0, float("inf"), checkpoint_callback
    
    latest_checkpoint = None
    latest_epoch = -1
    best_val_auc = None

    for f in checkpoint_files:
        match = re.match(r"model_(\d+)-([\d.]+)\.keras", f)
        if match:
            epoch = int(match.group(1))
            val_auc = float(match.group(2))
            if epoch > latest_epoch:
                latest_epoch = epoch
                best_val_auc = val_auc
                latest_checkpoint = os.path.join(checkpoint_path, f)

    if latest_checkpoint:
        try:
            custom_objects = {
                "FeatureExtractor": FeatureExtractor,
                "Fusion": Fusion,
                "Transformer": Transformer,
                "ModelBuilder": ModelBuilder
            }

            with custom_object_scope(custom_objects):
                loaded_model = load_model(latest_checkpoint, custom_objects=custom_objects)
                print(f"Loaded model from {latest_checkpoint} at epoch {latest_epoch} (roc_auc={best_val_auc:.4f})")

                checkpoint_callback = create_checkpoint_callback(checkpoint_path, monitor=monitor)
                if best_val_auc is not None:
                    checkpoint_callback.best = best_val_auc
                    print(f"Restored checkpoint_callback.best = {best_val_auc:.4f}")
                return loaded_model, latest_epoch, best_val_auc, checkpoint_callback
        except Exception as e:
            print(f"Error: Can't load model {latest_checkpoint}: {e}")
            checkpoint_callback = create_checkpoint_callback(checkpoint_path, monitor=monitor)
            return model, 0, float("inf"), checkpoint_callback
    else:
        print(f"Can't find checkpoint in {checkpoint_path}")
        checkpoint_callback = create_checkpoint_callback(checkpoint_path, monitor=monitor)
        return model, 0, float("inf"), checkpoint_callback
