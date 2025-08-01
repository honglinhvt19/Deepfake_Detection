import tensorflow as tf
import os

def create_checkpoint_callback(checkpoint_dir, monitor='val_loss', mode='min'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model_{epoch:02d}_{val_loss:.4f}.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        save_weights_only=False,
        verbose=1
    )
    return checkpoint_callback

def load_checkpoint(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    return model