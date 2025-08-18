import tensorflow as tf
import os
import re

def create_checkpoint_callback(checkpoint_dir, monitor='val_loss', mode='min'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model_{epoch:02d}_{val_loss:.4f}.weights.h5")
    print(f"Saving checkpoints to: {checkpoint_dir}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        save_weights_only=True,
        verbose=1
    )
    return checkpoint_callback

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Can't find checkpoint at {checkpoint_path}")
        return model, 0
    
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.weights.h5')]
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
            model.load_weights(latest_checkpoint)
            print(f"Loaded weights from {latest_checkpoint} at epoch {latest_epoch}")
            return model, latest_epoch
        except Exception as e:
            print(f"Error: Can't load weights {latest_checkpoint}: {e}")
            return model, 0
    else:
        print(f"Can't find checkpoint in {checkpoint_path}")
        return model, 0
