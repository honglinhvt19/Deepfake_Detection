import tensorflow as tf
import os
from datetime import datetime

class Logger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(Logger, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))

    def on_epoch_end(self, epoch, logs=None):
        # Ghi log vào TensorBoard
        with self.writer.as_default():
            for metric_name, metric_value in logs.items():
                tf.summary.scalar(metric_name, metric_value, step=epoch)
        
        # In log ra console
        print(f"Epoch {epoch + 1}: " + ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()]))

    def on_train_begin(self, logs=None):
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Logs will be saved to {self.log_dir}")

    def on_train_end(self, logs=None):
        self.writer.close()
