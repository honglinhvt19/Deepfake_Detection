import tensorflow as tf
import os
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator

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
        self.writer.flush()
        
        # In log ra console
        print(f"Epoch {epoch + 1}: " + ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()]))

    def on_train_begin(self, logs=None):
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Logs will be saved to {self.log_dir}")

    def on_train_end(self, logs=None):
        self.writer.close()

class HistoryWrapper:
    def __init__(self, history_dict):
        self.history = history_dict

def load_tb_history(log_dir):
    if os.path.isdir(log_dir):
        event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
        if not event_files:
            raise FileNotFoundError(f"No TensorBoard log files found in {log_dir}")
        event_file = os.path.join(log_dir, event_files[0])  # lấy file đầu tiên
    else:
        event_file = log_dir 
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    print("Available tags:", ea.Tags()['scalars'])

    history_dict = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        history_dict[tag] = [e.value for e in events]

    return HistoryWrapper(history_dict)
