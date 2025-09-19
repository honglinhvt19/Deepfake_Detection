import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import yaml
import math
import tensorflow as tf
import keras
import gc
from keras import mixed_precision
from data.dataset import Dataset
from utils.checkpoint import create_checkpoint_callback, load_checkpoint
from utils.logger import Logger
from utils.metrics import get_metrics
from training.optimizer import get_optimizer
from models.model import ModelBuilder

mixed_precision.set_global_policy("mixed_float16")
print("✅ Mixed precision policy:", mixed_precision.global_policy())

class ClearMemory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print(f"[Epoch {epoch+1}] Cleared RAM/graph cache.")

def train(config_path, resume_from_checkpoint=False):
    # ----------------- Load config -----------------
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    freeze_epochs   = config['training']['freeze_epochs']
    fine_tune_lr    = config['training']['fine-tune_lr']
    base_lr         = config['training']['learning_rate']
    optimizer_name  = config['training']['optimizer']
    total_epochs    = config['training']['epochs']
    batch_size      = config['data']['batch_size']

    # ----------------- Dataset -----------------
    train_dataset = Dataset(
        data_dir=os.path.join(config['data']['data_dir'], "train"),
        batch_size=batch_size,
        num_frames=config['data']['num_frames'],
        frame_size=tuple(config['data']['frame_size']),
        training=True
    )
    val_dataset = Dataset(
        data_dir=os.path.join(config['data']['data_dir'], "val"),
        batch_size=batch_size,
        num_frames=config['data']['num_frames'],
        frame_size=tuple(config['data']['frame_size']),
        training=False
    )

    steps_per_epoch = math.ceil(len(train_dataset.video_paths) / batch_size)
    val_steps = math.ceil(len(val_dataset.video_paths) / batch_size)

    print(f"Train videos: {len(train_dataset.video_paths)}, Steps per epoch: {steps_per_epoch}")
    print(f"Val videos: {len(val_dataset.video_paths)}, Validation steps: {val_steps}")

    train_dataset = train_dataset.as_dataset()
    val_dataset = val_dataset.as_dataset()

    # ----------------- Model -----------------
    model_builder = ModelBuilder(
        num_classes=config['model']['num_classes'],
        num_frames=config['model']['num_frames'],
        embed_dims=config['model']['embed_dims'],
        num_heads=config['model']['num_heads'],
        ff_dim=config['model']['ff_dim'],
        num_transformer_layers=config['model']['num_transformer_layers'],
        dropout_rate=config['model']['dropout_rate'],
        use_spatial_attention=config['model']['use_spatial_attention'],
        freeze_ratio=1.0   # Phase 1: freeze toàn bộ backbone
    )
    model = model_builder.create_model()

    initial_epoch = 0
    if resume_from_checkpoint:
        model, initial_epoch, best_val_loss, checkpoint_callback = load_checkpoint(model, config['training']['checkpoint_dir'])
        print(f"Resume training from epoch {initial_epoch}")

    # ----------------- Phase 1: Train với backbone freeze -----------------
    history_phase1 = None
    epochs_phase1 = min(freeze_epochs, total_epochs)
    if initial_epoch < epochs_phase1:
        print(f"--- Phase 1: Train {epochs_phase1 - initial_epoch} epochs (backbone frozen) ---")

        decay_steps = (epochs_phase1-initial_epoch)*steps_per_epoch
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            base_lr,
            decay_steps,
            alpha=0.0
        )
        metrics = get_metrics(config['training']['metrics'])
        opt = get_optimizer(optimizer_name, lr_scheduler)             
        model.compile(optimizer=opt,
                      loss=config['training']['loss'],
                      metrics=metrics)

        logger = Logger(log_dir=config['training']['log_dir'])

        history_phase1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs_phase1,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=[checkpoint_callback, logger, ClearMemory()],
            verbose=1
        )
    else:
        print("Skip Phase 1: already trained more than freeze_epochs.")

    # ----------------- Phase 2: Fine-tune backbone -----------------
    history_phase2 = None
    if initial_epoch < total_epochs:
        print(f"--- Phase 2: Fine-tune backbone with LR={fine_tune_lr} ---")

        # Rebuild model với freeze_ratio < 1.0 (ví dụ 0.7 nghĩa là freeze 70%, fine-tune 30% cuối)
        model_builder_ft = ModelBuilder(
            num_classes=config['model']['num_classes'],
            num_frames=config['model']['num_frames'],
            embed_dims=config['model']['embed_dims'],
            num_heads=config['model']['num_heads'],
            ff_dim=config['model']['ff_dim'],
            num_transformer_layers=config['model']['num_transformer_layers'],
            dropout_rate=config['model']['dropout_rate'],
            use_spatial_attention=config['model']['use_spatial_attention'],
            freeze_ratio=config['training'].get('fine_tune_freeze_ratio', 0.8)
        )
        model = model_builder_ft.create_model()
        
        decay_steps = (total_epochs-initial_epoch)*steps_per_epoch
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            fine_tune_lr,
            decay_steps,
            alpha=0.0
        )
        metrics = get_metrics(config['training']['metrics'])
        opt_ft = get_optimizer(optimizer_name, lr_scheduler)
        model.compile(optimizer=opt_ft,
                      loss=config['training']['loss'],
                      metrics=metrics)

        logger2 = Logger(log_dir=config['training']['log_dir'])

        history_phase2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=total_epochs,
            initial_epoch=max(initial_epoch, epochs_phase1),
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=[checkpoint_callback, logger2, ClearMemory()],
            verbose=1
        )
    else:
        print("Skip Phase 2: training already completed.")

    return (history_phase1, history_phase2), model


if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    history, model = train(config_path, resume_from_checkpoint=True)
