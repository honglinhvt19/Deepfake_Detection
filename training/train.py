import yaml
import os
from data.dataset import Dataset
from models.model import ModelBuilder
from utils.checkpoint import create_checkpoint_callback
from utils.logger import Logger
import tensorflow as tf
from optimizer import get_optimizer, set_submodel_trainable

def train(config_path):
    # Đọc cấu hình
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    freeze_epochs = config['training']['freeze_epochs']
    fine_tune_lr = config['training']['fine_tune_lr']
    base_lr = config['training']['learning_rate']
    
    # Khởi tạo dataset
    train_dataset = Dataset(
        data_dir=config['data']['data_dir'] + '/train',
        batch_size=config['data']['batch_size'],
        num_frames=config['data']['num_frames'],
        frame_size=tuple(config['data']['frame_size']),
        training=True
    ).as_dataset()
    val_dataset = Dataset(
        data_dir=config['data']['data_dir'] + '/val',
        batch_size=config['data']['batch_size'],
        num_frames=config['data']['num_frames'],
        frame_size=tuple(config['data']['frame_size']),
        training=False
    ).as_dataset()
    
    # Khởi tạo mô hình
    model_builder = ModelBuilder(
        num_classes=config['model']['num_classes'],
        num_frames=config['model']['num_frames'],
        embed_dims=config['model']['embed_dims'],
        num_heads=config['model']['num_heads'],
        ff_dim=config['model']['ff_dim'],
        num_transformer_layers=config['model']['num_transformer_layers'],
        dropout_rate=config['model']['dropout_rate'],
        use_spatial_attention=config['model']['use_spatial_attention']
    )
    model = model_builder.build()
    
    found = set_submodel_trainable(model, 'EffecientNet', False)
    if not found:
        print("Warning: Cannot find EffecientNet to Freeze.")
    else:
        print(f"EfficientNet has been frozen first {freeze_epochs} epochs.")

    opt = get_optimizer(optimizer_name, base_lr)
    model.compile(optimizer=opt, loss=config['training']['loss'], metrics=config['training']['metrics'])
    
    # Callbacks
    logger = Logger(log_dir=config['training']['log_dir'])
    checkpoint_callback = create_checkpoint_callback(config['training']['checkpoint_dir'])

    history_phase1 = None
    epochs_phase1 = min(freeze_epochs, total_epochs)
    if epochs_phase1 > 0:
        print(f"--- Phase 1: train {epochs_phase1} epoch (EfficientNet frozen) ---")
        history_phase1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs_phase1,
            callbacks=[checkpoint_callback, logger],
            verbose=1
        )

    if freeze_epochs >= total_epochs:
        print("Training completed in Phase 1, no fine-tuning needed.")
    return history_phase1, model

    unfound = set_submodel_trainable(model, "EfficientNet", True)
    if not unfound:
        print("Warning: Cannot find EfficientNet to unfreeze.")
    else:
        print("EfficientNet has been unfrozen for fine-tuning.")

    opt_ft = get_optimizer(optimizer_name, fine_tune_lr)
    model.compile(optimizer=opt_ft, loss=config['training']['loss'], metrics=config['training']['metrics'])
    
    remaining_epochs = total_epochs - epochs_phase1
    print(f"--- Phase 2: fine-tune {remaining_epochs} epoch (LR={fine_tune_lr}) ---")
    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=total_epochs,  # Keras expects final epoch number; will continue from epochs_phase1 to total_epochs
        initial_epoch=epochs_phase1,
        callbacks=[checkpoint_callback, logger],
        verbose=1
    )
    
    return (history_phase1, history_phase2), model

if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    history, model = train(config_path)
