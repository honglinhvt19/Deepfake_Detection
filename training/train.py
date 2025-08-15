import yaml
import os
import math
from data.dataset import Dataset
from utils.checkpoint import create_checkpoint_callback, load_checkpoint
from utils.logger import Logger
import tensorflow as tf
from training.optimizer import get_optimizer, set_submodel_trainable
from models.xception import Xception, block
from models.efficientnet import EfficientNet
from models.feature_extractor import FeatureExtractor
from models.fusion import Fusion
from models.transformer import Transformer
from models.model import ModelBuilder

def train(config_path, resume_from_checkpoint=False):
    # Đọc cấu hình
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    freeze_epochs = config['training']['freeze_epochs']
    fine_tune_lr = config['training']['fine-tune_lr']
    base_lr = config['training']['learning_rate']
    optimizer_name = config['training']['optimizer']

    total_epochs = config['training']['epochs']
    batch_size=config['data']['batch_size']
    
    # Khởi tạo dataset
    train_dataset = Dataset(
        data_dir=config['data']['data_dir'] + '/train',
        batch_size=config['data']['batch_size'],
        num_frames=config['data']['num_frames'],
        frame_size=tuple(config['data']['frame_size']),
        training=True
    )
    val_dataset = Dataset(
        data_dir=config['data']['data_dir'] + '/val',
        batch_size=config['data']['batch_size'],
        num_frames=config['data']['num_frames'],
        frame_size=tuple(config['data']['frame_size']),
        training=False
    )

    steps_per_epoch = math.ceil(len(train_dataset.video_paths)/batch_size)
    val_steps = math.ceil(len(val_dataset.video_paths)/batch_size)

    train_dataset = train_dataset.as_dataset()
    val_dataset = val_dataset.as_dataset()
    
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
    model = model_builder.create_model()

    print("Running GPU warmup...")
    for batch in train_dataset.take(1):
        model(batch[0], training=True)
    print("GPU warmup completed.")

    initial_epoch = 0
    if resume_from_checkpoint:
        model, initial_epoch = load_checkpoint(model, config['training']['checkpoint_dir'])
        print(f"Tiếp tục huấn luyện từ epoch {initial_epoch}")
    
    found = set_submodel_trainable(model, EfficientNet, False)
    if not found:
        print("Warning: Cannot find EffecientNet to Freeze.")
        return None
    else:
        print(f"EfficientNet has been frozen first {freeze_epochs} epochs.")

    opt = get_optimizer(optimizer_name, base_lr)
    model.compile(optimizer=opt, loss=config['training']['loss'], metrics=config['training']['metrics'])
    
    # Callbacks
    logger = Logger(log_dir=config['training']['log_dir'])
    checkpoint_callback = create_checkpoint_callback(config['training']['checkpoint_dir'])

    history_phase1 = None
    epochs_phase1 = min(freeze_epochs, total_epochs)
    if initial_epoch < epochs_phase1:
        found = set_submodel_trainable(model, EfficientNet, False)
        if not found:
            print("Error:Can't find EfficientNet to freeze.")
            return None
        else:
            print(f"EfficientNet has been frozen from epoch {initial_epoch + 1} to {epochs_phase1}.")
        
        opt = get_optimizer(optimizer_name, base_lr)
        model.compile(optimizer=opt, loss=config['training']['loss'], metrics=config['training']['metrics'])
        
        print(f"--- Phase 1: Train {epochs_phase1 - initial_epoch} epoch (Freezing EfficientNet) ---")
        history_phase1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs_phase1,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=[checkpoint_callback, logger],
            verbose=1
        )
    else:
        print("Pass phase 1: Trained more than freeze_epochs.")

    history_phase2 = None   
    if initial_epoch >= total_epochs:
        print("Training completed in Phase 1, no fine-tuning needed.")
        return history_phase1, model

    unfound = set_submodel_trainable(model, EfficientNet, True)
    if not unfound:
        print("Warning: Cannot find EfficientNet to unfreeze.")
    else:
        print("EfficientNet has been unfrozen for fine-tuning.")

    opt_ft = get_optimizer(optimizer_name, fine_tune_lr)
    model.compile(optimizer=opt_ft, loss=config['training']['loss'], metrics=config['training']['metrics'])
    
    remaining_epochs = total_epochs - max(epochs_phase1, initial_epoch)
    if remaining_epochs > 0:
        print(f"--- Phase 2: fine-tune {remaining_epochs} epoch (LR={fine_tune_lr}) ---")
        history_phase2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=total_epochs, 
            initial_epoch=max(initial_epoch, epochs_phase1),
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=[checkpoint_callback, logger],
            verbose=1
        )
    else:
        print("Pass phase 2: Not enough epochs to train.")
    
    return (history_phase1, history_phase2), model

if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    history, model = train(config_path, resume_from_checkpoint=True)
