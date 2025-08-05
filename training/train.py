import yaml
import os
from data.dataset import Dataset
from models.model import ModelBuilder
from utils.checkpoint import create_checkpoint_callback
from utils.logger import Logger
import tensorflow as tf

def train(config_path):
    # Đọc cấu hình
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
    
    # Biên dịch mô hình
    model.compile(
        optimizer=config['training']['optimizer'],
        loss=config['training']['loss'],
        metrics=config['training']['metrics']
    )
    
    # Callbacks
    logger = Logger(log_dir=config['training']['log_dir'])
    checkpoint_callback = create_checkpoint_callback(config['training']['checkpoint_dir'])
    
    # Huấn luyện
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['training']['epochs'],
        callbacks=[checkpoint_callback, logger],
        verbose=1
    )
    
    return history, model

if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    history, model = train(config_path)
