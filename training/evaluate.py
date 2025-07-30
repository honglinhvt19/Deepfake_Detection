import tensorflow as tf
import yaml
from data.dataset import Dataset
from models.model import ModelBuilder
from utils.metrics import compute_metrics

def evaluate(config_path, checkpoint_path):
    # Đọc cấu hình
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Khởi tạo dataset
    test_dataset = Dataset(
        data_dir=config['data']['data_dir'] + '/test',
        batch_size=config['data']['batch_size'],
        num_frames=config['data']['num_frames'],
        frame_size=tuple(config['data']['frame_size']),
        training=False
    )
    
    # Khởi tạo và tải mô hình
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
    model.load_weights(checkpoint_path)
    
    # Đánh giá
    results = model.evaluate(test_dataset, verbose=1)
    metrics = compute_metrics(model, test_dataset)
    
    print("Evaluation results:", dict(zip(model.metrics_names, results)))
    print("Additional metrics:", metrics)

if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    checkpoint_path = "./checkpoints/best_model.h5"
    evaluate(config_path, checkpoint_path)