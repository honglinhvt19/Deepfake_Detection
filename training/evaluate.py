import os
import yaml
import numpy as np
import tensorflow as tf
from data.dataset import Dataset
from utils.checkpoint import load_checkpoint
from utils.metrics import get_metrics
from models.model import ModelBuilder
from utils.visualization import plot_eval_curves, plot_confusion_matrix, print_classification_report
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
def test(config_path):
    # ----------------- Load config -----------------
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['data']['batch_size']
    num_frames = config['data']['num_frames']

    # ----------------- Dataset -----------------
    test_dataset = Dataset(
        data_dir=os.path.join(config['data']['data_dir']),
        batch_size=batch_size,
        num_frames=num_frames,
        training=False
    )
    test_dataset = test_dataset.load_test_dataset()
    
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
        freeze_ratio=1.0
    )
    model = model_builder.create_model()

    # Load checkpoint
    model, initial_epoch, best_auc, checkpoint_callback = load_checkpoint(
        model,
        "model_15-0.9955.keras"
    )
    print(f"Loaded checkpoint from epoch {initial_epoch}, best auc={best_auc:.4f}")

    # ----------------- Evaluate -----------------
    metrics = get_metrics(config['training']['metrics'])
    model.compile(optimizer="adam",   # chỉ cần compile để gắn metric
                  loss=config['training']['loss'],
                  metrics=metrics)

    results = model.evaluate(test_dataset, verbose=1)
    print("=== Test results ===")
    for name, val in zip(model.metrics_names, results):
        print(f"{name}: {val:.4f}")

    y_true, y_pred = [], []
    for x, y in test_dataset:
        p = model.predict(x, verbose=0)
        y_true.extend(y.numpy())
        y_pred.extend(p.squeeze())

    y_pred = np.array(y_pred)
    y_pred_labels = (y_pred >= 0.5).astype(int)

    # Vẽ plot
    plot_eval_curves(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred_labels)

    # In classification report
    print_classification_report(y_true, y_pred_labels)

if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    test(config_path)
