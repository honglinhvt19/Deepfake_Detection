import yaml
import tensorflow as tf
from models.model import ModelBuilder
from data.preprocessing import extract_frames
from utils.checkpoint import load_checkpoint
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

def inference_one_video(config_path, checkpoint_path, video_path):
    # ----------------- Load config -----------------
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

    num_frames = config['data']['num_frames']
    img_size = tuple(config['data']['frame_size'])

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

    # ----------------- Load checkpoint -----------------
    model, initial_epoch, best_auc, _ = load_checkpoint(
        model,
        checkpoint_path
    )
    print(f"Loaded checkpoint from epoch {initial_epoch}, best auc={best_auc:.4f}")

    # ----------------- Load & Predict -----------------
    video_tensor = extract_frames(video_path, num_frames=num_frames, target_size=img_size)
    video_tensor = tf.cast(video_tensor, tf.float32) / 255.0
    video_tensor = tf.expand_dims(video_tensor, axis=0)
    pred = model.predict(video_tensor, verbose=0).squeeze()

    print(f"Prediction for {video_path}: {pred:.4f}")
    return pred


if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    checkpoint_path = "./checkpoints"
    video_path = "id40_0003.mp4"

    pred = inference_one_video(config_path, checkpoint_path, video_path)
    label = "FAKE" if pred >= 0.5 else "REAL"
    print(f"Final result: {label}")

