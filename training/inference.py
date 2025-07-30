import tensorflow as tf
import yaml
from data.preprocessing import preprocess_video
from models.model import ModelBuilder

def inference(video_path, config_path, checkpoint_path):
    # Đọc cấu hình
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Tiền xử lý video
    frames = preprocess_video(
        video_path,
        num_frames=config['data']['num_frames'],
        frame_size=tuple(config['data']['frame_size']),
        training=False
    )
    if frames is None:
        print("Error processing video.")
        return None
    
    frames = tf.expand_dims(frames, axis=0)  # [1, num_frames, 299, 299, 3]
    
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
    
    # Dự đoán
    predictions = model.predict(frames)
    class_idx = tf.argmax(predictions, axis=1).numpy()[0]
    class_name = "Real" if class_idx == 0 else "Fake"
    confidence = predictions[0][class_idx]
    
    print(f"Prediction: {class_name} (Confidence: {confidence:.4f})")
    return class_name, confidence

if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    checkpoint_path = "./checkpoints/best_model.h5"
    video_path = "./sample_video.mp4"
    inference(video_path, config_path, checkpoint_path)