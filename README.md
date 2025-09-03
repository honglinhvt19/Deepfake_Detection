# 🕵️‍♂️ Deepfake Detection <img src="https://img.shields.io/badge/python-3.8%2B-blue?logo=python" height="20"> <img src="https://img.shields.io/badge/tensorflow-2.18.0-orange?logo=tensorflow" height="20"> <img src="https://img.shields.io/badge/streamlit-app-red?logo=streamlit" height="20">

<p align="center">
    <img src="https://img.icons8.com/color/96/artificial-intelligence.png" height="80"/>
    <img src="https://img.icons8.com/color/96/video.png" height="80"/>
    <img src="https://img.icons8.com/color/96/face-id.png" height="80"/>
</p>

## 🚀 Overview

This repository provides a robust deepfake detection pipeline using state-of-the-art deep learning techniques. The system is designed to classify videos as <b>REAL</b> or <b>FAKE</b> by leveraging advanced feature extraction, fusion, and transformer-based temporal modeling.

## ✨ Features

- 🧠 <b>Custom Model Architecture</b>: Combines Xception and EfficientNetB0 for feature extraction, a fusion module, and transformer layers for temporal analysis.
- ⚡ <b>Mixed Precision Training</b>: Utilizes TensorFlow mixed precision for faster training and lower memory usage.
- 🎞️ <b>Flexible Data Pipeline</b>: Supports face-cropped frame extraction and smart upsampling for imbalanced datasets.
- 🛠️ <b>Comprehensive Utilities</b>: Includes logging, checkpointing, metric computation, and visualization tools.
- 🖥️ <b>Streamlit Inference App</b>: User-friendly web interface for model inference on uploaded videos.

## 📁 Project Structure

```text
configs/         # YAML configuration files
data/            # Data loading, preprocessing, and frame extraction
models/          # Model components: feature extractor, fusion, transformer, builder
training/        # Training, evaluation, inference, optimizer logic
utils/           # Checkpointing, logging, metrics, visualization
checkpoints/     # Saved model checkpoints
logs/            # TensorBoard logs
main.py          # CLI entry point for training/evaluation/inference
script.py        # Streamlit inference app
CelebDF_split.py # Utility for dataset organization
requirements.txt # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**
     ```bash
     git clone https://github.com/honglinhvt19/Deepfake_Detection.git
     cd Deepfake_Detection
     ```

2. **Install dependencies**
     ```bash
     pip install -r requirements.txt
     ```
     > **Note:** Ensure you have a compatible GPU and CUDA/cuDNN installed for best performance.

## 🏃 Usage

### 1. Training

```bash
python main.py --mode train --config configs/config.yaml
```

### 2. Evaluation

```bash
python main.py --mode evaluate --config configs/config.yaml --checkpoint checkpoints/model_xx-xxxx.keras
```

### 3. Inference (CLI)

```bash
python main.py --mode inference --config configs/config.yaml --checkpoint checkpoints/model_xx-xxxx.keras --video path/to/video.mp4
```

### 4. Inference (Streamlit App)

```bash
streamlit run script.py
```
- Upload a video and a model checkpoint (`.keras`).
- Optionally upload a config file if using weights-only checkpoints.
- Adjust preprocessing options (face crop, frame count, image size) in the sidebar.

## ⚙️ Configuration

Edit `configs/config.yaml` to adjust model, data, and training parameters. Example:
```yaml
model:
    num_classes: 1
    num_frames: 8
    embed_dims: 256
    num_heads: 8
    ff_dim: 2048
    num_transformer_layers: 6
    dropout_rate: 0.4
    use_spatial_attention: true

data:
    data_dir: "/path/to/dataset"
    batch_size: 16
    num_frames: 8
    frame_size: [224, 224]
    normalize: true

training:
    epochs: 80
    learning_rate: 0.001
    fine-tune_lr: 0.00001
    freeze_epochs: 10
    optimizer: "adam"
    loss: "binary_crossentropy"
    metrics: ["accuracy", "precision", "recall", "roc_auc", "pr_auc"]
    checkpoint_dir: "./checkpoints"
    log_dir: "./logs"
```

## 📦 Dataset Preparation

- Organize your dataset into `real` and `fake` subfolders under `train`, `val`, and `test`.
- Use `CelebDF_split.py` to split and organize the Celeb-DF dataset if needed.

## 🧰 Utilities

- <b>Logging</b>: TensorBoard logs are saved in the `logs/` directory.
- <b>Checkpoints</b>: Best models are saved in `checkpoints/` with epoch and validation AUC in the filename.
- <b>Visualization</b>: Training curves, confusion matrices, and ROC/PR curves are available in `utils/visualization.py`.

## 📄 Citation

If you use this codebase, please cite the repository.

## 📝 License

This project is licensed under the MIT License.
