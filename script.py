import os
import io
import sys
import cv2
import yaml
import time
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf
import keras
from keras import mixed_precision
from typing import Optional, Dict

from models.model import ModelBuilder
from models.feature_extractor import FeatureExtractor
from models.fusion import Fusion
from models.transformer import Transformer

mixed_precision.set_global_policy("mixed_float16")

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(page_title="Deepfake Detection – Inference", page_icon="🕵️‍♂️", layout="wide")

# ------------------------------
# Global defaults (edit to match your training)
# ------------------------------
DEFAULTS = {
    "img_size": (224, 224),
    "num_frames": 16,
}

# ------------------------------
# Helpers
# ------------------------------
@st.cache_resource(show_spinner=False)
def get_face_detector():
    """Init insightface detector once. Fallback to None if missing."""
    try:
        import insightface  # lazy import so app still runs without it
        app = insightface.app.FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        # ctx_id=0 (GPU) if available
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        st.warning(f"Face detector not available (insightface): {e}")
        return None


def sample_indices(total, k):
    if total <= 0:
        return np.zeros(k, dtype=int)
    return np.linspace(0, max(0, total - 1), k, dtype=int)


def extract_frames(path, num_frames=8, target_size=(224, 224), use_face_crop=True, detector=None):
    """Read k frames evenly from a video file and optionally crop largest face.
    Returns np.ndarray of shape (num_frames, H, W, 3), dtype=uint8 (RGB).
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0 or not cap.isOpened():
        cap.release()
        return np.zeros((num_frames, *target_size, 3), dtype=np.uint8)

    idxs = sample_indices(total, num_frames)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if use_face_crop and detector is not None:
            try:
                faces = detector.get(frame)
            except Exception:
                faces = []
            if len(faces) > 0:
                best = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                x1, y1, x2, y2 = best.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                face = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame
            else:
                face = frame
        else:
            face = frame

        face = cv2.resize(face, target_size, interpolation=cv2.INTER_LINEAR)
        frames.append(face)

    cap.release()

    # pad if not enough
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))

    return np.asarray(frames, dtype=np.uint8)


def preprocess(frames):
    """(T,H,W,3 uint8) -> (1,T,H,W,3 float32 in [0,1])"""
    x = frames.astype("float32") / 255.0
    return np.expand_dims(x, 0)


def save_uploaded_file(uploaded: st.runtime.uploaded_file_manager.UploadedFile, suffix: str):
    """Save an uploaded file to a NamedTemporaryFile and return its path."""
    if uploaded is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name


# Try to build Model from config if user supplies their own ModelBuilder
# If you don't use a custom ModelBuilder, replace this with your own keras model def

def build_model_from_config(cfg: dict):
    """Example builder using a custom ModelBuilder (edit for your codebase)."""
    # ---- BEGIN: edit to your project ----
    # Expecting your own ModelBuilder to be available. If not, raise.
    try:
        from models.model import ModelBuilder  # noqa
    except Exception:
        raise RuntimeError(
            "ModelBuilder import failed. Either provide a full .keras model or ensure models.model is importable."
        )

    model_builder = ModelBuilder(
        num_classes=cfg.get("num_classes", 1),
        num_frames=cfg.get("num_frames", DEFAULTS["num_frames"]),
        embed_dims=cfg.get("embed_dims", 256),
        num_heads=cfg.get("num_heads", 4),
        ff_dim=cfg.get("ff_dim", 512),
        num_transformer_layers=cfg.get("num_transformer_layers", 2),
        dropout_rate=cfg.get("dropout_rate", 0.1),
        use_spatial_attention=cfg.get("use_spatial_attention", True),
        freeze_ratio=cfg.get("freeze_ratio", 1.0),
    )
    model = model_builder.create_model()
    return model
    # ---- END: edit to your project ----


def try_load_full_model(path: str):
    """Try loading a full saved model (.keras). Returns model or raises."""
    # If you used custom classes/layers at train-time, import & pass via custom_objects
    custom_objects = {}
    try:
        from models.model import ModelBuilder  # noqa: F401
        from models.feature_extractor import FeatureExtractor
        from models.fusion import Fusion
        from models.transformer import Transformer
        custom_objects.update({
            "FeatureExtractor": FeatureExtractor,
            "Fusion": Fusion,
            "Transformer": Transformer,
            "ModelBuilder": ModelBuilder,
        })
    except Exception:
        pass
    return keras.models.load_model(path, custom_objects=custom_objects, compile=False)


def load_model_any(checkpoint_file: str, config_dict: Optional[Dict] = None):
    """Load a model from a .keras file.
    - First try as a full SavedModel (.keras)
    - If that fails and a config is given, build model and load weights
    """
    try:
        m = try_load_full_model(checkpoint_file)
        return m, "loaded_full_model"
    except Exception as e_full:
        if not config_dict:
            raise RuntimeError(
                f"Failed to load full model: {e_full}. Provide a config.yaml and weights file."
            )
        # Fallback: build and load weights
        m = build_model_from_config(config_dict.get("model", {}))
        m.load_weights(checkpoint_file)
        return m, "loaded_weights"


# ------------------------------
# Sidebar – model and options
# ------------------------------
with st.sidebar:
    st.title("🔧 Options")

    st.subheader("Model Checkpoint")
    ckpt_upload = st.file_uploader("Upload checkpoint (.keras)", type=["keras", "h5"], accept_multiple_files=False)
    ckpt_path_text = st.text_input("…or path to checkpoint on disk", value="", help="Absolute or relative path on your machine")

    st.subheader("Config (optional if checkpoint is a full model)")
    cfg_upload = st.file_uploader("Upload config.yaml", type=["yaml", "yml"], accept_multiple_files=False)

    st.divider()
    st.subheader("Preprocessing")
    use_face_crop = st.checkbox("Crop face before inference (insightface)", value=True)
    num_frames = st.slider("Frames to sample", min_value=4, max_value=64, value=DEFAULTS["num_frames"], step=4)
    img_w = st.number_input("Width", min_value=64, max_value=1024, value=DEFAULTS["img_size"][0])
    img_h = st.number_input("Height", min_value=64, max_value=1024, value=DEFAULTS["img_size"][1])

    st.divider()
    st.caption("Tip: If you trained with face crops, keep this enabled for best results.")

# ------------------------------
# Main – header
# ------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px;">
      <h1 style="margin:0;">🕵️‍♂️ Deepfake Detection – Inference</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Upload a video and a checkpoint to classify it as **REAL** or **FAKE**.")

# ------------------------------
# Config load (if provided)
# ------------------------------
config_dict = None
if cfg_upload is not None:
    try:
        config_text = cfg_upload.read().decode("utf-8")
        config_dict = yaml.safe_load(config_text)
    except Exception as e:
        st.error(f"Failed to parse config.yaml: {e}")

# Override UI settings with config if present
if config_dict and "data" in config_dict:
    num_frames = int(config_dict["data"].get("num_frames", num_frames))
    if "img_size" in config_dict["data"] and isinstance(config_dict["data"]["img_size"], (list, tuple)):
        img_w, img_h = config_dict["data"]["img_size"]

# ------------------------------
# Load / Build model
# ------------------------------
model = None
model_status = st.empty()

ckpt_file_path = None
if ckpt_upload is not None:
    ckpt_file_path = save_uploaded_file(ckpt_upload, suffix=os.path.splitext(ckpt_upload.name)[1] or ".keras")
elif ckpt_path_text.strip():
    ckpt_file_path = ckpt_path_text.strip()

if ckpt_file_path:
    with st.spinner("Loading model…"):
        try:
            model, load_mode = load_model_any(ckpt_file_path, config_dict)
            model_status.success(f"Model ready ({load_mode}).")
        except Exception as e:
            model_status.error(f"Failed to load model: {e}")
else:
    model_status.info("Upload a checkpoint file or input a path in the sidebar.")

# ------------------------------
# Face detector (optional)
# ------------------------------
detector = get_face_detector() if use_face_crop else None
if use_face_crop and detector is None:
    st.info("Proceeding without face crop.")

# ------------------------------
# Video upload & inference
# ------------------------------
st.subheader("🎬 Video")
video_file = st.file_uploader("Upload a video (mp4/mov/avi)", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)

colA, colB = st.columns([1, 1])

with colA:
    if video_file is not None:
        st.video(video_file)

run_btn = st.button("🔎 Run Inference", type="primary", use_container_width=True, disabled=(model is None or video_file is None))

with colB:
    if run_btn and model is not None and video_file is not None:
        # Save video to temp then read with OpenCV
        video_path = save_uploaded_file(video_file, suffix=os.path.splitext(video_file.name)[1])
        if not video_path:
            st.error("Could not save uploaded video.")
        else:
            with st.spinner("Extracting frames and running the model…"):
                try:
                    frames = extract_frames(
                        video_path,
                        num_frames=num_frames,
                        target_size=(int(img_w), int(img_h)),
                        use_face_crop=use_face_crop,
                        detector=detector,
                    )
                    x = preprocess(frames)
                    t0 = time.time()
                    pred = model.predict(x, verbose=0)
                    infer_ms = (time.time() - t0) * 1000.0

                    # Assume sigmoid output; adjust if your model differs
                    score = float(np.squeeze(pred))
                    label = "FAKE" if score >= 0.5 else "REAL"
                    conf = score if label == "FAKE" else (1.0 - score)

                    # Results card
                    st.markdown("### Result")
                    st.metric(label="Prediction", value=label, delta=f"confidence {conf*100:.2f}%")
                    st.progress(min(1.0, max(0.0, conf)))
                    st.caption(f"Inference time: {infer_ms:.1f} ms for {num_frames} frames @ {img_w}x{img_h}.")

                    # Show a contact sheet of sampled frames
                    try:
                        import math
                        grid_cols = min(8, num_frames)
                        grid_rows = int(math.ceil(num_frames / grid_cols))
                        h, w = int(img_h), int(img_w)
                        canvas = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.uint8)
                        for i, fr in enumerate(frames):
                            r = i // grid_cols
                            c = i % grid_cols
                            canvas[r*h:(r+1)*h, c*w:(c+1)*w, :] = fr
                        st.image(canvas, caption="Sampled frames (after crop/resize)")
                    except Exception:
                        pass

                except Exception as e:
                    st.error(f"Inference failed: {e}")

# ------------------------------
# Footer
# ------------------------------
st.divider()
st.caption("Note: If your checkpoint was saved as \"weights only\", please provide config.yaml so the app can rebuild the model and load weights. If you saved a full \".keras\" model, config is optional.")
