import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# Khởi tạo detector một lần
detector = MTCNN()

def extracts_frames(video_path, num_frames=8, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Chuẩn hóa sang RGB
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- Crop mặt bằng MTCNN ----
        faces = detector.detect_faces(frame)
        if len(faces) > 0:
            best_face = max(faces, key=lambda d: d['confidence'])
            x, y, w, h = best_face['box']

            # clamp to frame boundaries
            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

            if x2 > x and y2 > y:
                face = frame[y:y2, x:x2]
            else:
                face = frame 
        else:
            face = frame

        face = cv2.resize(face, target_size)
        frames.append(face)

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    return np.array(frames, dtype=np.uint8)[:num_frames]

def frequency_augment(frame):
    frame_freq = tf.signal.fft2d(tf.cast(frame, tf.complex64))
    # Thêm noise nhẹ vào frequency
    noise = tf.complex(tf.random.normal(tf.shape(frame_freq), stddev=0.01), tf.zeros(tf.shape(frame_freq)))
    frame_freq += noise
    return tf.cast(tf.math.real(tf.signal.ifft2d(frame_freq)), tf.float32)

def augment_frame(frame):
    frame = tf.image.random_flip_left_right(frame)
    frame = tf.image.rot90(frame, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    frame = tf.image.random_brightness(frame, max_delta=0.2)  # Tăng delta
    frame = tf.image.random_contrast(frame, lower=0.8, upper=1.2)
    frame = tf.image.random_hue(frame, max_delta=0.1)  # Thêm hue
    frame = tf.image.random_saturation(frame, lower=0.8, upper=1.2)  # Thêm saturation
    if tf.random.uniform([]) > 0.5:
        frame = tf.clip_by_value(frame + tf.random.normal(tf.shape(frame), stddev=0.05), 0.0, 1.0)  # Gaussian noise
    if tf.random.uniform([]) > 0.5:
        frame = tf.image.random_jpeg_quality(frame, min_jpeg_quality=70, max_jpeg_quality=95)  # Compression
    if tf.random.uniform([]) > 0.5:
        frame = frequency_augment(frame)
    return frame


def preprocess_video(video_path, num_frames=8, training=False, normalize=True, frame_size=(224, 224)):
    frames = extracts_frames(video_path, num_frames, target_size=frame_size)

    frames = tf.convert_to_tensor(frames, dtype=tf.float32)
    if normalize:
        frames = frames / 255.0

    if training:
        frames = tf.map_fn(augment_frame, frames)

    return frames   # [num_frames, H, W, 3], chưa resize
