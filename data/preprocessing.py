import cv2
import numpy as np
import tensorflow as tf

def extracts_frames(video_path, num_frames=8, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, *frame_size, 3), dtype=np.uint8)
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Đảm bảo frame luôn là 3 kênh RGB
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))

    return np.array(frames, dtype=np.uint8)[:num_frames]  # numpy, uint8

def augment_frame(frame):
    if tf.random.uniform([]) > 0.5:
        frame = tf.image.flip_left_right(frame)
    frame = tf.image.rot90(frame, k=tf.random.uniform([], 0, 2, dtype=tf.int32))
    frame = tf.image.random_brightness(frame, max_delta=0.1)
    frame = tf.image.random_contrast(frame, lower=0.9, upper=1.1)
    return frame


def preprocess_video(video_path, num_frames=8, frame_size=(224, 224), training=False, normalize=True):
    frames = extracts_frames(video_path, num_frames, frame_size)  # numpy uint8
    if frames is None:
        frames = np.zeros((num_frames, *frame_size, 3), dtype=np.uint8)

    frames = tf.convert_to_tensor(frames, dtype=tf.float32)
    if normalize:
        frames = frames / 255.0

    if training:
        frames = tf.map_fn(augment_frame, frames)

    return frames  # [num_frames, H, W, 3] Tensor
