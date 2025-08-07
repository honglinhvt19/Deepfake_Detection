import cv2
import numpy as np
import tensorflow as tf

def extracts_frames(video_path, num_frames = 8, frame_size = (299, 299)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
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

    return np.array(frames)[:num_frames] #[num_frames, 299, 299, 3]

def preprocess_frame(frame, normalize = True):
    frame = frame.astype(np.float32)
    if normalize:
        frame = frame / 255.0 #[0, 1]
    return frame

def augment_frame(frame):
    if np.random.rand() > 0.5:
        frame = tf.image.flip_left_right(frame)
    frame = tf.image.rot90(frame, k=np.random.randint(0, 2))
    frame = tf.image.random_brightness(frame, max_delta=0.1)
    frame  = tf.image.random_contrast(frame, lower=0.9, upper=1.1)
    return frame

def preprocess_video(video_path, num_frames=8, frame_size=(299, 299), training=False, normalize=True):
    frames = extracts_frames(video_path, num_frames, frame_size)
    if frames is None:
        return None

    processed_frames = []
    for frame in frames:
        frame = preprocess_frame(frame, normalize=normalize)
        if training:
            frame = augment_frame(frame)
        processed_frames.append(frame)
    
    frames = np.array(processed_frames, dtype=np.float32)  # [num_frames, 299, 299, 3]

    return frames
    