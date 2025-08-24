import tensorflow as tf
import os
import random
import numpy as np
from .preprocessing import extract_frames, IMAGE_SIZE

class Dataset:
    def __init__(self, data_dir, batch_size=16, num_frames=8, training=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.training = training
        self.video_paths, self.labels = self._load_data()

    def _load_data(self):
        video_paths, labels = [], []
        for label, cls in enumerate(["real", "fake"]):
            class_dir = os.path.join(self.data_dir, cls)
            if not os.path.exists(class_dir):
                continue
            for f in os.listdir(class_dir):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_paths.append(os.path.join(class_dir, f))
                    labels.append(label)
        # shuffle toàn bộ dataset
        combined = list(zip(video_paths, labels))
        random.shuffle(combined)
        video_paths, labels = zip(*combined)
        return list(video_paths), list(labels)

    def _process_video(self, video_path, label):
        frames = extract_frames(video_path.decode("utf-8"), self.num_frames, IMAGE_SIZE)
        frames = frames.astype(np.float32) / 255.0
        return frames, np.int32(label)

    def _tf_wrapper(self, video_path, label):
        frames, label = tf.numpy_function(
            self._process_video, [video_path, label], [tf.float32, tf.int32]
        )
        frames.set_shape((self.num_frames, *IMAGE_SIZE, 3))
        label.set_shape(())
        return frames, label

    def _augment_frame(self, frame):
        frame = tf.image.random_flip_left_right(frame)
        frame = tf.image.random_brightness(frame, max_delta=0.1)
        frame = tf.image.random_contrast(frame, 0.9, 1.1)
        frame = tf.image.random_saturation(frame, 0.9, 1.1)
        return frame

    def _augment_video(self, frames, label):
        frames = tf.map_fn(self._augment_frame, frames)
        return frames, label

    # -------------------- Video-level dataset --------------------
    def as_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.video_paths, self.labels))
        if self.training:
            dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
        dataset = dataset.map(self._tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        if self.training:
            dataset = dataset.map(self._augment_video, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    # -------------------- Image-level dataset --------------------
    def as_image_dataset(self):
        # load video-level dataset
        ds_video = tf.data.Dataset.from_tensor_slices((self.video_paths, self.labels))
        if self.training:
            ds_video = ds_video.shuffle(1000, reshuffle_each_iteration=True)

        # decode video -> (num_frames,224,224,3), label
        ds_video = ds_video.map(self._tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

        # augment nếu training
        if self.training:
            ds_video = ds_video.map(self._augment_video, num_parallel_calls=tf.data.AUTOTUNE)

        # --------- chuyển video-level -> frame-level ---------
        # (num_frames,224,224,3), ()  -->  (224,224,3), ()
        ds_frames = ds_video.unbatch()

        # batch lại chuẩn cho Xception/EfficientNet
        ds_frames = ds_frames.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds_frames

