
import tensorflow as tf
import numpy as np
import os
from .preprocessing import preprocess_video

class Dataset:
    def __init__(self, data_dir, batch_size=32, num_frames=8, frame_size=(299, 299), training=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.training = training
        self.video_path, self.labels = self._load_data()

    def _load_data(self):
        video_paths = []
        labels = []

        for label, class_dir in enumerate(['real', 'fake']):
            class_path = os.path.join(self.data_dir, class_dir)
            if not os.path.exists(class_path):
                continue
            for video_name in os.listdir(class_path):
                video_path = os.path.join(class_path, video_name)

        return video_paths, labels

    def _generator(self):
        for video_path, label in zip(self.video_paths, self.labels):
            video_tensor = preprocess_video(video_path, self.num_frames, self.frame_size)
            yield video_tensor, label

    def as_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(self.num_frames, *self.frame_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
        dataset = tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
        if self.training:
            dataset = dataset.shuffle(100)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
