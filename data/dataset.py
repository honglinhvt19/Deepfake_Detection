import tensorflow as tf
import numpy as np
import os
import random
from .preprocessing import preprocess_video

class Dataset:
    def __init__(self, data_dir, batch_size=16, num_frames=8, frame_size=(224, 224), training=True, video_paths=None, labels=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.training = training
        self.video_paths, self.labels = self._load_data()

    def _load_data(self):
        video_paths = []
        labels = []

        for label, class_dir in enumerate(['real', 'fake']):
            class_path = os.path.join(self.data_dir, class_dir)
            if not os.path.exists(class_path):
                continue
            for video_name in os.listdir(class_path):
                video_path = os.path.join(class_path, video_name)
                video_paths.append(video_path)
                labels.append(label)

        n_real, n_fake = len(video_paths[0]), len(video_paths[1])
        if n_real > 0 and n_fake > 0:
            if n_real > n_fake:
                # oversample fake
                oversampled = random.choices(video_paths[1], k=n_real)
                video_paths[1] = oversampled
            elif n_fake > n_real:
                # oversample real
                oversampled = random.choices(video_paths[0], k=n_fake)
                video_paths[0] = oversampled

        # gộp lại
        all_paths = video_paths[0] + video_paths[1]
        all_labels = [0] * len(video_paths[0]) + [1] * len(video_paths[1])

        combined = list(zip(all_paths, all_labels))
        random.shuffle(combined)
        video_paths, labels = zip(*combined)

        return list(video_paths), list(labels)

    def _generator(self):
        dummy = np.zeros((self.num_frames, *self.frame_size, 3), dtype=np.float32)
        for video_path, label in zip(self.video_paths, self.labels):
            try:
                frames = preprocess_video(video_path, self.num_frames, self.frame_size, training=self.training)
                yield frames, tf.cast(label, tf.int32)
            except Exception as e:
                yield tf.convert_to_tensor(dummy), tf.cast(label, tf.int32)

    def as_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(self.num_frames, *self.frame_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
        dataset = tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
        if self.training:
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
