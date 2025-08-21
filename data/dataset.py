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
        real_paths, fake_paths = [], []

        for video_name in os.listdir(os.path.join(self.data_dir, "real")):
            real_paths.append(os.path.join(self.data_dir, "real", video_name))

        for video_name in os.listdir(os.path.join(self.data_dir, "fake")):
            fake_paths.append(os.path.join(self.data_dir, "fake", video_name))

        # oversampling
        n_real, n_fake = len(real_paths), len(fake_paths)
        if n_real > 0 and n_fake > 0:
            if n_real > n_fake:
                fake_paths = random.choices(fake_paths, k=n_real)   # oversample fake
            elif n_fake > n_real:
                real_paths = random.choices(real_paths, k=n_fake)   # oversample real

        # gộp lại
        all_paths = real_paths + fake_paths
        all_labels = [0] * len(real_paths) + [1] * len(fake_paths)

        # shuffle
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
