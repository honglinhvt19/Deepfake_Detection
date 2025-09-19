import os
import random
import tensorflow as tf
import numpy as np
from data.preprocessing import extract_frames, IMAGE_SIZE

class Dataset:
    def __init__(self, root_dir, split="train", batch_size=16, num_frames=8, shuffle=True, augment=False):

        self.root_dir = root_dir
        self.split = split
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.shuffle = shuffle
        self.augment = augment

        self.video_paths, self.labels = self._load_video_list()

    def _load_video_list(self):
        split_dir = os.path.join(self.root_dir, self.split)
        real_dir = os.path.join(split_dir, "real")
        fake_dir = os.path.join(split_dir, "fake")

        real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
        fake_videos = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]

        video_paths = real_videos + fake_videos
        labels = [0] * len(real_videos) + [1] * len(fake_videos)

        if self.shuffle:
            combined = list(zip(video_paths, labels))
            random.shuffle(combined)
            video_paths, labels = zip(*combined)

        return list(video_paths), list(labels)

    def _process_video(self, video_path, label):
        # video_path là string
        frames = extract_frames(video_path.decode("utf-8"), num_frames=self.num_frames, target_size=IMAGE_SIZE)
        frames = frames.astype(np.float32) / 255.0
        return frames, np.int32(label)

    def _tf_wrapper(self, video_path, label):
        frames, label = tf.numpy_function(
            self._process_video, [video_path, label], [tf.float32, tf.int32]
        )
        frames.set_shape((self.num_frames, *IMAGE_SIZE, 3))
        label.set_shape(())
        return frames, label

    def _augment(self, frames, label):
        frames = tf.image.random_flip_left_right(frames)
        frames = tf.image.random_brightness(frames, max_delta=0.1)
        frames = tf.image.random_contrast(frames, 0.9, 1.1)
        frames = tf.image.random_saturation(frames, 0.9, 1.1)
        return frames, label

    def as_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices((self.video_paths, self.labels))
        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(self.video_paths))

        ds = ds.map(self._tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        if self.augment and self.split == "train":
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
