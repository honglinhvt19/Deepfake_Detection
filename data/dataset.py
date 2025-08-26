import tensorflow as tf
import os
import random
import numpy as np
from .preprocessing import extract_frames, IMAGE_SIZE

class Dataset:
    def __init__(self, data_dir, batch_size=16, num_frames=8, training=False, max_upsampling_ratio=5.0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.training = training
        self.max_upsampling_ratio = max_upsampling_ratio
        self.video_paths, self.labels = self._load_data()

    def _load_data(self):
        video_paths, labels = [], []
        real_videos, fake_videos = [], []
        
        real_dir = os.path.join(self.data_dir, "real")
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    real_videos.append(os.path.join(real_dir, f))
        
        fake_dir = os.path.join(self.data_dir, "fake") 
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    fake_videos.append(os.path.join(fake_dir, f))

        print(f"Original data - Real: {len(real_videos)}, Fake: {len(fake_videos)}")

        if self.training:
            all_videos, all_labels = self._smart_upsampling(real_videos, fake_videos)
        else:
            all_videos = real_videos + fake_videos
            all_labels = [0] * len(real_videos) + [1] * len(fake_videos)

        combined = list(zip(all_videos, all_labels))
        random.shuffle(combined)
        video_paths, labels = zip(*combined) if combined else ([], [])

        return list(video_paths), list(labels)

    
    def _smart_upsampling(self, real_videos, fake_videos):
        if len(real_videos) == 0 or len(fake_videos) == 0:
            return real_videos + fake_videos, [0] * len(real_videos) + [1] * len(fake_videos)
        
        minority_class = real_videos if len(real_videos) < len(fake_videos) else fake_videos
        majority_class = fake_videos if len(real_videos) < len(fake_videos) else real_videos
        minority_label = 0 if len(real_videos) < len(fake_videos) else 1
        majority_label = 1 - minority_label
        
        imbalance_ratio = len(majority_class) / len(minority_class)
        
        if imbalance_ratio <= 3.0:
            target_minority_size = len(minority_class)
            
        elif imbalance_ratio <= 5.0:
            target_minority_size = int(len(majority_class) / 2.5)
            
        else:
            max_minority_size = int(len(minority_class) * self.max_upsampling_ratio)
            target_from_ratio = len(majority_class) // 2
            target_minority_size = min(max_minority_size, target_from_ratio)
        
        num_to_add = max(0, target_minority_size - len(minority_class))
        
        if num_to_add > 0:
            upsampled_minority = random.choices(minority_class, k=num_to_add)
            all_videos = minority_class + upsampled_minority + majority_class
            all_labels = ([minority_label] * (len(minority_class) + num_to_add) + 
                         [majority_label] * len(majority_class))
        else:
            all_videos = minority_class + majority_class
            all_labels = [minority_label] * len(minority_class) + [majority_label] * len(majority_class)
        
        return all_videos, all_labels

    def load_test_dataset(self):
        real_dir = os.path.join(self.data_dir, "test", "real")
        fake_dir = os.path.join(self.data_dir, "test", "fake")

        video_paths, labels = [], []

        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_paths.append(os.path.join(real_dir, f))
                    labels.append(0)  # real = 0

        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_paths.append(os.path.join(fake_dir, f))
                    labels.append(1)  # fake = 1

        print(f"Loaded test set - Real: {labels.count(0)}, Fake: {labels.count(1)}")

        def generator():
            for path, label in zip(video_paths, labels):
                try:
                    frames = extract_frames(path, self.num_frames, IMAGE_SIZE)
                    yield frames, label
                except Exception as e:
                    print(f"[ERROR] {path} -> {e}")

        output_signature = (
            tf.TensorSpec(shape=(self.num_frames, *IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

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

    def as_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.video_paths, self.labels))
        if self.training:
            dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
        dataset = dataset.map(self._tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        if self.training:
            dataset = dataset.map(self._augment_video, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def load_split_dataset(base_dir, split='train', batch_size=8, num_frames=8, 
                         max_upsampling_ratio=3.0):

        split_dir = os.path.join(base_dir, split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        training = (split == 'train')
        
        dataset = Dataset(
            data_dir=split_dir,
            batch_size=batch_size,
            num_frames=num_frames,
            training=training,
            max_upsampling_ratio=max_upsampling_ratio
        )
        
        return dataset


