import os
import shutil
import random
from pathlib import Path

def organize_celebdf(source_dir, target_dir, test_list_file, train_ratio=0.8):
    # Xóa thư mục đích nếu đã tồn tại để tránh dư file cũ
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Tạo thư mục đích
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split, 'real'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, 'fake'), exist_ok=True)

    # Đọc danh sách video test (chỉ lấy phần đường dẫn, bỏ nhãn)
    with open(test_list_file, 'r') as f:
        test_videos = set(line.strip().split(' ', 1)[1].replace("\\", "/") for line in f if line.strip())

    # Lấy danh sách video real và fake (đảm bảo dùng dấu '/')
    real_videos = []
    for subdir in ['Celeb-real', 'YouTube-real']:
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.exists(subdir_path):
            real_videos.extend([os.path.join(subdir, f).replace("\\", "/") for f in os.listdir(subdir_path) if f.endswith('.mp4')])

    fake_videos = [os.path.join('Celeb-synthesis', f).replace("\\", "/") for f in os.listdir(os.path.join(source_dir, 'Celeb-synthesis')) if f.endswith('.mp4')]

    # Chia train/val/test
    test_real = [v for v in real_videos if v in test_videos]
    test_fake = [v for v in fake_videos if v in test_videos]

    non_test_real = [v for v in real_videos if v not in test_videos]
    non_test_fake = [v for v in fake_videos if v not in test_videos]

    random.shuffle(non_test_real)
    random.shuffle(non_test_fake)

    train_size_real = int(len(non_test_real) * train_ratio)
    train_size_fake = int(len(non_test_fake) * train_ratio)

    train_real = non_test_real[:train_size_real]
    val_real = non_test_real[train_size_real:]
    train_fake = non_test_fake[:train_size_fake]
    val_fake = non_test_fake[train_size_fake:]

    # Sao chép file
    def copy_files(file_list, target_dir):
        for file_path in file_list:
            src_path = os.path.join(source_dir, file_path)
            target_path = os.path.join(target_dir, os.path.basename(file_path))
            shutil.copy(src_path, target_path)

    copy_files(train_real, os.path.join(target_dir, 'train', 'real'))
    copy_files(val_real, os.path.join(target_dir, 'val', 'real'))
    copy_files(test_real, os.path.join(target_dir, 'test', 'real'))
    copy_files(train_fake, os.path.join(target_dir, 'train', 'fake'))
    copy_files(val_fake, os.path.join(target_dir, 'val', 'fake'))
    copy_files(test_fake, os.path.join(target_dir, 'test', 'fake'))

    # Sao chép file danh sách test
    shutil.copy(test_list_file, target_dir)
    print(f"Dataset organized at {target_dir}")

if __name__ == "__main__":
    source_dir = "./Celeb-DF"
    target_dir = "./dataset"
    test_list_file = "./Celeb-DF/List_of_testing_videos.txt"
    organize_celebdf(source_dir, target_dir, test_list_file)