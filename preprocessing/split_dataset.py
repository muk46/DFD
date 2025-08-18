import argparse
import json
import os
import random
from collections import defaultdict
from itertools import chain

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def split_videos_and_save(list_file, ratio, output_path):
    """
    Reads a list of videos, splits them into train, validation, and test sets
    based on the specified ratio, and saves the split information to separate txt files.
    """
    
    print(f"Reading video list from: {list_file}")
    
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    labeled_videos = [line.strip().split(" ") for line in lines if line.strip()]
    
    fake_videos = [v[1] for v in labeled_videos if v[0].upper() == 'FAKE']
    real_videos = [v[1] for v in labeled_videos if v[0].upper() == 'REAL']
    
    # 각 비디오 목록을 섞습니다.
    random.shuffle(fake_videos)
    random.shuffle(real_videos)
    
    # 비율에 따라 비디오를 분할합니다.
    train_ratio, val_ratio, test_ratio = ratio
    
    train_fake_count = int(len(fake_videos) * train_ratio)
    val_fake_count = int(len(fake_videos) * val_ratio)
    
    train_real_count = int(len(real_videos) * train_ratio)
    val_real_count = int(len(real_videos) * val_ratio)
    
    train_videos = fake_videos[:train_fake_count] + real_videos[:train_real_count]
    val_videos = fake_videos[train_fake_count:train_fake_count + val_fake_count] + real_videos[train_real_count:train_real_count + val_real_count]
    test_videos = fake_videos[train_fake_count + val_fake_count:] + real_videos[train_real_count + val_real_count:]
    
    print(f"Total videos: {len(fake_videos) + len(real_videos)}")
    print(f"Train set: {len(train_videos)} videos")
    print(f"Validation set: {len(val_videos)} videos")
    print(f"Test set: {len(test_videos)} videos")
    
    # 분할된 데이터를 별도의 txt 파일로 저장합니다.
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, 'train_videos.txt'), 'w') as f:
        for video_name in train_videos:
            f.write(f"FAKE {video_name.replace('.mp4', '')}\n") if video_name in fake_videos else f.write(f"REAL {video_name.replace('.mp4', '')}\n")

    with open(os.path.join(output_path, 'val_videos.txt'), 'w') as f:
        for video_name in val_videos:
            f.write(f"FAKE {video_name.replace('.mp4', '')}\n") if video_name in fake_videos else f.write(f"REAL {video_name.replace('.mp4', '')}\n")

    with open(os.path.join(output_path, 'test_videos.txt'), 'w') as f:
        for video_name in test_videos:
            f.write(f"FAKE {video_name.replace('.mp4', '')}\n") if video_name in fake_videos else f.write(f"REAL {video_name.replace('.mp4', '')}\n")

    print(f"Splits saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--list_file', default='./video_list.txt', type=str,
                        help='Path to the video list file.')
    parser.add_argument('--faces_path', default='../faces_output/crops', type=str,
                        help='Path to the directory containing cropped face images.')
    parser.add_argument('--output_path', default='../splits', type=str,
                        help='Path to save the split files.')
    parser.add_argument('--ratio', nargs=3, type=float, default=[0.75, 0.1, 0.15],
                        help='Train, validation, test split ratio.')
    
    opt = parser.parse_args()
    print(opt)
    
    if sum(opt.ratio) != 1.0:
        print("Error: The sum of the split ratios must be 1.0.")
        exit()
    
    # 데이터 분할 실행
    split_videos_and_save(opt.list_file, opt.ratio, opt.output_path)