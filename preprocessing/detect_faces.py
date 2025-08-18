import argparse
import os
import json
import torch
import shutil
import pandas as pd
from ultralytics import YOLO
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial
import cv2
import sys
from torch.utils.data import Dataset, DataLoader

if sys.platform.startswith('win'):
    import multiprocessing
    multiprocessing.freeze_support()

class VideoDataset(Dataset):
    def __init__(self, videos):
        self.videos = videos

    def __len__(self):
        return len(self.videos)
        
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        video_name = os.path.basename(video_path)
        
        frames = []
        try:
            # OpenCV를 사용하여 비디오를 읽습니다.
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"Warning: Could not open video file {video_path} with cv2. Skipping.")
                return video_name, None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            if not frames:
                print(f"Warning: No frames found in video {video_path}. Skipping.")
                return video_name, None
        
        except Exception as e:
            print(f"Warning: Error reading video {video_path}. Skipping. Error: {e}")
            return video_name, None
        
        return video_name, frames

def process_videos(videos, opt):
    device = torch.device('cuda:{}'.format(opt.gpu_id) if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # YOLO 모델 로드
    model = YOLO('yolov8n-face.pt')
    model.to(device)
    model.eval()
    
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, num_workers=opt.workers, batch_size=1, collate_fn=lambda x: x)
    
    os.makedirs(opt.output_path, exist_ok=True)
    
    bar = tqdm(loader, desc="비디오 얼굴 감지 중")
    
    for item in bar:
        video_name, frames = item[0]

        if frames is None:
            continue

        video_name_no_ext = os.path.splitext(video_name)[0]
        output_dir = os.path.join(opt.output_path, video_name_no_ext)
        os.makedirs(output_dir, exist_ok=True)
        
        results = model.predict(source=frames, device=device, stream=False, conf=0.5, iou=0.7, verbose=False)
        
        result = {}
        for frame_idx, res in enumerate(results):
            if res.boxes:
                result[str(frame_idx)] = res.boxes.xyxy.cpu().numpy().tolist()
        
        with open(os.path.join(output_dir, "video.json"), "w") as f:
            json.dump(result, f)
        
        found_faces = False
        for key in result:
            if isinstance(result[key], list) and len(result[key]) > 0:
                found_faces = True
                break

        if not found_faces:
            print(f"Faces not found for {video_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default='./video_list.txt', type=str)
    parser.add_argument('--data_path', default='../videos', type=str)
    parser.add_argument('--output_path', default='../boxes', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--workers', default=0, type=int)
    opt = parser.parse_args()

    videos_paths = []
    with open(opt.list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                videos_paths.append(os.path.join(opt.data_path, parts[1].strip()))

    if os.path.exists(opt.output_path):
        processed_videos = os.listdir(opt.output_path)
        videos_paths = [v for v in videos_paths if os.path.splitext(os.path.basename(v))[0] not in processed_videos]

    print(f"실제로 처리할 비디오 수: {len(videos_paths)}")

    if not videos_paths:
        print("Warning: No new videos to process. Exiting.")
    else:
        process_videos(videos_paths, opt)

if __name__ == '__main__':
    main()