import argparse
import os
import json
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from PIL import Image
from ultralytics import YOLO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoDataset(Dataset):
    def __init__(self, video_paths, target_fps=5):
        self.video_paths = video_paths
        self.target_fps = target_fps

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = []
        original_indices = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video file {video_path}")
                return video_path, [], []

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, round(fps / self.target_fps)) if fps > 0 else 1

            for frame_id in range(frames_num):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    frames.append(pil_img)
                    original_indices.append(frame_id)
            cap.release()
            return video_path, frames, original_indices
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            return video_path, [], []


def collate_fn(batch):
    return batch[0]


def main(opt):
    detector = YOLO('yolov8n-face.pt').to(device).eval()

    with open(opt.list_file, 'r') as f:
        video_files = [line.strip().split()[1] for line in f if line.strip()]
    video_paths = list(dict.fromkeys([os.path.join(opt.data_path, fname) for fname in video_files]))

    os.makedirs(opt.output_path, exist_ok=True)

    processed = {d for d in os.listdir(opt.output_path) if os.path.isdir(os.path.join(opt.output_path, d))}
    video_paths = [vp for vp in video_paths if os.path.splitext(os.path.basename(vp))[0] not in processed]

    print(f"실제로 처리할 비디오 수: {len(video_paths)}")

    dataset = VideoDataset(video_paths, target_fps=opt.target_fps)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers, collate_fn=collate_fn)

    for video_path, frames, original_indices in tqdm(dataloader, desc="Detecting Faces"):
        if not frames:
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(opt.output_path, video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "video.json")

        results = detector.predict(
            frames,
            conf=0.5,
            iou=0.7,
            device=device,
            verbose=False,
            stream=False,
            batch=opt.batch_size,
        )

        output_data = {}
        score_threshold = 0.8
        for i, res in enumerate(results):
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            high_conf_boxes = [box.tolist() for box, conf in zip(boxes_xyxy, confs) if conf > score_threshold]

            if high_conf_boxes:
                frame_idx = original_indices[i]
                output_data[frame_idx] = high_conf_boxes

        if output_data:
            with open(output_file, 'w') as f:
                json.dump(output_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default='./video_list.txt', type=str)
    parser.add_argument('--data_path', default='../videos', type=str)
    parser.add_argument('--output_path', default='../boxes', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--target_fps', default=5, type=int, help='Frames per second to sample from video')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for YOLOv8 prediction')

    opt = parser.parse_args()

    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()

    main(opt)
