import argparse
import json
import os
import cv2
from multiprocessing.pool import Pool
from progress.bar import ChargingBar
from functools import partial
import av
import sys
import multiprocessing

# Windows에서 multiprocessing 이슈 방지
if sys.platform.startswith('win'):
    multiprocessing.freeze_support()

def read_video_list(list_file):
    video_list = {}
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = parts[0].strip()
                video_name = parts[1].strip()
                video_list[video_name] = label
    return video_list

def extract_crops_from_video(video_name, output_path, boxes_path, video_list, data_path):
    # 비디오 이름에 확장자를 추가하지 않고 바로 경로를 구성
    video_path = os.path.join(data_path, video_name)
    
    if video_name not in video_list:
        print(f"Warning: {video_name} not found in video list. Skipping.")
        return

    boxes_file_path = os.path.join(boxes_path, os.path.splitext(video_name)[0], "video.json")

    if not os.path.exists(boxes_file_path):
        print(f"Warning: Bounding box data not found for {video_name}. Skipping.")
        return

    with open(boxes_file_path, "r") as f:
        video_data = json.load(f)

    try:
        container = av.open(video_path)
    except av.error.FFmpegError as e:
        print(f"Warning: Could not open video {video_path}. Skipping. Error: {e}")
        return
    except Exception as e:
        print(f"Warning: Unexpected error while opening video {video_path}. Skipping. Error: {e}")
        return

    # 비디오 리스트에서 레이블(FAKE/REAL)을 가져옵니다.
    label = video_list[video_name]
    label_dir = "FAKE" if label.upper() == 'FAKE' else "REAL"
    video_id = os.path.splitext(video_name)[0]

    # FAKE/REAL 중간 폴더를 포함한 최종 경로를 만듭니다.
    output_dir = os.path.join(output_path, label_dir, video_id)

    os.makedirs(output_dir, exist_ok=True)

    for frame_idx, frame in enumerate(container.decode(video=0)):
        frame_idx_str = str(frame_idx)
        if frame_idx_str not in video_data:
            continue
        frame_boxes = video_data[frame_idx_str]
        if not frame_boxes:
            continue

        img = frame.to_rgb().to_ndarray()
        for face_idx, box in enumerate(frame_boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            face_crop = img[y1:y2, x1:x2]

            filename = f"{frame_idx}_{face_idx}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

    container.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default='./video_list.txt', type=str,
                        help='Path to file containing a list of video files to extract faces from.')
    parser.add_argument('--data_path', default='../videos', type=str,
                        help='Path of folder containing videos to be processed.')
    parser.add_argument('--output_path', default='../faces_output', type=str,
                        help='Path of folder where extracted faces will be saved.')
    parser.add_argument('--boxes_path', default='../boxes', type=str,
                        help='Path to the directory containing bounding box JSON files.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of parallel processes.')
    
    opt = parser.parse_args()
    print(opt)

    video_list = read_video_list(opt.list_file)
    if not video_list:
        print("Error: No videos found in the list file.")
        exit()

    processed_videos = set(os.listdir(opt.output_path)) if os.path.isdir(opt.output_path) else set()
    videos_to_process = [
        # 확장자를 추가하지 않고 바로 사용
        video_name
        for video_name in video_list
        if os.path.splitext(video_name)[0] not in processed_videos
    ]

    print(f"Found {len(videos_to_process)} videos to process.")

    if not videos_to_process:
        print("No new videos to process. Exiting.")
        exit()

    with Pool(processes=opt.workers) as p:
        bar = ChargingBar('Extracting face crops', max=len(videos_to_process))
        for _ in p.imap_unordered(
            partial(extract_crops_from_video, output_path=opt.output_path, video_list=video_list, boxes_path=opt.boxes_path, data_path=opt.data_path),
            videos_to_process
        ):
            bar.next()
        bar.finish()

    print("All face crops extracted.")