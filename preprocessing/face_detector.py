import cv2
from PIL import Image
from ultralytics import YOLO
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import List

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

class VideoFaceDetector(ABC):
    @property
    @abstractmethod
    def _batch_size(self) -> int:
        pass

    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass

class FacenetDetector(VideoFaceDetector):
    def __init__(self, device="cpu") -> None:
        self.device = torch.device(device)
        self.detector = YOLO('yolov8n-face.pt').to(self.device).eval()

    def _detect_faces(self, frames) -> List:
        detected_boxes_per_frame = []
        results = self.detector.predict(frames, conf=0.5, iou=0.7, device=self.device, verbose=False, stream=False)

        score_threshold = 0.8
        for res in results:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            high_conf_boxes = [box.tolist() for box, conf in zip(boxes_xyxy, confs) if conf > score_threshold]
            detected_boxes_per_frame.append(high_conf_boxes if high_conf_boxes else None)
        return detected_boxes_per_frame

    @property
    def _batch_size(self):
        return 1

class VideoDataset(Dataset):
    def __init__(self, videos) -> None:
        self.videos = videos

    def __getitem__(self, index: int):
        video = self.videos[index]
        try:
            capture = cv2.VideoCapture(video)
            if not capture.isOpened():
                return video, [], 0, []

            frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                capture.release()
                return video, [], 0, []

            frames_list_pil = []
            for _ in range(frames_num):
                success, frame = capture.read()
                if not success:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb).resize((640, 480))
                frames_list_pil.append(frame_pil)
            capture.release()

            if not frames_list_pil:
                return video, [], 0, []

            return video, list(range(len(frames_list_pil))), fps, frames_list_pil
        except:
            if 'capture' in locals() and capture.isOpened():
                capture.release()
            return video, [], 0, []

    def __len__(self) -> int:
        return len(self.videos)
