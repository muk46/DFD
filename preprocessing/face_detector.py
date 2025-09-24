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
    # ✅ 1. 배치 사이즈를 조절할 수 있도록 batch_size 인자 추가
    def __init__(self, device="cpu", batch_size=16) -> None:
        self.device = torch.device(device)
        self.detector = YOLO('yolov8n-face.pt').to(self.device).eval()
        self._batch = batch_size

    def _detect_faces(self, frames) -> List:
        detected_boxes_per_frame = []
        # ✅ 2. predict 함수에 batch 인자를 적용하여 GPU 병렬 처리 효율 극대화
        results = self.detector.predict(
            frames,
            conf=0.5,
            iou=0.7,
            device=self.device,
            verbose=False,
            stream=False,
            batch=self._batch,
        )

        score_threshold = 0.8
        for res in results:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            high_conf_boxes = [box.tolist() for box, conf in zip(boxes_xyxy, confs) if conf > score_threshold]
            detected_boxes_per_frame.append(high_conf_boxes if high_conf_boxes else None)
        return detected_boxes_per_frame

    @property
    def _batch_size(self):
        return self._batch


class VideoDataset(Dataset):
    # ✅ 3. 초당 처리할 프레임 수를 조절하는 target_fps 인자 추가
    def __init__(self, videos, target_fps=5) -> None:
        self.videos = videos
        self.target_fps = target_fps

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

            # ✅ 4. 원본 fps와 target_fps를 기반으로 프레임 샘플링 간격 계산
            frame_interval = max(1, round(fps / self.target_fps))
            
            frames_list_pil = []
            original_indices = [] # ✅ 원본 프레임 번호를 저장할 리스트

            for frame_id in range(frames_num):
                success, frame = capture.read()
                if not success:
                    break
                
                # ✅ 5. 계산된 간격에 맞는 프레임만 추출 (다운샘플링)
                if frame_id % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb).resize((640, 480))
                    frames_list_pil.append(frame_pil)
                    original_indices.append(frame_id) # 원본 프레임 번호 저장
            capture.release()

            if not frames_list_pil:
                return video, [], 0, []
            
            # ✅ 6. 원본 프레임 번호 리스트를 반환하여 후속 처리 단계와의 호환성 유지
            return video, original_indices, fps, frames_list_pil
        except Exception:
            if 'capture' in locals() and capture.isOpened():
                capture.release()
            return video, [], 0, []

    def __len__(self) -> int:
        return len(self.videos)