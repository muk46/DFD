import cv2
import numpy as np
import os

def create_xai_visualization(video_path, xai_data, save_path):
    """
    XAI 데이터를 기반으로 히트맵을 생성하고 원본 프레임에 합성하여 저장합니다.
    """
    # 1. 가장 중요한 XAI 데이터 선택 (일단 첫 번째 프레임 사용)
    if not xai_data:
        print("Warning: XAI data is empty. Cannot generate visualization.")
        return None
    
    principal_frame_data = xai_data[0]
    frame_index = principal_frame_data['frame_index']
    bbox = [int(v) for v in principal_frame_data['bbox']]
    heatmap_data = np.array(principal_frame_data['heatmap'], dtype=np.float32)

    # 2. OpenCV로 특정 프레임 읽어오기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Failed to read frame at index {frame_index}")
        return None

    # 3. 히트맵 생성 및 합성
    x1, y1, x2, y2 = bbox
    
    # Bbox 좌표가 프레임 경계를 벗어나지 않도록 보정
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if y1 >= y2 or x1 >= x2:
        print("Warning: Invalid bounding box dimensions.")
        return None

    face_roi = frame[y1:y2, x1:x2]
    face_height, face_width, _ = face_roi.shape

    if face_height == 0 or face_width == 0:
        print("Warning: Face ROI has zero dimension.")
        return None

    # 히트맵 크기를 얼굴 바운딩 박스에 맞게 확대
    heatmap_resized = cv2.resize(heatmap_data, (face_width, face_height))
    
    # 컬러맵 적용 (0~1 값을 0~255로 스케일링 후)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # 원본 얼굴과 히트맵 합성
    overlayed_face = cv2.addWeighted(face_roi, 0.6, heatmap_colored, 0.4, 0)
    
    # 원본 프레임에 합성된 얼굴 다시 넣기
    frame[y1:y2, x1:x2] = overlayed_face

    # 4. 최종 이미지 저장 및 경로 반환
    try:
        cv2.imwrite(save_path, frame)
        return save_path
    except Exception as e:
        print(f"Error saving final image: {e}")
        return None