import cv2
from ultralytics import YOLO
from tqdm import tqdm
import os

# --- 설정 (여기만 수정하세요) ---
MODEL_PATH = 'yolov8n-face.pt'
INPUT_VIDEO_PATH = 'input_video.mp4'  # 얼굴을 찾을 원본 영상 파일
OUTPUT_VIDEO_PATH = 'output_video.mp4' # 네모 박스가 그려진 결과 영상 파일
CONFIDENCE_THRESHOLD = 0.5 # 얼굴로 인식할 최소 신뢰도 (0.5 = 50%)
# ------------------------------------

# 1. YOLO 모델 로드
try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLO 모델을 성공적으로 로드했습니다.")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    exit()

# 2. 원본 비디오 파일 열기
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 오류: 비디오 파일을 열 수 없습니다: {INPUT_VIDEO_PATH}")
    exit()

# 3. 결과 비디오 저장을 위한 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 코덱 설정 (mp4v는 .mp4 파일 형식에 적합)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
print(f"🎬 결과 영상 파일 '{OUTPUT_VIDEO_PATH}' (해상도: {frame_width}x{frame_height}, FPS: {fps})")

# 4. 프레임별로 얼굴 탐지 및 결과 영상 제작
with tqdm(total=total_frames, desc="영상 처리 중") as pbar:
    while cap.isOpened():
        # 프레임 하나씩 읽기
        success, frame = cap.read()
        if not success:
            break

        # 현재 프레임에서 얼굴 탐지 실행
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # ultralytics 라이브러리의 plot() 함수로 결과 이미지를 바로 생성
        annotated_frame = results[0].plot()

        # 네모 박스가 그려진 프레임을 결과 비디오에 쓰기
        out.write(annotated_frame)
        
        pbar.update(1) # 진행상황 표시 업데이트

# 5. 작업 완료 후 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✨ 완료! 얼굴 탐지 결과가 '{OUTPUT_VIDEO_PATH}' 파일로 저장되었습니다.")