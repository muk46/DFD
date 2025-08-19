# 1. 베이스 이미지 선택
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 시스템 라이브러리 설치 (wget 추가)
RUN apt-get update && apt-get install -y \
    wget \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# 4. 사용자 생성
RUN useradd -m -u 1000 user

# 5. 캐시 디렉토리 환경변수 설정
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV HF_HUB_CACHE=/tmp/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

# 6. 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gdown

# 7. 모델 다운로드
RUN mkdir -p /app/models
RUN wget -O /app/models/yolov8n-face.pt https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt
RUN gdown --id '1kP4I12SlN_2lEyr3Z1Lh4k-EyVtRdGwS' -O /app/models/Model_checkpoint14.pth
RUN gdown --id '1bBNhCrV9KzZaJgp5xZ5Ir6E2ADy4OW-D' -O /app/models/Extractor_checkpoint14.pth

# 8. 프로젝트 소스 코드 복사
COPY . .

# 9. 파일 소유권 변경
RUN chown -R user:user /app

# 10. 일반 사용자로 전환
USER user

# 11. 서버 실행 명령어
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:7860", "app:app"]
