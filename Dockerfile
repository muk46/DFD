# 1. 베이스 이미지 선택 (가벼운 Python 3.12 버전)
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 시스템 라이브러리 설치 (Aptfile의 역할)
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# 4. Hugging Face, matplotlib, Ultralytics 캐시 디렉토리 환경변수 설정
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV HF_HUB_CACHE=/tmp/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

# 5. requirements.txt를 먼저 복사하여 라이브러리 설치 (캐시 활용)
COPY requirements.txt .
# gdown을 requirements.txt에 추가했거나, 여기서 직접 설치
RUN pip install --no-cache-dir -r requirements.txt gdown

# 모델을 저장할 디렉토리 생성
RUN mkdir -p /app/models

# 1. TimeSformer 모델 다운로드 (파일 ID만 사용)
RUN gdown --id '1kP4I12SlN_2lEyr3Z1Lh4k-EyVtRdGwS' -O /app/models/Model_checkpoint14.pth

# 2. Extractor 모델 다운로드 (파일 ID만 사용)
RUN gdown --id '1bBNhCrV9KzZaJgp5xZ5Ir6E2ADy4OW-D' -O /app/models/Extractor_checkpoint14.pth

# 6. 나머지 프로젝트 소스 코드를 복사
COPY . .

# 7. 서버 실행 명령어 (Procfile의 역할)
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "app:app"]
