# 1. 베이스 이미지 선택 (가벼운 Python 3.12 버전)
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 시스템 라이브러리 설치 (Aptfile의 역할)
# && echo "--- apt-get install finished ---" 로 설치 완료 확인 메시지 추가
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    --no-install-recommends && \
    echo "--- apt-get install finished ---" && \
    rm -rf /var/lib/apt/lists/*

# 4. requirements.txt를 먼저 복사하여 라이브러리 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 👇 디버깅 명령어 추가 👇 ---
# OpenCV 라이브러리가 의존하는 시스템 라이브러리 목록을 직접 출력해봅니다.
RUN ldd $(python -c "import cv2; print(cv2.__file__)")
# ---------------------------

# 5. 나머지 프로젝트 소스 코드를 복사
COPY . .

# 6. Gunicorn이 사용할 포트 지정
EXPOSE 8000

# 7. 서버 실행 명령어 (Procfile의 역할)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
