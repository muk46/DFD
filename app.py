import argparse
import os
import uuid
import torch
import yaml

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# 필요한 함수만 가져오기
from predict import (
    detect_faces, extract_crops, predict,
    load_models
)

# ----- 모델 초기화 -----
opt = argparse.Namespace(
    detector_type='FacenetDetector',
    random_state=42,
    gpu_id=-1,
    workers=0,
    config='config/size_invariant_timesformer.yaml',
    # 1. model_weights 경로를 환경 변수에서 읽어오기
    model_weights=os.getenv('MODEL_WEIGHTS_PATH', '/app/models/Model_checkpoint14.pth'), 
    
    # 2. extractor_weights 경로도 환경 변수에서 읽어오기
    extractor_weights=os.getenv('EXTRACTOR_WEIGHTS_PATH', '/app/models/Extractor_checkpoint14.pth'),
    
    output_type=0,
    save_attentions=False,
    extractor_model=2
)

# 모델 및 설정 불러오기 (EfficientNetV2 + TimeSformer)
model, features_extractor, config, device = load_models(opt)

# ----- Flask 앱 설정 -----
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def _is_allowed(filename: str) -> bool:
    """허용된 확장자 여부 확인"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    # 업로드된 파일 유효성 확인
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not _is_allowed(f.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    # 고유 이름으로 파일 저장
    filename = secure_filename(f.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    video_path = os.path.join(UPLOAD_FOLDER, unique_name)
    f.save(video_path)

    try:
        # 1. 얼굴 탐지
        bboxes_dict = detect_faces(video_path, opt.detector_type, opt)

        # 2. 얼굴 크롭
        crops = extract_crops(video_path, bboxes_dict)

        # 3. 딥페이크 예측 (EfficientNetV2 기반)
        pred, _, _, _, _ = predict(
            video_path,
            crops,
            config,
            opt,
            model=model,
            features_extractor=features_extractor,
            device_override=device
        )

        result = {
            "prediction": float(pred),
            "is_fake": bool(pred > 0.5)
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # 업로드된 파일 정리
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
