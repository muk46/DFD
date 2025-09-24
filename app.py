import argparse
import os
import uuid
import torch
import yaml
import cv2 # OpenCV for visualization
import numpy as np # Numpy for heatmap analysis

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# 저희의 predict.py에서 필요한 모든 함수를 가져옵니다.
from predict import (
    detect_faces,
    extract_crops,
    predict,
    load_models
)

# ----------------------------------------
# Flask 앱 및 모델 설정
# ----------------------------------------

# 1. 모델 로딩을 위한 설정값 정의
args = argparse.Namespace(
    config='config/size_invariant_timesformer.yaml',
    # 저희가 훈련시킨 메인 모델(분류기)의 경로입니다.
    model_weights='models/Model_checkpoint14.pth',
    # 특징 추출기는 사전 훈련 모델을 사용하므로, 별도 경로가 필요 없습니다.
    detector_type='FacenetDetector',
    gpu_id=0,
    save_attentions=True # XAI를 위해 항상 True로 설정
)

# 2. YAML 설정 파일 로드
with open(args.config, 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

# 3. 모델 로딩 (서버 시작 시 1회만 실행)
print("Loading models, this may take a moment...")
try:
    # predict.py의 load_models 함수를 호출합니다.
    # 이 함수는 이제 사전 훈련된 특징 추출기를 자동으로 불러옵니다.
    model, features_extractor, config, device = load_models(args, config)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model, features_extractor, device = None, None, None

# 4. Flask 앱 초기화
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


# ----------------------------------------
# XAI 히트맵 생성 함수
# ----------------------------------------
def create_xai_visualization(video_path, xai_data):
    """XAI 데이터를 받아 히트맵 이미지를 생성하고, 웹 경로를 반환합니다."""
    if not xai_data:
        return None

    # 분석에 사용할 핵심 데이터 추출
    principal_frame_data = xai_data[0]
    frame_index = principal_frame_data['frame_index']
    bbox = [int(p) for p in principal_frame_data['bbox']]
    heatmap_data = np.array(principal_frame_data['heatmap'])

    # 원본 영상에서 해당 프레임 읽기
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    # 얼굴 영역(ROI) 추출
    x1, y1, x2, y2 = bbox
    face_roi = frame[y1:y2, x1:x2]

    # 얼굴이 없는 경우 예외 처리
    face_height, face_width, _ = face_roi.shape
    if face_height == 0 or face_width == 0:
        return None

    # 히트맵 생성 및 덧씌우기
    heatmap_resized = cv2.resize(heatmap_data, (face_width, face_height))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlayed_face = cv2.addWeighted(face_roi, 0.6, heatmap_colored, 0.4, 0)

    # 원본 프레임에 히트맵이 적용된 얼굴 합성
    frame[y1:y2, x1:x2] = overlayed_face

    # 최종 결과 이미지 저장
    heatmap_filename = f"heatmap_{uuid.uuid4()}.jpg"
    heatmap_save_path = os.path.join(app.config['RESULT_FOLDER'], heatmap_filename)
    cv2.imwrite(heatmap_save_path, frame)

    # 웹 브라우저에서 접근 가능한 이미지 경로 반환
    return url_for('static', filename=f'results/{heatmap_filename}')


# ----------------------------------------
# 웹 라우팅 (URL 처리)
# ----------------------------------------

@app.route('/')
def index():
    """메인 페이지를 보여줍니다."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def handle_prediction():
    """동영상 파일을 받아 딥페이크를 탐지하고, 결과를 JSON으로 반환합니다."""
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    # 1. 파일 유효성 검사
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No file part"}), 400

    f = request.files['file']

    # 2. 임시 파일 저장
    unique_name = f"{uuid.uuid4()}_{secure_filename(f.filename)}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    f.save(video_path)

    try:
        # 3. 딥페이크 탐지 파이프라인 실행
        bboxes_dict = detect_faces(video_path, args.detector_type, args)
        crops = extract_crops(video_path, bboxes_dict)

        # predict.py의 predict 함수는 이제 5개의 값을 반환합니다.
        pred_score, heatmap_data, identities, frames_list, bboxes = predict(
            video_path, crops, config, args,
            model=model, features_extractor=features_extractor, device_override=device
        )

        # 4. XAI 데이터 가공 및 시각화
        # 분석 결과 중 가장 중요한 첫 번째 프레임의 정보를 사용합니다.
        target_frame_index = frames_list[0][0]
        target_bbox = bboxes[0]
        target_heatmap = heatmap_data[0]

        xai_data = [{
            "frame_index": target_frame_index,
            "bbox": target_bbox,
            "heatmap": target_heatmap.tolist()
        }]

        heatmap_path = create_xai_visualization(video_path, xai_data)

        # 5. 최종 결과 JSON으로 반환
        result = {
            "prediction": float(pred_score),
            "is_fake": bool(pred_score > 0.5),
            "heatmap_path": heatmap_path
        }
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        # 6. 임시 파일 삭제
        if os.path.exists(video_path):
            os.remove(video_path)

# ----------------------------------------
# 서버 실행
# ----------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)