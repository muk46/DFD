import os
import uuid
import yaml
import argparse
import torch
import yt_dlp
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
# 프로젝트 모듈
# (실제 프로젝트에서는 predict, detect_faces, extract_crops 등을 import 해야 함)
# 예시를 위해 DFD_predict 모듈로 가정
import predict as DFD_predict 

# ----------------------------------------
# Flask 앱 설정
# ----------------------------------------
app = Flask(__name__)
CORS(app)  # ✨ 2. 이 줄을 추가!
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ----------------------------------------
# 모델 로딩 (서버 시작 시 1회 실행)
# ----------------------------------------
args = argparse.Namespace(
    config='config/size_invariant_timesformer.yaml',
    model_weights=os.getenv('MODEL_WEIGHTS_PATH', 'outputs/models/Model_checkpoint3.pth'), #  훈련시킨 메인 모델 경로
    detector_type="FacenetDetector",
    gpu_id=0,
    save_attentions=True
)


with open(args.config, 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

print("Loading models, this may take a moment...")
try:
    model, features_extractor, config, device = DFD_predict.load_models(args, config)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model, features_extractor, device = None, None, None

# ----------------------------------------
# XAI 히트맵 생성 (개발 중인 실제 함수)
# TODO: 이 함수를 완성해야 함
# ----------------------------------------
import cv2
import numpy as np

def create_xai_visualization(video_path, xai_data):
    if not xai_data: return None
    
    principal_frame_data = xai_data[0] # 가장 중요한 첫번째 프레임 정보 사용
    frame_index = principal_frame_data['frame_index']
    bbox = [int(p) for p in principal_frame_data['bbox']] # 정수형으로 변환
    heatmap_data = np.array(principal_frame_data['heatmap'])

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None

    x1, y1, x2, y2 = bbox
    face_roi = frame[y1:y2, x1:x2]
    face_height, face_width, _ = face_roi.shape

    if face_height == 0 or face_width == 0: return None

    heatmap_resized = cv2.resize(heatmap_data, (face_width, face_height))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    overlayed_face = cv2.addWeighted(face_roi, 0.6, heatmap_colored, 0.4, 0)
    frame[y1:y2, x1:x2] = overlayed_face

    heatmap_filename = f"heatmap_{uuid.uuid4()}.jpg"
    heatmap_save_path = os.path.join(app.config['RESULT_FOLDER'], heatmap_filename)
    cv2.imwrite(heatmap_save_path, frame)
    
    return url_for('static', filename=f'results/{heatmap_filename}')

# ----------------------------------------
# 라우팅
# ----------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    video_path = None
    try:
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(video_path)
        elif 'youtube_url' in request.form and request.form['youtube_url'] != '':
            url = request.form['youtube_url']
            unique_filename = f"{uuid.uuid4()}.mp4"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            ydl_opts = {'format': 'bestvideo[ext=mp4]/best[ext=mp4]','outtmpl': video_path,'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        else:
            return jsonify({"error": "No file or URL provided."}), 400

        # 딥페이크 탐지 파이프라인
        bboxes_dict = DFD_predict.detect_faces(video_path, args.detector_type, args)
        if not bboxes_dict or not any(bboxes_dict.values()):
             return jsonify({"error": "No faces detected in the video."}), 400
        
        crops = DFD_predict.extract_crops(video_path, bboxes_dict)
        
        pred_score, attentions, identities, frames_list, bboxes = DFD_predict.predict(
            video_path, crops, config, args,
            model=model, features_extractor=features_extractor, device_override=device
        )
        
        # XAI 데이터 가공 (utils.py의 함수를 호출한다고 가정)
        # xai_data = DFD_utils.aggregate_attentions(attentions, identities, frames_list, bboxes)

        # 임시 데이터 (위 함수가 완성되면 대체)
        # ✨✨✨ 여기가 수정된 부분! ✨✨✨
        # bboxes[0] 대신 bboxes 리스트 전체를 넘겨줌
        xai_data = [{
            "frame_index": frames_list[0][0],
            "bbox": bboxes[0], # <-- 다시 [0]을 추가!
            "heatmap": np.random.rand(14, 14).tolist() # 실제로는 attentions 값으로 생성
        }]




        # XAI 히트맵 생성
        heatmap_path = create_xai_visualization(video_path, xai_data)
        if not heatmap_path:
             # 히트맵 생성 실패 시 플레이스홀더 사용
             heatmap_path = url_for('static', filename='placeholder_heatmap.jpg')


        response_data = {
            "prediction": float(pred_score),
            "is_fake": bool(pred_score > 0.5),
            "heatmap_path": heatmap_path
        }
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)