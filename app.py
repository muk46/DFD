import argparse
import os
import uuid
import torch
import yaml
import cv2 
import numpy as np 

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from youtube_downloader import download_youtube_video
from predict import generate_face_analysis_results

from predict import (
    detect_faces,
    extract_crops,
    predict,
    load_models
)

# 설정값 정의
args = argparse.Namespace(
    config='config/size_invariant_timesformer.yaml',
    model_weights="outputs/models/Model_checkpoint10.pth",
    extractor_weights="outputs/models/Extractor_checkpoint10.pth",
    detector_type='FacenetDetector',
    gpu_id=0,
)

with open(args.config, 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

print("Loading models...")
try:
    model, features_extractor, config, device = load_models(args, config)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model, features_extractor, device = None, None, None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def handle_prediction():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No file part"}), 400

    f = request.files['file']

    unique_name = f"{uuid.uuid4()}_{secure_filename(f.filename)}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    f.save(video_path)

    try:
        bboxes_dict = detect_faces(video_path, args.detector_type, args)
        crops = extract_crops(video_path, bboxes_dict)

        pred_score, _, heatmap_per_frame, _, frames_list, _ = predict(
            video_path, crops, config, args,
            model=model, features_extractor=features_extractor, device_override=device
        )
        org_vid_name, map_vid_name, attention_graph_data = generate_face_analysis_results(
            frames_list, 
            heatmap_per_frame, 
            app.config['RESULT_FOLDER'], 
            unique_name
        )
        result = {
            "prediction": float(pred_score),
            "is_fake": bool(pred_score > 0.97),
            # 얼굴만 잘린 영상 경로 반환
            "face_org_video": url_for('static', filename=f'results/{org_vid_name}'), 
            "face_map_video": url_for('static', filename=f'results/{map_vid_name}'),
            # 그래프용 데이터 (배열)
            "attention_graph": attention_graph_data 
        }
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route('/predict_url', methods=['POST'])
def handle_url_prediction():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data['url']
    start_time = data.get('start_time', None)
    end_time = data.get('end_time', None)
    video_path = None 

    try:
        print(f"URL 수신: {url}. 다운로드 시작...")
        
        video_path = download_youtube_video(
            url, 
            app.config['UPLOAD_FOLDER'],
            start_time, 
            end_time
        )

        if video_path is None or not os.path.exists(video_path):
            raise Exception("Video download failed or file not found.")
        
        print(f"다운로드 완료: {video_path}. 예측 시작...")

        bboxes_dict = detect_faces(video_path, args.detector_type, args)
        crops = extract_crops(video_path, bboxes_dict)

        pred_score, _, _, _, _ = predict(
            video_path, crops, config, args,
            model=model, features_extractor=features_extractor, device_override=device
        )

        result = {
            "prediction": float(pred_score),
            "is_fake": bool(pred_score > 0.97)
        }
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        if video_path and os.path.exists(video_path):
            print(f"임시 파일 정리: {video_path}")
            os.remove(video_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
