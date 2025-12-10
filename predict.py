import argparse
import cv2
import numpy as np
import yaml
import random
import os

from typing import Type
from PIL import Image
from statistics import mean
from einops import rearrange

import torch
import timm

from utils import aggregate_attentions, draw_border, save_attention_plots, save_explanation_image
import uuid 
from torch.utils.data.dataloader import DataLoader

# 프로젝트 모듈
import preprocessing.face_detector as face_detector
from preprocessing.face_detector import VideoDataset, VideoFaceDetector
from preprocessing.utils import preprocess_images, _generate_connected_components

from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

from albumentations import Compose, PadIfNeeded, Resize
from transforms.albu import IsotropicResize

from models.size_invariant_timesformer import SizeInvariantTimeSformer
from utils import aggregate_attentions, draw_border, save_attention_plots

# 전역 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def identity_collate_fn(x):
    return x

# 사이즈 임베딩 구간 테이블
RANGE_SIZE = 5
SIZE_EMB_DICT = [(1+i*RANGE_SIZE, (i+1)*RANGE_SIZE) if i != 0 else (0, RANGE_SIZE) for i in range(20)]

# 1. 얼굴 탐지
def detect_faces(video_path, detector_cls: Type[VideoFaceDetector], opt):

    if torch.cuda.is_available() and opt.gpu_id >= 0:
        device_for_detector = torch.device(f"cuda:{opt.gpu_id}")
    else:
        device_for_detector = torch.device("cpu")
            
    detector = face_detector.__dict__[detector_cls](device=device_for_detector)
    dataset = VideoDataset([video_path])
    loader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1, collate_fn=identity_collate_fn)
    
    # 얼굴 탐지
    for item in loader:
        bboxes = {}
        video, indices, fps, frames = item[0]
        detections = detector._detect_faces(frames)  
        bboxes.update({i: b for i, b in zip(indices, detections)})
        
        found_faces = any(isinstance(bboxes[k], list) and len(bboxes[k]) > 0 for k in bboxes)
        if not found_faces:
            raise Exception("No faces found.")
            
    return bboxes
    
# 2. 얼굴 크롭 
def extract_crops(video_path, bboxes_dict):

    frames = []
    cap = cv2.VideoCapture(video_path)
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 25

    # 원본 프레임 전체 로딩
    for _ in range(frames_num):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    crops = []
    explored_indexes = []

    # 초당 1장 간격으로 샘플링 
    for i in range(0, len(frames), max(fps, 1)):
        while i not in bboxes_dict:
            if i >= frames_num - 1:
                i = frames_num - 1
            if i in explored_indexes:
                break
            explored_indexes.append(i)

        frame = frames[i]
        index = i
        limit = min(i + max(fps, 1) - 1, frames_num - 1)
        keys = list(bboxes_dict.keys())

        while index < limit:
            index += 1
            if index in keys and bboxes_dict[index] is not None and len(bboxes_dict[index]) > 0:
                break
        if index == limit:
            continue
            
        bboxes_small = bboxes_dict[index]  
        H, W = frame.shape[:2]
        sx = W / 640.0
        sy = H / 480.0

        for bbox in bboxes_small:
            x1, y1, x2, y2 = bbox
            xmin = int(round(x1 * sx))
            ymin = int(round(y1 * sy))
            xmax = int(round(x2 * sx))
            ymax = int(round(y2 * sy))

            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3

            crop_y1 = max(ymin - p_h, 0)
            crop_y2 = min(ymax + p_h, H)
            crop_x1 = max(xmin - p_w, 0)
            crop_x2 = min(xmax + p_w, W)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            ch, cw = crop.shape[:2]
            if ch > cw:
                diff = (ch - cw) // 2
                crop = crop[diff:diff+cw, :]
            elif ch < cw:
                diff = (cw - ch) // 2
                crop = crop[:, diff:diff+ch]

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append((i, Image.fromarray(crop_rgb), [xmin, ymin, xmax, ymax]))

    return crops
# 3. 얼굴 클러스트링링
def cluster_faces(crops, valid_cluster_size_ratio=0.20, similarity_threshold=0.45):
    crops_images = [row[1] for row in crops]
    if not crops_images:
        raise Exception("No face crops available for clustering.")

    embeddings_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    faces = [preprocess_images(face) for face in crops_images]
    faces = np.stack([np.uint8(face) for face in faces])
    faces = torch.as_tensor(faces).permute(0, 3, 1, 2).float()
    faces = fixed_image_standardization(faces)
    embeddings = embeddings_extractor(faces.to(device)).detach().cpu().numpy()

    similarities = np.dot(embeddings, embeddings.T)
    components = _generate_connected_components(similarities, similarity_threshold=similarity_threshold)
    components = [sorted(component) for component in components]

    clustered_faces = {}
    for identity_index, component in enumerate(components):
        component_rows = [crops[idx] for idx in component]
        clustered_faces[identity_index] = component_rows

    return clustered_faces

def get_identity_information(identity, faces):
    mean_side = mean([row[1].size[0] for row in faces])
    number_of_faces = len(faces)
    return [identity, mean_side, number_of_faces, faces]

def get_sorted_identities(identities, discarded_faces, max_identities=2, num_frames=16):
    sorted_identities = []
    discarded_faces = [] 

    for identity in identities:
        sorted_identities.append(get_identity_information(identity, identities[identity]))

    if len(sorted_identities) == 0:
        return sorted_identities, discarded_faces

    sorted_identities = sorted(sorted_identities, key=lambda x: x[1], reverse=True)
    if len(sorted_identities) > max_identities:
        sorted_identities = sorted_identities[:max_identities]

    identities_number = len(sorted_identities)
    available_additional_faces = []

    if identities_number > 1:
        max_faces_per_identity = {
            1: [num_frames],
            2: [num_frames // 2, num_frames // 2],
            3: [num_frames // 3, num_frames // 3, num_frames // 4],
            4: [num_frames // 3, num_frames // 3, num_frames // 8, num_frames // 8]
        }[identities_number]

        for i in range(identities_number):
            faces_now = sorted_identities[i][2]
            if faces_now < max_faces_per_identity[i] and i < identities_number - 1:
                sorted_identities[i+1][2] += max_faces_per_identity[i] - faces_now
                available_additional_faces.append(0)
            elif faces_now > max_faces_per_identity[i]:
                extra = faces_now - max_faces_per_identity[i]
                available_additional_faces.append(extra)
                sorted_identities[i][2] = max_faces_per_identity[i]
            else:
                available_additional_faces.append(0)
    else:
        sorted_identities[0][2] = num_frames
        available_additional_faces.append(0)

    input_len = sum(n for _, _, n, _ in sorted_identities)
    if input_len < num_frames:
        for i in range(identities_number):
            need = num_frames - input_len
            if available_additional_faces[i] > 0:
                add = min(available_additional_faces[i], need)
                sorted_identities[i][2] += add
                input_len += add
                if input_len == num_frames:
                    break

        if input_len < num_frames:
            need = num_frames - input_len
            sorted_identities[-1][2] += need

    return sorted_identities, discarded_faces


# 4. 모델 입력 마스크/시퀀스 생성
def create_val_transform(size, additional_targets):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        Resize(height=size, width=size)
    ], additional_targets=additional_targets)

def generate_masks(video_path, identities, discarded_faces, num_frames, image_size, num_patches):
    mask = []
    sequence = []
    size_embeddings = []
    images_frames = []

    for _, identity in enumerate(identities):
        max_faces = identity[2]
        identity_images = identity[3][:]

        if len(identity_images) > max_faces:
            idx = np.round(np.linspace(0, len(identity_images) - 2, max_faces)).astype(int)
            identity_images = [identity_images[i] for i in idx]

        images_frames.extend(identity_image[0] for identity_image in identity_images)
        identity_images = [identity_image[1] for identity_image in identity_images]

        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_area = (width * height) / 2
        cap.release()

        identity_size_embeddings = []
        for image in identity_images:
            face_area = image.size[0] * image.size[1]
            ratio = int(face_area * 100 / video_area)
            side_ranges = list(map(lambda a_: ratio in range(a_[0], a_[1] + 1), SIZE_EMB_DICT))
            identity_size_embeddings.append(np.where(side_ranges)[0][0] + 1)

        if len(identity_images) < max_faces:
            diff = max_faces - len(identity_size_embeddings)
            identity_size_embeddings = np.concatenate((identity_size_embeddings, np.zeros(diff)))
            identity_images.extend([np.zeros((image_size, image_size, 3), dtype=np.uint8) for _ in range(diff)])
            mask.extend([1 if i < max_faces - diff else 0 for i in range(max_faces)])
            images_frames.extend([max(images_frames) for _ in range(diff)])
        else:
            mask.extend([1 for _ in range(max_faces)])

        size_embeddings.extend(identity_size_embeddings)
        sequence.extend(identity_images)

    sequence = [np.asarray(image) for image in sequence]
    additional_keys = [f"image{i if i > 0 else ''}" for i in range(num_frames)]
    transform = create_val_transform(image_size, {k: "image" for k in additional_keys})
    transformed = transform(**{k: sequence[i] for i, k in enumerate(additional_keys)})
    sequence = [transformed[k] for k in additional_keys]

    identities_mask = []
    position = 0
    for identity in identities:
        faces_count = identity[2]
        for _ in range(faces_count):
            row = [False] * num_frames
            for i in range(faces_count):
                if position + i < num_frames:
                    row[position + i] = True
            identities_mask.append(row)
        position += faces_count

    if len(identities_mask) < num_frames:
        identities_mask.extend([[False]*num_frames for _ in range(num_frames - len(identities_mask))])
    elif len(identities_mask) > num_frames:
        identities_mask = identities_mask[:num_frames]

    images_frames_positions = {k: v+1 for v, k in enumerate(sorted(set(images_frames)))}
    frame_positions = [images_frames_positions[frame] for frame in images_frames]
    positions = []
    for fp in frame_positions:
        start_idx = (fp - 1) * num_patches + 1
        end_idx = start_idx + num_patches
        positions.extend(range(start_idx, end_idx))
    if num_patches is not None:
        positions.insert(0, 0)  

    tokens_per_identity = []
    for i in range(len(identities)):
        if i == 0:
            tokens_per_identity.append((identities[i][0], identities[i][2]*num_patches))
        else:
            tokens_per_identity.append((identities[i][0], identities[i][2]*num_patches + identities[i-1][2]*num_patches))

    return (torch.tensor([sequence]).float(),
            torch.tensor([size_embeddings]).int(),
            torch.tensor([mask]).bool(),
            torch.tensor([identities_mask]).bool(),
            torch.tensor([positions]),
            tokens_per_identity)


# 5. 모델/특징추출기 로드
class TimmFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s_in21k', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')

    def forward(self, x):
        return self.backbone.forward_features(x)

def load_models(opt, config=None, device_override=None):
    dev = device_override if device_override is not None else device

    if config is None:
        with open(opt.config, "r") as f:
            config = yaml.safe_load(f)

    # 1. 특징 추출기(Extractor) 로드
    feat = TimmFeatureExtractor('tf_efficientnetv2_s_in21k', pretrained=True).to(dev).eval()
    if hasattr(opt, 'extractor_weights') and opt.extractor_weights is not None:
        if os.path.exists(opt.extractor_weights):
            print(f"Loading Feature Extractor weights from: {opt.extractor_weights}")
            ext_checkpoint = torch.load(opt.extractor_weights, map_location='cpu')
            
            if next(iter(ext_checkpoint.keys())).startswith("module."):
                ext_checkpoint = {k.replace("module.", "", 1): v for k, v in ext_checkpoint.items()}
                
            # 가중치 적용 
            feat.load_state_dict(ext_checkpoint, strict=False)
            print("Feature Extractor weights loaded successfully.")
        else:
            print(f"Warning: Extractor weights file not found at {opt.extractor_weights}")

    # 2. 메인 모델(TimeSformer) 로드
    mdl = SizeInvariantTimeSformer(config=config, require_attention=True).to(dev).eval()

    if not os.path.exists(opt.model_weights):
        raise Exception(f"TimeSformer weights not found at: {opt.model_weights}")
        
    print(f"Loading TimeSformer weights from: {opt.model_weights}")
    state_dict = torch.load(opt.model_weights, map_location='cpu')

    if next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    mdl.load_state_dict(state_dict, strict=False)
    
    return mdl, feat, config, dev


# 6) 추론
def predict(video_path, crops, config, opt, model=None, features_extractor=None, device_override=None):
    
    dev = device_override if device_override is not None else device
    if model is None or features_extractor is None:
        model, features_extractor, config, dev = load_models(opt, config=config, device_override=dev)

    num_patches = config['model']['num-patches']

    clustered_faces = cluster_faces(crops)
    identities, discarded_faces = get_sorted_identities(clustered_faces, None, num_frames=config['model']['num-frames'])
    
    frames_list = [[face[0] for face in identity[3]] for identity in identities]
    bboxes = [face[2] for identity in identities for face in identity[3]]

    videos, size_embeddings, mask, identities_mask, positions, tokens_per_identity = generate_masks(
        video_path, identities, discarded_faces,
        config["model"]["num-frames"], config["model"]["image-size"], num_patches
    )

    b, f, h, w, c = videos.shape
    videos = videos.to(dev)

    with torch.no_grad():
        video = rearrange(videos, "b f h w c -> (b f) c h w").to(dev)
        features = features_extractor(video)
        features = rearrange(features, '(b f) c h w -> b f c h w', b=b, f=f)
        
        test_pred, attentions = model(
            features, 
            mask=mask.to(dev), 
            size_embedding=size_embeddings.to(dev),
            identities_mask=identities_mask.to(dev), 
            positions=positions.to(dev)
        )

        heatmap_per_frame, attention_scores = aggregate_attentions(
            attentions=attentions,
            num_heads=config['model']['heads'],
            num_frames=config['model']['num-frames'],
            frames_per_identity=[int(row[1] / num_patches) for row in tokens_per_identity]
        )
        
        xai_result_path = None # 초기값

        if identities and len(identities) > 0:
            # 1. 가장 많이 등장한 인물의 얼굴 데이터 가져오기
            faces_data = identities[0][3]
            
            # 히트맵 데이터를 numpy로 변환 
            if isinstance(heatmap_per_frame, torch.Tensor):
                heatmaps_np = heatmap_per_frame.cpu().numpy()
            else:
                heatmaps_np = heatmap_per_frame

            if faces_data and len(faces_data) > 0:
                # 모델이 처리한 프레임들 중에서 가장 Attention 값이 높은 프레임을 찾기
                
                best_idx = 0
                max_intensity = -1.0
                
                # 데이터 개수 불일치 방지를 위해 최소 길이만큼만 반복
                num_check = min(len(faces_data), len(heatmaps_np))
                
                for i in range(num_check):
                    # 해당 프레임 히트맵의 최대값을 구함
                    current_max = np.max(heatmaps_np[i])
                    
                    if current_max > max_intensity:
                        max_intensity = current_max
                        best_idx = i
                
                print(f"-> XAI 분석 결과: {best_idx}번째 프레임이 가장 의심스럽습니다. (강도: {max_intensity:.4f})")

                # 3. 선정된 '베스트 프레임'의 이미지와 히트맵 선택
                best_face_image = faces_data[best_idx][1]
                
                best_heatmap_data = [heatmaps_np[best_idx]] 

                # 4. 이미지 저장
                filename = f"xai_{uuid.uuid4().hex[:8]}.jpg"
                save_full_path = os.path.join("static", "results", filename)
                
                save_explanation_image(best_heatmap_data, best_face_image, save_full_path)
                
                # 웹에서 접근할 경로 저장
                xai_result_path = f"static/results/{filename}"


        return (
            torch.sigmoid(test_pred[0]).item(), 
            attention_scores,                 
            heatmap_per_frame.cpu().numpy(), 
            identities, 
            frames_list,
            xai_result_path  
        )

def get_identities_bboxes(identities):
    identities_bboxes = {}
    for row in identities:
        identity = row[3]
        for face in identity:
            frame = face[0]
            if frame in identities_bboxes:
                identities_bboxes[frame].append(face[2])
            else:
                identities_bboxes[frame] = [face[2]]
    return identities_bboxes

def generate_output_video(video_path, pred, identity_attentions, aggregated_attentions, identities, frames_per_identity):
    identities_bboxes = get_identities_bboxes(identities)
    available_frames_keys = [frame for frame in identities_bboxes]

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 25

    out_path = os.path.join("examples", "preds", os.path.basename(video_path).replace(".mp4", ".avi"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(available_frames_keys) == 0:
            output.write(frame)
            frame_index += 1
            continue

        nearest_frame_index = min(available_frames_keys, key=lambda x: abs(x - frame_index))
        if nearest_frame_index - frame_index > fps:
            output.write(frame)
            frame_index += 1
            continue

        bboxes = identities_bboxes[nearest_frame_index]
        for identity_index, identity_bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = [int(v) for v in identity_bbox]
            if pred > 0.5:
                red = 255 * (identity_attentions[identity_index] if identity_attentions else 1.0)
                green = 255 - red
                text = f"Fake {round(pred*100,2)}%" if red > green else 'Pristine'
            else:
                green = int(255 * (1 - pred))
                red = 255 - green
                text = f"Pristine {round((1-pred)*100,2)}%"
            color = (0, int(green), int(red))
            frame = draw_border(frame, (xmin, ymin), (xmax, ymax), color, 2, 10, 20)
            cv2.putText(frame, text, (xmin, max(0, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        output.write(frame)
        frame_index += 1

    output.release()
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='입력 비디오 경로')
    parser.add_argument("--detector_type", default="FacenetDetector", choices=["FacenetDetector"], help="얼굴 검출기 타입")
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--config', type=str, required=True, help="YAML 설정 파일 경로")
    parser.add_argument('--model_weights', type=str, required=True, help='TimeSformer 모델 체크포인트 경로')
    parser.add_argument('--output_type', default=0, type=int, help='0: 점수 출력, 1: 결과 비디오 생성')
    parser.add_argument('--save_attentions', default=False, action="store_true", help='어텐션 플롯 저장 여부')
    opt = parser.parse_args()
    print(opt)

    # 디바이스 고정
    if torch.cuda.is_available() and opt.gpu_id >= 0:
        dev = torch.device(f"cuda:{opt.gpu_id}")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")

    # 설정 로드
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    num_frames = config['model']['num-frames']
    if num_frames not in [8, 16, 32]:
        raise Exception("Invalid number of frames in config")
    if not os.path.exists(opt.video_path):
        raise Exception("Invalid video path.")

    # 시드 고정
    torch.backends.cudnn.deterministic = True
    random.seed(opt.random_state)
    torch.manual_seed(opt.random_state)
    if dev.type == 'cuda':
        torch.cuda.manual_seed(opt.random_state)
    np.random.seed(opt.random_state)

    # 파이프라인 실행
    print("Detecting faces...")
    bboxes_dict = detect_faces(opt.video_path, opt.detector_type, opt)
    print("Face detection completed.")

    print("Cropping faces from the video...")
    crops = extract_crops(opt.video_path, bboxes_dict)
    print("Faces cropping completed.")

    print("Clustering faces...")
    clustered_faces = cluster_faces(crops)
    print("Faces clustering completed.")

    print("Searching for fakes in the video...")
    pred, identity_attentions, aggregated_attentions, identities, frames_per_identity = predict(
        opt.video_path, crops, config, opt, model=None, features_extractor=None, device_override=dev
    )

    if pred > 0.5:
        print(f"The video is fake ({round(pred*100,2)}%), showing video result...")
    else:
        print(f"The video is pristine ({round((1-pred)*100,2)}%), showing video result...")

    if opt.output_type == 0:
        print("Prediction", pred)
    else:
        generate_output_video(opt.video_path, pred, identity_attentions, aggregated_attentions, identities, frames_per_identity)
