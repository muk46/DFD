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

# ----------------------------------------
# 전역 디바이스 설정
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def identity_collate_fn(x):
    """DataLoader에서 (비디오 단건) 튜플을 그대로 받기 위한 collate_fn"""
    return x

# 사이즈 임베딩 구간 테이블
RANGE_SIZE = 5
SIZE_EMB_DICT = [(1+i*RANGE_SIZE, (i+1)*RANGE_SIZE) if i != 0 else (0, RANGE_SIZE) for i in range(20)]

# ----------------------------------------
# 1) 얼굴 탐지
# ----------------------------------------
def detect_faces(video_path, detector_cls: Type[VideoFaceDetector], opt):
    """
    비디오에서 YOLOv8-face 기반으로 얼굴 박스를 탐지.
    반환: {프레임인덱스(int): [[x1,y1,x2,y2], ...] 또는 None}
    """
    # 감지기 초기화 (클래스명을 문자열로 넘기는 기존 관례 유지)
    detector = face_detector.__dict__[detector_cls](device=opt.gpu_id)

    # 비디오 로딩 (VideoDataset은 내부에서 640x480로 리사이즈하여 PIL Image 리스트 제공)
    dataset = VideoDataset([video_path])
    loader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1, collate_fn=identity_collate_fn)

    # 얼굴 탐지
    for item in loader:
        bboxes = {}
        video, indices, fps, frames = item[0]
        # indices: 원본 프레임 인덱스 리스트(int), frames: PIL 이미지(640x480)
        detections = detector._detect_faces(frames)  # 각 프레임별 박스 리스트 또는 None
        bboxes.update({i: b for i, b in zip(indices, detections)})

        # 한 프레임이라도 리스트(검출 존재)인지 확인
        found_faces = any(isinstance(bboxes[k], list) and len(bboxes[k]) > 0 for k in bboxes)
        if not found_faces:
            raise Exception("No faces found.")

    return bboxes

# ----------------------------------------
# 2) 얼굴 크롭 추출 (원본 프레임에서 정확한 스케일로)
# ----------------------------------------
def extract_crops(video_path, bboxes_dict):
    """
    YOLO가 본 프레임은 640x480. 원본 프레임 해상도와 다르므로,
    박스를 원본 해상도에 맞게 스케일링 후 크롭을 수행한다.
    반환: [(원본프레임idx, PIL.Image(crop), 원본스케일 bbox), ...]
    """
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

    # 초당 1장 간격으로 샘플링 (fps 간격)
    for i in range(0, len(frames), max(fps, 1)):
        # 해당 초 구간 내에서 탐지된 프레임을 찾는다
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

        # 같은 초 구간에서 탐지된 가장 가까운 프레임 찾기
        while index < limit:
            index += 1
            if index in keys and bboxes_dict[index] is not None and len(bboxes_dict[index]) > 0:
                break
        if index == limit:
            continue

        # 탐지 박스(640x480 기준) → 원본 해상도 스케일 보정
        bboxes_small = bboxes_dict[index]  # xyxy
        H, W = frame.shape[:2]
        sx = W / 640.0
        sy = H / 480.0

        for bbox in bboxes_small:
            # YOLO 출력은 float 좌표. 원본 크기로 변환
            x1, y1, x2, y2 = bbox
            xmin = int(round(x1 * sx))
            ymin = int(round(y1 * sy))
            xmax = int(round(x2 * sx))
            ymax = int(round(y2 * sy))

            # 패딩을 추가하여 배경 일부 포함 (안정성↑)
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3

            crop_y1 = max(ymin - p_h, 0)
            crop_y2 = min(ymax + p_h, H)
            crop_x1 = max(xmin - p_w, 0)
            crop_x2 = min(xmax + p_w, W)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # 정사각형 보정
            ch, cw = crop.shape[:2]
            if ch > cw:
                diff = (ch - cw) // 2
                crop = crop[diff:diff+cw, :]
            elif ch < cw:
                diff = (cw - ch) // 2
                crop = crop[:, diff:diff+ch]

            # BGR → RGB, PIL 변환
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append((i, Image.fromarray(crop_rgb), [xmin, ymin, xmax, ymax]))

    return crops

# ----------------------------------------
# 3) 얼굴 클러스터링 (동일 인물 묶기)
# ----------------------------------------
def cluster_faces(crops, valid_cluster_size_ratio=0.20, similarity_threshold=0.45):
    """
    얼굴 크롭들에 대해 InceptionResnetV1 임베딩 후, 유사도 그래프의
    연결요소를 이용해 동일 인물 클러스터를 생성한다.
    반환: {identity_idx: [(frame_idx, PIL, bbox), ...], ...}
    """
    crops_images = [row[1] for row in crops]
    if not crops_images:
        raise Exception("No face crops available for clustering.")

    # 얼굴 임베딩 추출
    embeddings_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    faces = [preprocess_images(face) for face in crops_images]
    faces = np.stack([np.uint8(face) for face in faces])
    faces = torch.as_tensor(faces).permute(0, 3, 1, 2).float()
    faces = fixed_image_standardization(faces)
    embeddings = embeddings_extractor(faces.to(device)).detach().cpu().numpy()

    # 유사도 행렬 → 연결요소
    similarities = np.dot(embeddings, embeddings.T)
    components = _generate_connected_components(similarities, similarity_threshold=similarity_threshold)
    components = [sorted(component) for component in components]

    clustered_faces = {}
    for identity_index, component in enumerate(components):
        component_rows = [crops[idx] for idx in component]
        clustered_faces[identity_index] = component_rows

    return clustered_faces

def get_identity_information(identity, faces):
    """각 identity에 대해: [identity_id, 평균크롭사이즈, 얼굴개수, 얼굴리스트]"""
    mean_side = mean([row[1].size[0] for row in faces])
    number_of_faces = len(faces)
    return [identity, mean_side, number_of_faces, faces]

def get_sorted_identities(identities, discarded_faces, max_identities=2, num_frames=16):
    """
    인물별 얼굴 개수를 num_frames에 맞추어 재분배.
    가장 큰 얼굴(사이즈) 우선, 부족 시 dummy 채우기.
    """
    sorted_identities = []
    discarded_faces = []  # 현재 사용하지 않음 (인터페이스 유지)

    for identity in identities:
        sorted_identities.append(get_identity_information(identity, identities[identity]))

    if len(sorted_identities) == 0:
        return sorted_identities, discarded_faces

    # 큰 얼굴(사이즈) 우선 정렬
    sorted_identities = sorted(sorted_identities, key=lambda x: x[1], reverse=True)
    if len(sorted_identities) > max_identities:
        sorted_identities = sorted_identities[:max_identities]

    identities_number = len(sorted_identities)
    available_additional_faces = []

    if identities_number > 1:
        # 인물 수에 따른 프레임 분배 규칙
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
        # 인물이 1명인 경우 전체 프레임 할당
        sorted_identities[0][2] = num_frames
        available_additional_faces.append(0)

    # 분배 후 총 프레임 부족 시 앞 인물에서 추가 차출
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
        # 그래도 부족하면 마지막 인물에 dummy 채워 넣기
        if input_len < num_frames:
            need = num_frames - input_len
            sorted_identities[-1][2] += need

    return sorted_identities, discarded_faces

# ----------------------------------------
# 4) 모델 입력 마스크/시퀀스 생성
# ----------------------------------------
def create_val_transform(size, additional_targets):
    """TimeSformer 입력 크기에 맞게 등방성 리사이즈 + 패딩"""
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        Resize(height=size, width=size)
    ], additional_targets=additional_targets)

def generate_masks(video_path, identities, discarded_faces, num_frames, image_size, num_patches):
    """
    아이덴티티별로 선택된 이미지들을 시간 순서대로 나열하고,
    마스크(유효프레임), 아이덴티티 마스크, 토큰 포지션 등을 생성.
    """
    mask = []
    sequence = []
    size_embeddings = []
    images_frames = []

    for _, identity in enumerate(identities):
        max_faces = identity[2]
        identity_images = identity[3][:]

        # 필요 개수만 균등 샘플링
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

        # 부족 시 dummy 이미지/마스크 0
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

    # albumentations 입력 준비
    sequence = [np.asarray(image) for image in sequence]
    additional_keys = [f"image{i if i > 0 else ''}" for i in range(num_frames)]
    transform = create_val_transform(image_size, {k: "image" for k in additional_keys})
    transformed = transform(**{k: sequence[i] for i, k in enumerate(additional_keys)})
    sequence = [transformed[k] for k in additional_keys]

    # 아이덴티티 마스크 (time × time)
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

    # 포지션 인덱스 (프레임별 패치 시작~끝 범위)
    images_frames_positions = {k: v+1 for v, k in enumerate(sorted(set(images_frames)))}
    frame_positions = [images_frames_positions[frame] for frame in images_frames]
    positions = []
    for fp in frame_positions:
        start_idx = (fp - 1) * num_patches + 1
        end_idx = start_idx + num_patches
        positions.extend(range(start_idx, end_idx))
    if num_patches is not None:
        positions.insert(0, 0)  # [CLS] 토큰 등용

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

# ----------------------------------------
# 5) 모델/특징추출기 로드 (1회)
# ----------------------------------------
class TimmFeatureExtractor(torch.nn.Module):
    """timm EfficientNetV2-S를 (B,C,H,W)→(B,C',H',W') 특성맵으로 반환하도록 래핑"""
    def __init__(self, model_name='tf_efficientnetv2_s_in21k', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')

    def forward(self, x):
        return self.backbone.forward_features(x)

def load_models(opt, config=None, device_override=None):
    """
    모델과 특징추출기를 로드하여 (모델, 특징추출기, config, device) 반환.
    - Flask 서버 시작 시 1회 호출해 전역 재사용
    - 단독 실행 시에도 사용 가능
    """
    dev = device_override if device_override is not None else device

    # config 인자 없으면 파일에서 로드
    if config is None:
        with open(opt.config, "r") as f:
            config = yaml.safe_load(f)

    # 특징추출기: EfficientNetV2-S (timm)
    feat = TimmFeatureExtractor('tf_efficientnetv2_s_in21k', pretrained=True).to(dev).eval()

    # 메인 모델: SizeInvariantTimeSformer
    mdl = SizeInvariantTimeSformer(config=config, require_attention=True).to(dev).eval()

    # 체크포인트 로드
    if not os.path.exists(opt.model_weights):
        raise Exception("No checkpoint loaded for the model.")
    state_dict = torch.load(opt.model_weights, map_location='cpu')

    # DataParallel 키 보정 (현재 기본은 단일 GPU 가정)
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        # 현재 mdl이 DP가 아니라면 'module.' 제거
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    mdl.load_state_dict(state_dict, strict=False)
    return mdl, feat, config, dev

# ----------------------------------------
# 6) 추론
# ----------------------------------------
def predict(video_path, crops, config, opt, model=None, features_extractor=None, device_override=None):
    """
    추론 진입점.
    - app.py에서는 (사전로드된) model/features_extractor/device를 전달
    - 단독 실행 시 None → 내부에서 load_models() 호출
    반환: (pred_score, identity_attns, aggregated_attns, identities, frames_per_identity)
    """
    dev = device_override if device_override is not None else device
    if model is None or features_extractor is None:
        model, features_extractor, config, dev = load_models(opt, config=config, device_override=dev)

    num_patches = config['model']['num-patches']

    # 얼굴 클러스터링 → 아이덴티티 정렬
    clustered_faces = cluster_faces(crops)
    identities, discarded_faces = get_sorted_identities(clustered_faces, None, num_frames=config['model']['num-frames'])

    # 마스크/시퀀스 생성
    videos, size_embeddings, mask, identities_mask, positions, tokens_per_identity = generate_masks(
        video_path, identities, discarded_faces,
        config["model"]["num-frames"], config["model"]["image-size"], num_patches
    )

    b, f, h, w, c = videos.shape
    videos = videos.to(dev)
    identities_mask = identities_mask.to(dev)
    mask = mask.to(dev)
    positions = positions.to(dev)

    with torch.no_grad():
        # (B,F,H,W,C) → (B*F,C,H,W)
        video = rearrange(videos, "b f h w c -> (b f) c h w").to(dev)
        features = features_extractor(video)                               # (B*F, C', H', W')
        features = rearrange(features, '(b f) c h w -> b f c h w', b=b, f=f)
        test_pred, attentions = model(
            features, mask=mask, size_embedding=size_embeddings,
            identities_mask=identities_mask, positions=positions
        )

        identity_names = [row[0] for row in tokens_per_identity]
        frames_per_identity = [int(row[1] / num_patches) for row in tokens_per_identity]

        if opt.save_attentions:
            aggregated_attentions, identity_attentions = aggregate_attentions(
                attentions, config['model']['heads'], config['model']['num-frames'], frames_per_identity
            )
            save_attention_plots(
                aggregated_attentions, identity_names, frames_per_identity,
                config['model']['num-frames'], os.path.basename(video_path)
            )
        else:
            identity_attentions = []
            aggregated_attentions = []

        return torch.sigmoid(test_pred[0]).item(), identity_attentions, aggregated_attentions, identities, frames_per_identity

# ----------------------------------------
# 7) 결과 비디오 생성 (옵션)
# ----------------------------------------
def get_identities_bboxes(identities):
    """아이덴티티별 프레임-박스 매핑 생성"""
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
    """탐지 결과를 원본 영상에 오버레이하여 저장"""
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

# ----------------------------------------
# 8) 스크립트 단독 실행 (디버그/테스트용)
# ----------------------------------------
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
