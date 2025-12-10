# Utility functions for training process

import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from random import random
from scipy.special import softmax
from einops import rearrange
from statistics import mean
import cv2
import math
from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     ShortSideScale,
#     UniformTemporalSubsample,
#     UniformCropVideo
# ) 

PLOTS_NAMES = ["space", "time", "combined"]


# Convert the preds into final video-level prediction
def check_correct(preds, labels, multiclass_labels = None, multiclass_errors = None, videos_ids = None):
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]

    correct = 0
    positive_class = 0
    negative_class = 0
    videos_errors = []
    for i in range(len(labels)):
        pred = int(preds[i])
        if labels[i] == pred:
            correct += 1
        if labels[i] != pred:
            if multiclass_labels is not None and not math.isnan(multiclass_labels[i]):
                multiclass_errors[multiclass_labels[i].item()][0] += 1
            if videos_ids != None:
                videos_errors.append(videos_ids[i])
            
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1

    if multiclass_errors != None:
        return correct, positive_class, negative_class, multiclass_errors, videos_errors
    else:
        return correct, positive_class, negative_class


def unix_time_millis(dt):
    return dt.total_seconds() * 1000.0


def multiple_lists_mean(a):
    return sum(a) / len(a)

# Aggregate space and time attention 
def aggregate_attentions(attentions, num_heads, num_frames, frames_per_identity):
    """Aggregates the attentions from the different heads and layers."""
    
    # 1. 리스트 입력 처리 (Layer별 Attention들을 하나로 쌓기)
    if isinstance(attentions, (list, tuple)):
        attentions = torch.stack(attentions)

    # 2. [수정 핵심] 입력 변수 'attentions'를 사용해서 평균을 구해야 함 (이전 코드 에러 원인 해결)
    attention_scores = torch.mean(attentions, dim=0)

    # 3. 1차원(Tokens)으로 차원 축소
    if attention_scores.dim() > 1:
        attention_scores = torch.mean(attention_scores, dim=tuple(range(attention_scores.dim() - 1)))
    
    # 4. CLS 토큰 제거 (393 -> 392 에러 방지)
    if attention_scores.shape[0] % num_frames != 0:
        attention_scores = attention_scores[1:]

    num_patches_per_frame = attention_scores.shape[0] // num_frames

    w = h = int(np.sqrt(num_patches_per_frame))
    heatmap_per_frame = rearrange(attention_scores, '(f h w) -> f h w', f=num_frames, h=h, w=w)
    
    identity_attentions = []
    start_frame_idx = 0
    
    if isinstance(frames_per_identity, int):
        frames_per_identity = [frames_per_identity]

    for n_faces in frames_per_identity:
        end_frame_idx = start_frame_idx + n_faces
        
        start_patch_idx = start_frame_idx * num_patches_per_frame
        end_patch_idx = end_frame_idx * num_patches_per_frame
        
        identity_attn = attention_scores[start_patch_idx:end_patch_idx]
        
        if identity_attn.numel() > 0:
            identity_attentions.append(identity_attn.mean().detach().cpu().numpy())
        else:
            identity_attentions.append(np.array(0.0, dtype=np.float32))
            
        start_frame_idx = end_frame_idx

    return heatmap_per_frame, identity_attentions



# Visualize the attention
def save_attention_plots(attention, identity_names, frames_per_identity, num_frames, video_name, suffix=""):
    """Saves the attention plots to a file in a non-conflicting directory."""
    if attention is None or len(attention) == 0:
        return

    plt.figure(figsize=(num_frames, len(identity_names) * 2))
    
    for i, (name, frames) in enumerate(zip(identity_names, frames_per_identity)):
        ax = plt.subplot(len(identity_names), 1, i + 1)
        # Assuming attention is a 1D array, reshape for imshow
        # This part might need adjustment based on the exact shape of 'attention'
        im = ax.imshow(np.array(attention).reshape(1, -1), cmap="seismic", vmin=-1, vmax=1)
        ax.set_title(f"Identity: {name}")
        ax.set_yticks([])
        ax.set_xticks(list(range(num_frames)))
        ax.set_xticklabels(list(range(1, num_frames + 1)))

    file_path = f"xai_results/tokens/{video_name}_{suffix}.jpg"
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    plt.savefig(file_path)
    plt.close()


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    return img

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

SLOWFAST_ALPHA = 4

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // SLOWFAST_ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def slowfast_input_transform(videos, crop_size = 256, side_size = 256, num_frames = 32, sampling_rate = 2, frames_per_second = 30, mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225]):
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    )
    transformed_videos = [[],[]]
    for video in videos:
        output = transform(video)
        transformed_videos[0].append(output[0])
        transformed_videos[1].append(output[1])
    
        
    return transformed_videos

def save_explanation_image(heatmap_data, original_image, output_path):
    """
    3단계 최종: 히트맵을 원본과 겹쳐서 지정된 경로(output_path)에 저장합니다.
    """
    import cv2
    import numpy as np
    import os

    # 1. 첫 번째 프레임 히트맵 추출 & 정규화
    first_frame_map = heatmap_data[0] 
    norm_map = cv2.normalize(first_frame_map, None, 0, 255, cv2.NORM_MINMAX)
    norm_map = np.uint8(norm_map)

    # 2. 원본 이미지 처리
    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image)
    
    # RGB -> BGR 변환 (OpenCV 호환)
    if original_image.shape[-1] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    h, w = original_image.shape[:2]

    # 3. 히트맵 확대 및 컬러 입히기
    resized_map = cv2.resize(norm_map, (w, h), interpolation=cv2.INTER_CUBIC)
    color_map = cv2.applyColorMap(resized_map, cv2.COLORMAP_JET)

    # 4. 겹치기 (Overlay)
    overlay = cv2.addWeighted(original_image, 0.6, color_map, 0.4, 0)

    # 5. 지정된 경로에 저장
    # 폴더가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    
    return output_path
