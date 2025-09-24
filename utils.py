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
    # attentions는 [space_attention, time_attention] 리스트.
    # 마지막 어텐션인 time_attention을 사용.
    # time_attention의 shape는 (num_heads, 1, num_tokens) 형태의 3D 텐서.
    
    # ✨✨✨ 여기가 수정된 핵심 부분! ✨✨✨
    # 3D 텐서에 맞게 인덱싱을 수정하여 "too many indices" 오류를 해결.
    # CLS 토큰(인덱스 0)이 다른 모든 패치 토큰(인덱스 1부터)에 대해 갖는 어텐션 값을 추출.
    attention_scores = attentions[-1][:, 0, 1:]  # (num_heads, num_patches)
    
    # 모든 헤드의 어텐션 점수를 평균내어 최종 어텐션 맵 생성.
    attention_scores = torch.mean(attention_scores, dim=0)  # (num_patches)
    
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
            # 각 identity(얼굴)에 대한 평균 어텐션 값 계산
            identity_attentions.append(identity_attn.mean().item())
        else:
            identity_attentions.append(0.0)
            
        start_frame_idx = end_frame_idx

    aggregated_attentions = attention_scores.cpu().numpy()
    
    # aggregated_attentions: 전체 패치에 대한 1D 어텐션 배열
    # identity_attentions: 각 얼굴(identity)별 평균 어텐션 스코어 리스트
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

    # 저장 경로를 'xai_results'로 분리
    file_path = f"xai_results/tokens/{video_name}_{suffix}.jpg"
    
    # 폴더가 없으면 자동으로 생성
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