# train.py — EfficientNetV2-S + SizeInvariantTimeSformer (최종 코드)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import argparse
from tqdm import tqdm
import math
import yaml
from utils import check_correct, unix_time_millis, slowfast_input_transform
from datetime import datetime
import collections
import os
from itertools import chain
import random
from einops import rearrange
import pandas as pd
from progress.bar import ChargingBar
from torch.optim import lr_scheduler
from deepfakes_dataset import DeepFakesDataset
from models.size_invariant_timesformer import SizeInvariantTimeSformer
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler
import sys
import timm
import torch.nn as nn
import torch

# --------------------------------
# 유틸 함수
# --------------------------------
def set_seed(seed: int = 42):
    """랜덤 시드 고정 → 재현성 확보"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class TimmFeatureExtractor(nn.Module):
    """timm 모델 래퍼 → EfficientNetV2-S를 로드하여 특성맵 반환"""
    def __init__(self, model_name: str = 'tf_efficientnetv2_s_in21k', pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool=''
        )
    def forward(self, x):
        return self.backbone.forward_features(x)

# --------------------------------
# 메인 실행부
# --------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_list_file', default='split/train_videos.txt', type=str)
    parser.add_argument('--validation_list_file', default='split/val_videos.txt', type=str)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--config', required=True, type=str)
    
    # --- [수정] 불필요한 인자들 제거 ---
    parser.add_argument('--video_path', default='videos', type=str) # deepfakes_dataset.py 호환성을 위해 유지
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--logger_name', default='runs/train')
    parser.add_argument('--models_output_path', default='outputs/models')
    # ------------------------------------

    opt = parser.parse_args()
    print(opt)
    
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() and opt.gpu_id >= 0 else 'cpu')
    print(f"Using device: {device}")

    set_seed(opt.random_state)

    os.makedirs(opt.logger_name, exist_ok=True)
    os.makedirs(opt.models_output_path, exist_ok=True)

    # --- [수정] 특징 추출기 로딩 로직 간소화 ---
    print("Loading feature extractor: EfficientNetV2-S")
    features_extractor = TimmFeatureExtractor(model_name='tf_efficientnetv2_s_in21k', pretrained=True)
    features_extractor = features_extractor.to(device)
    # ---------------------------------------

    # --- [수정] 모델 로딩 로직 간소화 ---
    print("Loading model: SizeInvariantTimeSformer")
    model = SizeInvariantTimeSformer(config=config)
    num_patches = config['model']['num-patches']
    model = model.to(device)
    # ---------------------------------
    
    model.train()
    features_extractor.train()
    
    parameters = chain(features_extractor.parameters(), model.parameters())

    if config['training']['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    elif config['training']['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    else: # 기본값 Adam
        optimizer = torch.optim.Adam(parameters, lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])

    col_names = ["label", "video"]
    df_train = pd.read_csv(opt.train_list_file, sep=' ', names=col_names)
    df_train["label"] = df_train["label"].map({"REAL": 0, "FAKE": 1})
    df_validation = pd.read_csv(opt.validation_list_file, sep=' ', names=col_names)
    df_validation["label"] = df_validation["label"].map({"REAL": 0, "FAKE": 1})
    
    df_train = df_train.sample(frac=1, random_state=opt.random_state).reset_index(drop=True)
    df_validation = df_validation.sample(frac=1, random_state=opt.random_state).reset_index(drop=True)
    
    # 없는 비디오 데이터 제거
    for df in [df_train, df_validation]:
        indexes_to_drop = []
        for index, row in df.iterrows():
            split_dir = "REAL" if row["label"] == 0 else "FAKE"
            video_dir_path = os.path.join(opt.data_path, split_dir, row["video"])
            if not os.path.isdir(video_dir_path) or len(os.listdir(video_dir_path)) == 0:
                indexes_to_drop.append(index)
        df.drop(df.index[indexes_to_drop], inplace=True)
    
    train_videos = df_train['video'].tolist()
    train_labels = df_train['label'].tolist()
    validation_videos = df_validation['video'].tolist()
    validation_labels = df_validation['label'].tolist()
    
    train_samples = len(train_videos)
    validation_samples = len(validation_videos)

    print("Train videos:", train_samples, "Validation videos:", validation_samples)
    print("__TRAINING STATS__")
    train_counters = collections.Counter(train_labels)
    print(train_counters)
    
    class_weights = train_counters.get(0, 1) / max(train_counters.get(1, 1), 1)
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(validation_labels)
    print(val_counters)
    print("___________________")

    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')
    
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).to(device))

    train_dataset = DeepFakesDataset(train_videos, train_labels, augmentation=config['training']['augmentation'], image_size=config['model']['image-size'], data_path=opt.data_path, video_path=opt.video_path, num_frames=config['model']['num-frames'], num_patches=num_patches, max_identities=config['model']['max-identities'])
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, num_workers=opt.workers)

    validation_dataset = DeepFakesDataset(validation_videos, validation_labels, image_size=config['model']['image-size'], data_path=opt.data_path, video_path=opt.video_path, num_frames=config['model']['num-frames'], num_patches=num_patches, max_identities=config['model']['max-identities'], mode='val')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['val_bs'], shuffle=False, num_workers=opt.workers)

    scheduler = None
    if config['training']['scheduler'].lower() == 'steplr':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    elif config['training']['scheduler'].lower() == 'cosinelr':
        num_steps = int(opt.num_epochs * len(train_dl))
        scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=config['training']['lr'] * 1e-1,
                cycle_limit=1,
                t_in_epochs=False,
        )

    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        print(f"Resuming from checkpoint: {opt.resume}")
    else:
        print("No checkpoint loaded for the model.")

    not_improved_loss = 0
    previous_loss = math.inf
    
    for t in range(starting_epoch, opt.num_epochs):
        model.train()
        features_extractor.train()
            
        if not_improved_loss >= opt.patience:
            print(f"Early stopping at epoch {t} due to no improvement in validation loss.")
            break

        total_loss = 0
        train_correct = 0
        
        for data in tqdm(train_dl, desc=f"EPOCH #{t} [TRAIN]"):
            videos, size_embeddings, masks, identities_masks, positions, labels = data
            
            labels = labels.unsqueeze(1).float().to(device)
            videos = videos.to(device)
            
            videos_rearranged = rearrange(videos, "b f c h w -> (b f) c h w")
            
            features = features_extractor(videos_rearranged)
            
            b, f = videos.shape[0], videos.shape[1]
            features = rearrange(features, '(b f) c h w -> b f c h w', b=b, f=f)
            y_pred = model(features, mask=masks.to(device), size_embedding=size_embeddings.to(device), identities_mask=identities_masks.to(device), positions=positions.to(device))
            
            loss = loss_fn(y_pred, labels)
            
            corrects, _, _ = check_correct(y_pred.cpu(), labels.cpu())
            train_correct += corrects
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler is not None and config['training']['scheduler'].lower() == 'cosinelr':
                scheduler.step_update((t * len(train_dl) + index))
        
        total_val_loss = 0
        val_correct = 0
        
        model.eval()
        features_extractor.eval()

        with torch.no_grad():
            for data in tqdm(val_dl, desc=f"EPOCH #{t} [VAL]"):
                videos, size_embeddings, masks, identities_masks, positions, labels = data
                
                labels = labels.unsqueeze(1).float().to(device)
                videos = videos.to(device)

                videos_rearranged = rearrange(videos, 'b f c h w -> (b f) c h w')
                features = features_extractor(videos_rearranged)
                
                b, f = videos.shape[0], videos.shape[1]
                features = rearrange(features, '(b f) c h w -> b f c h w', b=b, f=f)
                val_pred = model(features, mask=masks.to(device), size_embedding=size_embeddings.to(device), identities_mask=identities_masks.to(device), positions=positions.to(device))

                val_loss = loss_fn(val_pred, labels)
                
                total_val_loss += val_loss.item()
                corrects, _, _ = check_correct(val_pred.cpu(), labels.cpu())
                val_correct += corrects
        
        if scheduler is not None and config['training']['scheduler'].lower() == 'steplr':
            scheduler.step()
        
        avg_loss = total_loss / len(train_dl)
        avg_val_loss = total_val_loss / len(val_dl)
        train_accuracy = train_correct / train_samples
        val_accuracy = val_correct / validation_samples

        print(f"\n[Epoch {t:02d}] Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2%} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2%}")

        if avg_val_loss < previous_loss:
            print(f"Validation loss improved from {previous_loss:.4f} to {avg_val_loss:.4f}, saving checkpoint...")
            not_improved_loss = 0
            previous_loss = avg_val_loss
            
            torch.save(features_extractor.state_dict(), os.path.join(opt.models_output_path, f"Extractor_checkpoint{t}.pth"))
            torch.save(model.state_dict(), os.path.join(opt.models_output_path, f"Model_checkpoint{t}.pth"))
        else:
            print(f"Validation loss did not improve from {previous_loss:.4f}")
            not_improved_loss += 1
        
        tb_logger.add_scalar("Training/Accuracy", train_accuracy, t)
        tb_logger.add_scalar("Training/Loss", avg_loss, t)
        tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], t)
        tb_logger.add_scalar("Validation/Loss", avg_val_loss, t)
        tb_logger.add_scalar("Validation/Accuracy", val_accuracy, t)