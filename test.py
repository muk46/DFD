# 파일명: test.py

import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from einops import rearrange
import pandas as pd

from deepfakes_dataset import DeepFakesDataset
from predict import load_models
from torch.utils.data._utils.collate import default_collate
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]  # None 제거
    if len(batch) == 0:
        return None
    return default_collate(batch)
def main(opt):
    # 1. 설정 및 모델 로드
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    model, features_extractor, _, device = load_models(opt)
    print(f"Model loaded onto {device}")

    # 2. train.py와 동일하게 test_videos.txt 파일을 읽어 리스트를 생성
    col_names = ["label", "video"]
    df_test = pd.read_csv(os.path.join(opt.splits_dir, "test_videos.txt"), sep=' ', names=col_names)
    df_test["label"] = df_test["label"].map({"REAL": 0, "FAKE": 1})

    test_videos = df_test['video'].tolist()
    test_labels = df_test['label'].tolist()
    
    # --- 💡 오류 수정 부분 시작 ---
    # 3. train.py와 완전히 동일한 인자 순서와 방식으로 DeepFakesDataset 호출
    test_dataset = DeepFakesDataset(
        test_videos, # 'videos=' 키워드 제거
        test_labels, # 'labels=' 키워드 제거
        augmentation=None, # 테스트 시에는 증강을 사용하지 않음
        image_size=config['model']['image-size'],
        data_path=opt.data_path,
        video_path=opt.video_path,
        num_frames=config['model']['num-frames'],
        num_patches=config['model']['num-patches'],
        max_identities=config['model']['max-identities'],
        mode='test'
    )
    # --- 💡 오류 수정 부분 끝 ---

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['bs'],
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=collate_skip_none
    )

    # 4. 평가 시작 (이하 동일)
    model.eval()
    features_extractor.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None:  # 빈 배치 skip
                continue

            videos, size_embeddings, masks, identities_masks, positions, labels = batch
            labels = labels.unsqueeze(1).float().to(device)
            videos = videos.to(device)

            videos_rearranged = rearrange(videos, "b f c h w -> (b f) c h w")
            features = features_extractor(videos_rearranged)
            b, f = videos.shape[0], videos.shape[1]
            features = rearrange(features, '(b f) c h w -> b f c h w', b=b, f=f)

            preds, _ = model(
                features,
                mask=masks.to(device),
                size_embedding=size_embeddings.to(device),
                identities_mask=identities_masks.to(device),
                positions=positions.to(device)
            )

            preds_probs = torch.sigmoid(preds).cpu().numpy()
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds_probs.flatten())

    # 5. 최종 성능 지표 계산 및 출력
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]

    accuracy = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    try:
        auc_score = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_score = "N/A"

    print("\n--- Test Results ---")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test AUC Score: {auc_score}")
    print("--------------------")
    # 6. Confusion Matrix 추가
    cm = confusion_matrix(all_labels, binary_preds, labels=[1,0])  # Positive=1(FAKE), Negative=0(REAL)
    print("\nConfusion Matrix:")
    print(cm)

    # 시각화 (기본 형태: [ [TP FN], [FP TN] ])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE (1)', 'REAL (0)'],   # 예측
                yticklabels=['FAKE (1)', 'REAL (0)'])   # 실제
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # 7. 상세 리포트 (Precision, Recall 포함)
    print("\nClassification Report:")
    print(classification_report(all_labels, binary_preds, target_names=['REAL', 'FAKE']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the preprocessed faces_output folder')
    parser.add_argument('--video_path', type=str, default='videos', help='Path to the original videos folder')
    parser.add_argument('--config', type=str, default='./config/size_invariant_timesformer.yaml', help='Model config file path')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the Model_checkpointXX.pth file')
    parser.add_argument('--extractor_weights', type=str, required=True, help='Path to the Extractor_checkpointXX.pth file')
    parser.add_argument('--splits_dir', type=str, default='./splits', help='Folder with train/val/test list files')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # predict.load_models가 요구하는 더미 인자들
    parser.add_argument('--detector_type', type=str, default='FacenetDetector')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--output_type', type=int, default=0)
    parser.add_argument('--save_attentions', action='store_true', default=False)
    parser.add_argument('--extractor_model', type=int, default=2)

    opt = parser.parse_args()
    main(opt)