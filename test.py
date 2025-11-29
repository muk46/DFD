import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
from einops import rearrange
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data._utils.collate import default_collate

from deepfakes_dataset import DeepFakesDataset
from predict import load_models

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def main(opt):
    # 설정 파일 로드
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # 모델 로드
    model, features_extractor, _, device = load_models(opt)
    print(f"Model loaded onto {device}")

    # 테스트 데이터 리스트 로드
    col_names = ["label", "video"]
    df_test = pd.read_csv(os.path.join(opt.splits_dir, "test_videos.txt"), sep=' ', names=col_names)
    df_test["label"] = df_test["label"].map({"REAL": 0, "FAKE": 1})

    test_videos = df_test['video'].tolist()
    test_labels = df_test['label'].tolist()
    
    # 데이터셋 초기화
    test_dataset = DeepFakesDataset(
        test_videos,
        test_labels,
        augmentation=None,
        image_size=config['model']['image-size'],
        data_path=opt.data_path,
        video_path=opt.video_path,
        num_frames=config['model']['num-frames'],
        num_patches=config['model']['num-patches'],
        max_identities=config['model']['max-identities'],
        mode='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['bs'],
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=collate_skip_none
    )

    # 추론 시작
    model.eval()
    features_extractor.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None:
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

    binary_preds = [1 if p > 0.97 else 0 for p in all_preds]

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

    cm = confusion_matrix(all_labels, binary_preds, labels=[1, 0]) # 1: FAKE, 0: REAL
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE (1)', 'REAL (0)'],
                yticklabels=['FAKE (1)', 'REAL (0)'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, binary_preds, target_names=['REAL', 'FAKE']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the preprocessed faces_output folder')
    parser.add_argument('--video_path', type=str, default='videos', help='Path to the original videos folder')
    parser.add_argument('--config', type=str, default='./config/size_invariant_timesformer.yaml', help='Model config file path')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the Model_checkpointXX.pth file')
    parser.add_argument('--extractor_weights', required=False, default=None, type=str)
    parser.add_argument('--splits_dir', type=str, default='./splits', help='Folder with train/val/test list files')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # 모델 로드에 필요한 추가 인자
    parser.add_argument('--detector_type', type=str, default='FacenetDetector')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--output_type', type=int, default=0)
    parser.add_argument('--save_attentions', action='store_true', default=False)
    parser.add_argument('--extractor_model', type=int, default=2)

    opt = parser.parse_args()
    main(opt)
