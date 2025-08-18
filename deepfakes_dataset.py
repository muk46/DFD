# DeepFakesDataset class used for data loading
# In this step the identities are also refined and organized in order to fit into the available number of frames per video. 
# The data augmentation is also applied to each face extracted from the video and several embeddings and masks are generated:
# 1. The Size Embedding, responsible to induct the information about face-frame area ratio of each face to the model. 
# 2. The Temporal Positional Embedding, responsible to maintain a coherent spatial and temporal positional information of the input tokens
# 3. The Mask, responsible to make the model ignore the "empty faces" added to fill wholes in the input sequence, if occur
# 4. The Identity Mask, used to tell the model each face to which identity it corresponds

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import random
import numpy as np
from datetime import datetime
import os
import magic
from albumentations import Cutout, CoarseDropout, RandomGamma, MedianBlur, ToSepia, RandomShadow, MultiplicativeNoise, RandomSunFlare, GlassBlur, RandomBrightness, MotionBlur, RandomRain, RGBShift, RandomFog, RandomContrast, Downscale, InvertImg, RandomContrast, ColorJitter, Compose, RandomBrightnessContrast, CLAHE, ISONoise, JpegCompression, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Normalize, Resize
from PIL import Image
from transforms.albu import IsotropicResize
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
import re
import cv2
from itertools import compress
from statistics import mean


ORIGINAL_VIDEOS_PATH = {"train": "../datasets/ForgeryNet/Training/video/train_video_release", "val": "../datasets/ForgeryNet/Training/video/train_video_release", "test": "../datasets/ForgeryNet/Validation/video/val_video_release"}
MODES = ["train", "val", "test"]
RANGE_SIZE = 5
SIZE_EMB_DICT = [(1+i*RANGE_SIZE, (i+1)*RANGE_SIZE) if i != 0 else (0, RANGE_SIZE) for i in range(20)]

class DeepFakesDataset(Dataset):
    def __init__(self, videos_paths, labels, data_path, video_path, image_size, augmentation = None, multiclass_labels = None, save_attention_plots = False, mode = 'train', model = 0, num_frames = 8, max_identities = 3, num_patches=49, enable_identity_attention = True, identities_ordering = 0):
        self.x = videos_paths
        self.y = labels
        self.multiclass_labels = multiclass_labels
        self.save_attention_plots = save_attention_plots
        self.data_path = data_path
        self.video_path = video_path
        self.image_size = image_size
        if mode not in MODES:
            raise Exception("Invalid dataloader mode.")
        self.mode = mode
        self.n_samples = len(videos_paths)
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.max_identities = max_identities
        self.augmentation = augmentation
        self.max_faces_per_identity = {1: [num_frames], 
                                     2:  [int(num_frames/2), int(num_frames/2)],
                                     3:  [int(num_frames/3), int(num_frames/3), int(num_frames/4)],
                                     4:  [int(num_frames/3), int(num_frames/3), int(num_frames/8), int(num_frames/8)]}
        self.enable_identity_attention = enable_identity_attention
        self.identities_ordering = identities_ordering
    
    def create_train_transforms(self, size, additional_targets, augmentation):
        if augmentation == "min":
            return Compose([
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                Resize(height=size, width=size),
                ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
                GaussNoise(p=0.3),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
                ToGray(p=0.2),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            ], additional_targets = additional_targets
            )
        else:
            return Compose([
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                Resize(height=size, width=size),
                ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
                OneOf([GaussianBlur(blur_limit=3), MedianBlur(), GlassBlur(), MotionBlur()], p=0.1),
                OneOf([HorizontalFlip(), InvertImg()], p=0.5),
                OneOf([RandomBrightnessContrast(), RandomContrast(), RandomBrightness(), FancyPCA(), HueSaturationValue()], p=0.5),
                OneOf([RGBShift(), ColorJitter()], p=0.1),
                OneOf([MultiplicativeNoise(), ISONoise(), GaussNoise()], p=0.3),
                OneOf([Cutout(), CoarseDropout()], p=0.1),
                OneOf([RandomFog(), RandomRain(), RandomSunFlare()], p=0.02),
                RandomShadow(p=0.05),
                RandomGamma(p=0.1),
                CLAHE(p=0.05),
                ToGray(p=0.2),
                ToSepia(p=0.05),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            ], additional_targets = additional_targets
            )
            
    def create_val_transform(self, size, additional_targets):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            Resize(height=size, width=size)
        ],  additional_targets = additional_targets
        )

    def get_identity_information(self, identity):
        faces = [os.path.join(identity, face) for face in os.listdir(identity)]
        try:
            mean_side = mean([int(re.search('(\d+) x (\d+)', magic.from_file(face)).groups()[0]) for face in faces])
        except:
            mean_side = 0

        number_of_faces = len(faces)
        return [identity, mean_side, number_of_faces]

    def get_sorted_identities(self, video_path):
        identities = [os.path.join(video_path, identity) for identity in os.listdir(video_path)]
        sorted_identities = []
        discarded_faces = []
        for identity in identities:
            if not os.path.isdir(identity):
                discarded_faces.append(identity)
                continue
            
            sorted_identities.append(self.get_identity_information(identity))
        
        if len(sorted_identities) == 0 and len(discarded_faces) > 0:
            sorted_identities.append(self.get_identity_information(os.path.dirname(discarded_faces[0])))
            discarded_faces = []

        if self.identities_ordering == 0:
            sorted_identities = sorted(sorted_identities, key=lambda x:x[1], reverse=True)
        elif self.identities_ordering == 1:
            sorted_identities = sorted(sorted_identities, key=lambda x:x[2], reverse=True)
        else:
            random.shuffle(sorted_identities)

        if len(sorted_identities) > self.max_identities:
            sorted_identities = sorted_identities[:self.max_identities]
            
        identities_number = len(sorted_identities)
        available_additional_faces = []
        if identities_number > 1:
            max_faces_per_identity = self.max_faces_per_identity[identities_number]
            for i in range(identities_number):
                if sorted_identities[i][2] < max_faces_per_identity[i] and i < identities_number - 1:
                    sorted_identities[i+1][2] += max_faces_per_identity[i] - sorted_identities[i][2] 
                    available_additional_faces.append(0)
                elif sorted_identities[i][2] > max_faces_per_identity[i]:
                    available_additional_faces.append(sorted_identities[i][2] - max_faces_per_identity[i])
                    sorted_identities[i][2] = max_faces_per_identity[i]
                else:
                    available_additional_faces.append(0)
        elif identities_number == 1:
            sorted_identities[0][2] = self.num_frames
            available_additional_faces.append(0)

        input_sequence_length = sum(faces for _, _, faces in sorted_identities)
        if input_sequence_length < self.num_frames:
            for i in range(identities_number):
                needed_faces = self.num_frames - input_sequence_length
                if available_additional_faces[i] > 0:
                    added_faces = min(available_additional_faces[i], needed_faces)
                    sorted_identities[i][2] += added_faces
                    input_sequence_length += added_faces
                    if input_sequence_length == self.num_frames:
                        break
            if input_sequence_length < self.num_frames:
                needed_faces = self.num_frames - input_sequence_length
                sorted_identities[-1][2] += needed_faces
                input_sequence_length += needed_faces
        
        return sorted_identities, discarded_faces
    
    def __getitem__(self, index):
        video_path = self.x[index]
        label = self.y[index]  
        video_id = os.path.basename(video_path)  
        label_str = "FAKE" if self.y[index] == 1 else "REAL"
        video_path = os.path.join(self.data_path, label_str, video_id)
        if self.mode not in video_path:
            for mode in MODES:
                if mode in video_path:
                    self.mode = mode
                    break

        video_id = os.path.basename(video_path)

        identities, discarded_faces = self.get_sorted_identities(video_path)

        mask = []
        sequence = []
        size_embeddings = []
        images_frames = []

        for identity_index, identity in enumerate(identities):
            identity_path = identity[0]
            max_faces = identity[2]
            
            # Check if identity_path is a directory before listing files
            if not os.path.isdir(identity_path):
                identity_faces = []
            else:
                identity_faces = [os.path.join(identity_path, face) for face in os.listdir(identity_path)]

            if identity_index == 0 and len(discarded_faces) > 0:
                frames = [int(os.path.basename(image_path).split("_")[0]) for image_path in identity_faces]
                discarded_frames = [int(os.path.basename(image_path).split("_")[0]) for image_path in discarded_faces]
                missing_frames = list(set(discarded_frames) - set(frames))
                missing_faces = [discarded_faces[discarded_frames.index(missing_frame)] for missing_frame in missing_frames]
                if len(missing_faces) > 0:
                    identity_faces = identity_faces + missing_faces

            if not identity_faces: # If no faces, fill with dummies
                identity_images = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * max_faces
                identity_size_embeddings = [0] * max_faces
                images_frames.extend([0] * max_faces)
            else:
                identity_faces = np.asarray(sorted(identity_faces, key=lambda x: int(os.path.basename(x).split("_")[0])))

                if len(identity_faces) > max_faces:
                    if index % 2:
                        idx = np.round(np.linspace(0, len(identity_faces) - 2, max_faces)).astype(int)
                    else:
                        idx = np.round(np.linspace(1, len(identity_faces) - 1, max_faces)).astype(int)
                    identity_faces = identity_faces[idx]

                identity_images = []
                video_area = 256 * 256
                identity_size_embeddings = []

                for image_index, image_path in enumerate(identity_faces):
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        print(f"Warning: Failed to load image {image_path}. Replacing with a dummy image.")
                        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                        identity_size_embeddings.append(0)
                    else:
                        face_area = image.shape[0] * image.shape[1] / 2
                        ratio = int(face_area * 100 / video_area)
                        side_ranges = list(map(lambda a_: ratio in range(a_[0], a_[1] + 1), SIZE_EMB_DICT))
                        try:
                            embedding = np.where(side_ranges)[0][0] + 1
                        except IndexError:
                            embedding = 1 
                        identity_size_embeddings.append(embedding)

                    frame = int(os.path.basename(image_path).split("_")[0])
                    images_frames.append(frame)
                    identity_images.append(image)

            if len(identity_images) < max_faces:
                diff = max_faces - len(identity_images)
                identity_size_embeddings.extend([0] * diff)
                identity_images.extend([np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * diff)
                try:
                    images_frames.extend([max(images_frames)] * diff)
                except ValueError:
                    images_frames.extend([0] * diff)

            mask.extend([1] * len(identity_images))
            if len(identity_images) < max_faces:
                mask.extend([0] * (max_faces - len(identity_images)))

            size_embeddings.extend(identity_size_embeddings)
            sequence.extend(identity_images)

        # Ensure sequence has exactly num_frames
        if len(sequence) < self.num_frames:
            diff = self.num_frames - len(sequence)
            sequence.extend([np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * diff)
            size_embeddings.extend([0] * diff)
            mask.extend([0] * diff)
        elif len(sequence) > self.num_frames:
            sequence = sequence[:self.num_frames]
            size_embeddings = size_embeddings[:self.num_frames]
            mask = mask[:self.num_frames]

        # Dynamic Augmentation
        images_to_transform = {'image': sequence[0]}
        additional_targets_keys = [f'image{i}' for i in range(1, len(sequence))]
        additional_targets = {key: 'image' for key in additional_targets_keys}
        for i, key in enumerate(additional_targets_keys):
            images_to_transform[key] = sequence[i+1]
        
        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size, additional_targets, self.augmentation)
        else:
            transform = self.create_val_transform(self.image_size, additional_targets)

        transformed_result = transform(**images_to_transform)
        sequence = [transformed_result['image']] + [transformed_result[key] for key in additional_targets_keys]
        
        identities_mask = []
        last_range_end = 0
        for identity_index in range(len(identities)):
            identity_mask = [True if i >= last_range_end and i < last_range_end + identities[identity_index][2] else False for i in range(self.num_frames)]
            for _ in range(identities[identity_index][2]):
                identities_mask.append(identity_mask)
            last_range_end += identities[identity_index][2]

        if not images_frames: images_frames.append(0)
        images_frames_positions = {k: v + 1 for v, k in enumerate(sorted(set(images_frames)))}
        frame_positions = [images_frames_positions.get(frame, 1) for frame in images_frames]

        if self.num_patches is not None:
            positions = [[i + 1 for i in range((frame_position - 1) * self.num_patches, self.num_patches * frame_position)]
                         for frame_position in frame_positions]
            positions = sum(positions, [])
            positions.insert(0, 0)
        else:
            positions = []
        
        tokens_per_identity = []
        if self.save_attention_plots:
            tokens_per_identity = [(os.path.basename(identities[i][0]), identities[i][2] * self.num_patches +
                                    (identities[i - 1][2] * self.num_patches if i > 0 else 0))
                                    for i in range(len(identities))]
        
        # Efficient Tensor Conversion
        sequence_np = np.stack(sequence, axis=0).transpose(0, 3, 1, 2)
        final_tensor = torch.from_numpy(sequence_np).float()
    
        if self.multiclass_labels is None:
            return (torch.from_numpy(sequence_np).float(), torch.tensor(size_embeddings).int(),
                torch.tensor(mask).bool(), torch.tensor(identities_mask).bool(),
                torch.tensor(positions), self.y[index])
        else:
            return (torch.from_numpy(sequence_np).float(), torch.tensor(size_embeddings).int(),
                torch.tensor(mask).bool(), torch.tensor(identities_mask).bool(),
                torch.tensor(positions), tokens_per_identity,
                self.y[index], self.multiclass_labels[index], video_id.replace("/", "_"))

    def __len__(self):
        return self.n_samples