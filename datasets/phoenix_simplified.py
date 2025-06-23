import os
import glob
import torch
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torch.nn.utils.rnn import pad_sequence
from typing import Callable, List, Dict, Any, Optional
from PIL import Image

from .phoenix_preprocessing import clean_phoenix_2014_trans
from visual_encoder import get_visual_encoder
from mediapipe_encoder import create_holistic_model
from .asl_dataset import compute_dense_optical_flow, extract_landmarks


# Atomic save function to prevent corruption during caching
def atomic_save(obj: Any, path: Path):
    """Write a .pt file atomically to avoid corruption."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class PhoenixSimplified(Dataset):
    """
    Simplified Phoenix dataset that returns a dict with keys:
        • motion    : (N, Dm) - optical flow features
        • spatial   : (T, Ds) - visual encoder features  
        • landmarks : (T, D_lmk) - MediaPipe landmarks or None if missing
        • sentence  : str - German translation text
        • video_id  : str - video identifier
    """
    
    def __init__(
        self, 
        root: str, 
        split: str = "train",
        max_frames: Optional[int] = None,
        flow_stride: int = 2,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.max_frames = max_frames
        self.flow_stride = flow_stride
        
        # Use the same cache structure as original Phoenix dataset
        self.main_cache_dir = self.root / "cache"
        self.main_cache_dir.mkdir(parents=True, exist_ok=True)
        self.videos_specific_cache_dir = self.main_cache_dir / self.split

        # Load CSV annotations
        csv_path = self.root / "annotations" / "manual" / f"PHOENIX-2014-T.{split}.corpus.csv"
        #csv_path = self.root / "annotations" / "manual" / f"{split}.corpus.csv"
        self.df = pd.read_csv(csv_path, delimiter="|")
        
        # Rename columns for consistency
        if 'name' in self.df.columns:
            self.df.rename(columns={'name': 'id'}, inplace=True)
        if 'speaker' in self.df.columns:
            self.df.rename(columns={'speaker': 'signer'}, inplace=True)

        # Path to PNG frames
        self.png_root = self.root / "features" / split

        self.visual_encoder = get_visual_encoder()
        self.holistic_model = create_holistic_model()
        
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.53724027, 0.5272855, 0.51954997], [1, 1, 1]),
        ])
        
        # Filter valid samples
        self._filter_valid_samples()
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.head(max_samples)
            print(f"Limited dataset to {max_samples} samples.")

    def _filter_valid_samples(self):
        """Remove samples without frames"""
        valid_indices = []
        for i in range(len(self.df)):
            video_id = self.df.iloc[i]['id']
            frame_paths = self._get_frame_paths(video_id)
            if frame_paths:
                valid_indices.append(i)
        
        if len(valid_indices) < len(self.df):
            print(f"Filtered out {len(self.df) - len(valid_indices)} samples without frames")
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def _get_frame_paths(self, video_id: str) -> List[Path]:
        """Get sorted frame paths for a video"""
        frame_pattern = self.png_root / video_id / "*.png"
        return sorted(glob.glob(str(frame_pattern)))

    def _get_cache_paths(self, video_id: str):
        """Get cache file paths for motion, spatial, and landmark features (same as original)"""
        video_specific_cache_dir = self.videos_specific_cache_dir / video_id
        video_specific_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the same filenames as the original Phoenix dataset
        motion_path = video_specific_cache_dir / f"flow_stride{self.flow_stride}_max_frames{self.max_frames}.pt"
        spatial_path = video_specific_cache_dir / f"visual_feats_max_frames{self.max_frames}.pt"
        landmark_path = video_specific_cache_dir / f"skeleton_feats_max_frames{self.max_frames}.pt"
        return motion_path, spatial_path, landmark_path

    def _compute_and_cache_features(self, video_id: str, frame_paths: List[str]):
        """Compute and cache all features for a video"""
        motion_path, spatial_path, landmark_path = self._get_cache_paths(video_id)
        
        images_pil = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            images_pil.append(img)
        
        # Sample frames if max_frames is set
        if self.max_frames is not None and len(images_pil) > self.max_frames:
            indices = np.linspace(0, len(images_pil) - 1, self.max_frames, dtype=int)
            images_pil = [images_pil[i] for i in indices]
        
        # Convert to tensors for visual encoder
        images_tensor = torch.stack([self.transform(img) for img in images_pil])
        
        # Convert to numpy for motion and landmarks
        images_np = [np.array(img) for img in images_pil]
        
        # Compute spatial features
        with torch.no_grad():
            device = next(self.visual_encoder.parameters()).device
            images_tensor = images_tensor.to(device, dtype=torch.float32)
            visual_output = self.visual_encoder.vision_model(pixel_values=images_tensor)
            spatial_feats = visual_output.last_hidden_state[:, 0, :].cpu()  # (T, D)
        
        # Compute motion features
        motion_feats = compute_dense_optical_flow(
            images_np, 
            stride=self.flow_stride, 
            max_len=self.max_frames
        )
        motion_feats = torch.tensor(motion_feats).float()
        
        # Compute landmark features
        landmarks = extract_landmarks(images_np, self.holistic_model)
        landmark_feats = torch.tensor(landmarks).float()
        
        # Cache features atomically
        atomic_save(motion_feats, motion_path)
        atomic_save(spatial_feats, spatial_path)
        atomic_save(landmark_feats, landmark_path)
        
        return motion_feats, spatial_feats, landmark_feats

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        video_id = row["id"]
        sentence = row.get("translation", "")
        
        # Clean the sentence
        if pd.isna(sentence):
            sentence = ""
        sentence = clean_phoenix_2014_trans(sentence).lower()
        
        motion_path, spatial_path, landmark_path = self._get_cache_paths(video_id)
        
        # Load or compute features
        if motion_path.exists() and spatial_path.exists() and landmark_path.exists():
            motion = torch.load(motion_path, map_location="cpu")
            spatial = torch.load(spatial_path, map_location="cpu")
            landmarks = torch.load(landmark_path, map_location="cpu")
        else:
            frame_paths = self._get_frame_paths(video_id)
            if not frame_paths:
                raise FileNotFoundError(f"No frames found for video {video_id}")
            motion, spatial, landmarks = self._compute_and_cache_features(video_id, frame_paths)
        
        return {
            "video_id": video_id,
            "motion": motion,
            "spatial": spatial, 
            "landmarks": landmarks,
            "sentence": sentence,
        }

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    """Collate function for DataLoader"""
    motions = [b["motion"] for b in batch]        # List of (Ni, Dm)
    spatials = [b["spatial"] for b in batch]      # List of (Ti, Ds)
    landmarks = [b["landmarks"] for b in batch]   # List of (Ti, 129) or None

    # Pad motion and spatial features
    motion_padded = pad_sequence(motions, batch_first=True)   # (B, N_max, Dm)
    spatial_padded = pad_sequence(spatials, batch_first=True) # (B, T_max, Ds)

    # Handle landmarks (convert None to empty tensor with correct dims)
    D_LMK = 129  # Standard landmark dimension
    landmark_fixed = []
    for l in landmarks:
        if l is not None:
            landmark_fixed.append(l)
        else:
            landmark_fixed.append(torch.empty(0, D_LMK))
    
    landmark_padded = pad_sequence(landmark_fixed, batch_first=True)  # (B, L_max, 129)

    return {
        "video_id": [b["video_id"] for b in batch],
        "motion": motion_padded,
        "spatial": spatial_padded,
        "landmarks": landmark_padded,
        "tgt": [b["sentence"] for b in batch],  # Target sentences
    } 