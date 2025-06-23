import os
import torch
import pathlib
HAS_TORCHCODEC = True
try:
    from torchcodec.decoders import VideoDecoder
except Exception as e:
    HAS_TORCHCODEC = False
from torch.utils.data import Dataset
from data_utils import download_video_and_caption, extract_caption_text, sample_frames
from visual_encoder import get_visual_encoder, preprocess_frames
from mediapipe_encoder import extract_landmarks, create_holistic_model
from optical_flow_encoder import compute_dense_optical_flow

class ASLDataset(Dataset):
    def __init__(self, ids_file, max_frames=256, flow_stride=2, cache_dir="cache/", video_dir="videos/", visual_encoder=None):
        with open(ids_file, "r") as f:
            self.video_ids = [line.strip() for line in f if line.strip()]
        
        self.max_frames = max_frames

        self.cache_dir = cache_dir
        self.video_dir = video_dir

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        
        self.visual_encoder = visual_encoder or get_visual_encoder()
        self.visual_encoder.eval()
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        self.holistic_model = create_holistic_model()

        self.flow_stride = flow_stride

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]

        video_path, vtt_path = download_video_and_caption(vid, output_dir=self.video_dir)

        caption = extract_caption_text(vid, output_dir=self.video_dir)

        if caption is None:
            raise ValueError(f"No caption found for video {vid}")

        # Frame sampling
        frames = sample_frames(video_path, max_frames=self.max_frames)

        # === Process MediaPipe
        mediapipe_cache = os.path.join(self.cache_dir, f"{vid}_mp.pt")
        if os.path.exists(mediapipe_cache):
            mediapipe_feats = torch.load(mediapipe_cache, weights_only=True)
        else:
            landmarks = extract_landmarks(frames, self.holistic_model)
            mediapipe_feats = torch.tensor(landmarks).float()  # (T, D)
            torch.save(mediapipe_feats, mediapipe_cache)

        # === Process Visual
        visual_cache = os.path.join(self.cache_dir, f"{vid}_vis.pt")
        if os.path.exists(visual_cache):
            visual_feats = torch.load(visual_cache, weights_only=True)
        else:
            frame_tensor = preprocess_frames(frames)  # Expected shape: (T, C, H, W)
            
            # Debug print: note use of .shape (not .shape())
            print("frame_tensor shape:", frame_tensor.shape)
            
            with torch.no_grad():
                device = next(self.visual_encoder.parameters()).device
                # Ensure frames are moved to the same device as the model
                batched_frames = frame_tensor.to(dtype=torch.float32, device=device)
                # Vision model expects inputs of shape (B, C, H, W), here B = T (each frame is an image)
                visual_output = self.visual_encoder.vision_model(pixel_values=batched_frames)  # shape: (T, n_patches, D)
                # Extract the [CLS] token for each frame (assumed at index 0)
                visual_feats = visual_output.last_hidden_state[:, 0, :]  # shape: (T, D)
                # If your downstream expects a batch dimension (i.e., (B, T, D) with B=1), add it:
                visual_feats = visual_feats.unsqueeze(0)  # shape: (1, T, D)
                
            torch.save(visual_feats, visual_cache)


        # === Compute (or load) Optical Flow
        flow_cache = os.path.join(self.cache_dir, f"{vid}_flow_stride{self.flow_stride}.pt")
        if os.path.exists(flow_cache):
            flow_feats = torch.load(flow_cache, weights_only=True)
        else:
            flow_feats = compute_dense_optical_flow(frames, stride=self.flow_stride, max_len=self.max_frames)
            flow_feats = torch.tensor(flow_feats).float()  # (T, 2)
            torch.save(flow_feats, flow_cache)

        return {
            "video_id": vid,
            "visual_feats": visual_feats,
            "mediapipe_feats": mediapipe_feats,
            "flow_feats": flow_feats,
            "flow_stride": self.flow_stride,
            "caption": caption
        }