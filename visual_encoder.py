# === In this class we load a pretrained visual encoder 
import torch
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel

# Example with CLIP
def get_visual_encoder():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def preprocess_frames(frames):  # frames = list of np arrays
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return torch.stack([transform(f) for f in frames])  # (T, C, H, W)
