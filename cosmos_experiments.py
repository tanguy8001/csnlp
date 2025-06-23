import torch
from torch.cuda.amp import autocast
from utils.cosmos.cosmos_tokenizer.video_lib import CausalVideoTokenizer
from slt.datasets.data_loading import sample_frames
import imageio
import os

model_name = "Cosmos-1.0-Tokenizer-CV8x8x8"
# Sample frames from the video
frames_np = sample_frames("/work/courses/csnlp/Team3/slt/videos/4eNt91uV02o.mp4", frame_rate=1, max_frames=100)
# Convert numpy arrays (HWC) to PyTorch tensors (CHW)
frames_tensor_chw = [torch.from_numpy(frame).permute(2, 0, 1).to(torch.bfloat16) for frame in frames_np]
# Stack frames to create a tensor of shape (T, C, H, W)
stacked_frames_tchw = torch.stack(frames_tensor_chw)
# Add batch dimension to get (B, T, C, H, W) and move to device
# The model expects [B, C, T, H, W], so we need to permute after stacking TCHW and unsqueezing
input_tensor = stacked_frames_tchw.unsqueeze(0).permute(0, 2, 1, 3, 4).to('cuda')

#input_tensor.shape = [B, C, T, H, W]
encoder = CausalVideoTokenizer(checkpoint_enc=f'utils/cosmos/pretrained_ckpts/{model_name}/encoder.jit')

with autocast():
    # Convert tensor to Float32 before performing padding (reflection pad)
    input_tensor = input_tensor.to(torch.float32)
    (latent,) = encoder.encode(input_tensor)

print(f"Actual latent shape: {latent.shape}")
#torch.testing.assert_close(latent.shape, (1, 16, 3, 64, 64))

decoder = CausalVideoTokenizer(checkpoint_dec=f'utils/cosmos/pretrained_ckpts/{model_name}/decoder.jit')

with autocast():
    reconstructed_tensor = decoder.decode(latent)

# Create output directory if needed
os.makedirs("reconstructed_videos", exist_ok=True)
# Detach, move to CPU, convert to [0,255] uint8
video_tensor = reconstructed_tensor.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
video_np = video_tensor.cpu().clamp(0, 1).mul(255).byte().numpy()

# Save using imageio
output_path = "reconstructed_videos/reconstructed.mp4"
imageio.mimwrite(output_path, video_np, fps=1, quality=8)  # Adjust fps/quality as needed

print(f"Reconstructed video saved at: {output_path}")

torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)
