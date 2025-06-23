import os
import torch

def save_pipeline_state(
    merger_model,
    fusion_model,
    save_dir="checkpoints",
    prefix=""
):
    os.makedirs(save_dir, exist_ok=True)

    torch.save(merger_model.state_dict(), os.path.join(save_dir, f"{prefix}merger_model.pt"))
    torch.save(fusion_model.state_dict(), os.path.join(save_dir, f"{prefix}fusion_model.pt"))

    print(f"✅ Saved merger_model and fusion model to {save_dir}/")

def load_pipeline_state(merger_model, fusion_model, save_dir="checkpoints", prefix=""):
    merger_model.load_state_dict(torch.load(os.path.join(save_dir, f"{prefix}merger_model.pt")))
    fusion_model.load_state_dict(torch.load(os.path.join(save_dir, f"{prefix}fusion_model.pt")))

    print(f"✅ Loaded merger_model and fusion model from {save_dir}/")
