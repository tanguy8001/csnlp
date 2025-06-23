"""
Training script for SignVLM variants
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
# import wandb  # Removed for submission
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from sign_vlm import SignVLMDynamic, SignVLMPooling, SignVLMAdapter, SignVLMDynamicNoVQ, SignVLMSimple, SignVLMLandmark, SignVLMT5Style
from datasets.phoenix_simplified import PhoenixSimplified, collate_fn


# === Configuration ===
APPROACH = "t5_style"
NUM_POOLED_TOKENS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
MAX_FRAMES = 256
VISUAL_DIM = 768
LANDMARK_DIM = 129
MOTION_DIM = 2
D_MODEL = 768
NUM_FUSION_LAYERS = 2
NUM_QUERY_TOKENS = 24
FLOW_STRIDE = 2
MAX_SAMPLES = None

# Resume Training Configuration
RESUME_FROM_CHECKPOINT = True  # Set to True to resume from checkpoint
CHECKPOINT_PATH = "best_signvlm_fixed_checkpoint.pt"  # Path to checkpoint file

# Model Configuration
LLM_NAME = "meta-llama/Llama-3.2-1B"
DATA_PATH = "datasets/Phoenix14T"
NUM_EPOCHS = 200
VAL_SPLIT_RATIO = 0.1
#LEARNING_RATE = 5e-5  # Lower learning rate for stability
LEARNING_RATE = 1e-5  # Much lower LR for full model training
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# Logging
WANDB_PROJECT_NAME = "signvlm-fixed-phoenix-translation"
WANDB_ENTITY = None
LOG_EVERY = 10
EVAL_EVERY = 2


def create_learning_rate_scheduler(optimizer, warmup_steps, total_steps):
    """Create a learning rate scheduler with warmup"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0.1, (total_steps - step) / (total_steps - warmup_steps))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load model and training state from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return 0, 0.0, 0  # start_epoch, best_rouge, global_step
    
    try:
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model state loaded")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state loaded")
        
        # Load training progress
        start_epoch = checkpoint.get('epoch', 0)
        best_rouge = checkpoint.get('best_rouge', 0.0)
        global_step = checkpoint.get('global_step', 0)
        
        print(f"✅ Resuming from epoch {start_epoch + 1}, best ROUGE-L: {best_rouge:.4f}")
        
        # Note: Scheduler state will be rebuilt based on global_step
        
        return start_epoch + 1, best_rouge, global_step
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("🔄 Starting training from scratch")
        return 0, 0.0, 0


def evaluate_model(model, val_dataloader, device):
    """Evaluate the model on validation set"""
    model.eval()
    
    total_bleu1, total_bleu2, total_bleu3, total_bleu4 = 0, 0, 0, 0
    total_rouge_l = 0
    num_samples = 0
    total_loss = 0
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothing_function = SmoothingFunction().method1
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating")):
            visual_feats = batch["spatial"].to(device)      # (B, T_spatial, D_spatial)
            motion_feats = batch["motion"].to(device)        # (B, T_motion, D_motion)
            landmark_feats = batch["landmarks"].to(device)   # (B, T_landmarks, D_landmarks)
            target_sentences = batch["tgt"]                  # List of strings
            
            # Compute loss
            try:
                loss = model.compute_loss(visual_feats, motion_feats, landmark_feats, target_sentences)
                total_loss += loss.item()
            except Exception as e:
                print(f"Loss computation failed: {e}")
                continue
            
            # Generate translations
            try:
                generated_texts = model.generate(visual_feats, motion_feats, landmark_feats, max_new_tokens=50)
            except Exception as e:
                print(f"Generation failed: {e}")
                generated_texts = [""] * len(target_sentences)
            
            # Compute metrics
            for ref_text, gen_text in zip(target_sentences, generated_texts):
                ref_tokens = ref_text.lower().split()
                gen_tokens = gen_text.lower().split()
                
                if not gen_tokens:
                    gen_tokens = [""]
                
                # BLEU scores
                total_bleu1 += sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
                total_bleu2 += sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
                total_bleu3 += sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
                total_bleu4 += sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
                
                # ROUGE-L
                rouge_scores = rouge_scorer_obj.score(ref_text, gen_text)
                total_rouge_l += rouge_scores['rougeL'].fmeasure
                num_samples += 1
            
            # Print examples
            if batch_idx == 0:
                print("\n=== Evaluation Examples ===")
                for i in range(min(2, len(target_sentences))):
                    print(f"Reference: {target_sentences[i]}")
                    print(f"Generated: {generated_texts[i]}")
                    print()
    
    # Compute averages
    avg_loss = total_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
    avg_bleu1 = total_bleu1 / num_samples if num_samples > 0 else 0
    avg_bleu2 = total_bleu2 / num_samples if num_samples > 0 else 0
    avg_bleu3 = total_bleu3 / num_samples if num_samples > 0 else 0
    avg_bleu4 = total_bleu4 / num_samples if num_samples > 0 else 0
    avg_rouge_l = total_rouge_l / num_samples if num_samples > 0 else 0
    
    return {
        "val_loss": avg_loss,
        "BLEU-1": avg_bleu1,
        "BLEU-2": avg_bleu2, 
        "BLEU-3": avg_bleu3,
        "BLEU-4": avg_bleu4,
        "ROUGE-L": avg_rouge_l
    }

#TEST
def evaluate_model_test(model, val_dataloader, device):
    """Evaluate the model on the validation set and compute metrics."""
    model.eval()

    metrics = {
        "bleu1": [],
        "bleu2": [],
        "bleu3": [],
        "bleu4": [],
        "rougeL": [],
        "losses": []
    }

    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothing_function = SmoothingFunction().method1

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating")):
            visual_feats = batch["spatial"].to(device)
            motion_feats = batch["motion"].to(device)
            landmark_feats = batch["landmarks"].to(device)
            target_sentences = batch["tgt"]

            try:
                loss = model.compute_loss(visual_feats, motion_feats, landmark_feats, target_sentences)
                metrics["losses"].append(loss.item())
            except Exception as e:
                print(f"[Batch {batch_idx}] Loss computation failed: {e}")
                continue

            try:
                generated_texts = model.generate(
                    visual_feats, motion_feats, landmark_feats, max_new_tokens=50
                )
            except Exception as e:
                print(f"[Batch {batch_idx}] Generation failed: {e}")
                generated_texts = [""] * len(target_sentences)

            for ref_text, gen_text in zip(target_sentences, generated_texts):
                ref_tokens = ref_text.lower().split()
                gen_tokens = gen_text.lower().split() or [""]

                # BLEU scores
                metrics["bleu1"].append(sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function))
                metrics["bleu2"].append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function))
                metrics["bleu3"].append(sentence_bleu([ref_tokens], gen_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing_function))
                metrics["bleu4"].append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function))

                # ROUGE-L
                rouge_scores = rouge_scorer_obj.score(ref_text, gen_text)
                metrics["rougeL"].append(rouge_scores['rougeL'].fmeasure)

            # Print qualitative examples for the first batch
            if batch_idx == 0:
                print("\n=== Evaluation Examples ===")
                for ref, gen in zip(target_sentences[:2], generated_texts[:2]):
                    print(f"  Reference: {ref}")
                    print(f"  Generated: {gen}\n")

    # Average metrics
    avg_metrics = {
        "val_loss": np.mean(metrics["losses"]) if metrics["losses"] else 0.0,
        "BLEU-1": np.mean(metrics["bleu1"]) if metrics["bleu1"] else 0.0,
        "BLEU-2": np.mean(metrics["bleu2"]) if metrics["bleu2"] else 0.0,
        "BLEU-3": np.mean(metrics["bleu3"]) if metrics["bleu3"] else 0.0,
        "BLEU-4": np.mean(metrics["bleu4"]) if metrics["bleu4"] else 0.0,
        "ROUGE-L": np.mean(metrics["rougeL"]) if metrics["rougeL"] else 0.0,
    }

    return avg_metrics


def main():
    # === Initialize Training ===
    print("✅ Starting SignVLM training (WandB disabled for TA submission)")
    print(f"📋 Configuration:")
    print(f"   - Device: {DEVICE}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Max frames: {MAX_FRAMES}")
    print(f"   - Visual dim: {VISUAL_DIM}")
    print(f"   - Landmark dim: {LANDMARK_DIM}")
    print(f"   - Motion dim: {MOTION_DIM}")
    print(f"   - Model dim: {D_MODEL}")
    print(f"   - Fusion layers: {NUM_FUSION_LAYERS}")
    print(f"   - Query tokens: {NUM_QUERY_TOKENS}")
    print(f"   - Flow stride: {FLOW_STRIDE}")
    print(f"   - LLM: {LLM_NAME}")
    print(f"   - Data path: {DATA_PATH}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Warmup steps: {WARMUP_STEPS}")
    print(f"   - Weight decay: {WEIGHT_DECAY}")
    print(f"   - Architecture: {APPROACH}")
    print()
    
    # === Load Model ===
    if APPROACH == "dynamic":
        print("Loading SignVLMDynamic model...")
        model = SignVLMDynamic(
        llm_name=LLM_NAME,
        visual_dim=VISUAL_DIM,
        motion_dim=MOTION_DIM,
        landmark_dim=LANDMARK_DIM,
        d_model=D_MODEL,
        num_fusion_layers=NUM_FUSION_LAYERS,
        num_query_tokens=NUM_QUERY_TOKENS,
        device=DEVICE
        )
    elif APPROACH == "pooling":
        print("Loading SignVLMPooling model...")
        model = SignVLMPooling(
            llm_name=LLM_NAME,
            visual_dim=VISUAL_DIM,
            motion_dim=MOTION_DIM,
            landmark_dim=LANDMARK_DIM,
            d_model=D_MODEL,
            num_fusion_layers=NUM_FUSION_LAYERS,
            num_pooled_tokens=NUM_POOLED_TOKENS,
            device=DEVICE
        )
    elif APPROACH == "adapter":
        print("Loading SignVLMAdapter model...")
        model = SignVLMAdapter(
            llm_name=LLM_NAME,
            visual_dim=VISUAL_DIM,
            motion_dim=MOTION_DIM,
            landmark_dim=LANDMARK_DIM,
            d_model=D_MODEL,
            num_query_tokens=NUM_QUERY_TOKENS,
            use_lite_adapter=True,
            device=DEVICE
        )
    elif APPROACH == "dynamic_no_vq":
        print("Loading SignVLMDynamicNoVQ model...")
        model = SignVLMDynamicNoVQ(
            llm_name=LLM_NAME,
            visual_dim=VISUAL_DIM,
            motion_dim=MOTION_DIM,
            landmark_dim=LANDMARK_DIM,
            d_model=D_MODEL,
            num_fusion_layers=NUM_FUSION_LAYERS,
            device=DEVICE
        )
    elif APPROACH == "simple":
        print("Loading SignVLMSimple model...")
        model = SignVLMSimple(
            llm_name=LLM_NAME,
            visual_dim=VISUAL_DIM,
            motion_dim=MOTION_DIM,
            landmark_dim=LANDMARK_DIM,
            d_model=D_MODEL,
            max_sequence_length=64,  # Reduced for efficiency
            landmark_weight=2.0,     # Emphasize landmarks as requested
            device=DEVICE
        )
    elif APPROACH == "landmark_only":
        print("Loading SignVLMLandmark model...")
        model = SignVLMLandmark(
            llm_name=LLM_NAME,
            landmark_dim=LANDMARK_DIM,
            d_model=512,
            num_temporal_layers=2,
            nhead=8,
            device=DEVICE
        )
    elif APPROACH == "t5_style":
        print("Loading SignVLMT5Style model (YouTubeASL approach with multimodal enhancement)...")
        model = SignVLMT5Style(
            llm_name="google/flan-t5-base",
            visual_dim=VISUAL_DIM,
            motion_dim=MOTION_DIM,
            landmark_dim=LANDMARK_DIM,
            max_frames=256,
            max_text_length=128,
            device=DEVICE
        )
        # Override learning rate for full T5 fine-tuning
        print(f"Using lower learning rate for full T5 training: {LEARNING_RATE}")

    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # === Load Dataset ===
    print("Loading dataset...")
    full_dataset = PhoenixSimplified(
        root=DATA_PATH,
        split='train',
        max_frames=MAX_FRAMES,
        flow_stride=FLOW_STRIDE,
        max_samples=MAX_SAMPLES
    )

    val_size = int(len(full_dataset) * VAL_SPLIT_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # === Setup Optimizer and Scheduler ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )
    
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = create_learning_rate_scheduler(optimizer, WARMUP_STEPS, total_steps)
    
    # === Load Checkpoint (if resuming) ===
    start_epoch = 0
    best_rouge = 0.0
    global_step = 0
    
    if RESUME_FROM_CHECKPOINT:
        start_epoch, best_rouge, global_step = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, DEVICE
        )
        # Fast-forward scheduler to correct step
        for _ in range(global_step):
            scheduler.step()
    
    # === Training Loop ===
    print(f"Starting training from epoch {start_epoch + 1}...")
    print(f"Current best ROUGE-L: {best_rouge:.4f}")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            visual_feats = batch["spatial"].to(DEVICE)      # (B, T_spatial, D_spatial)
            motion_feats = batch["motion"].to(DEVICE)        # (B, T_motion, D_motion)
            landmark_feats = batch["landmarks"].to(DEVICE)   # (B, T_landmarks, D_landmarks)
            target_sentences = batch["tgt"]                  # List of strings
            
            try:
                loss = model.compute_loss(visual_feats, motion_feats, landmark_feats, target_sentences)
                
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                if global_step % LOG_EVERY == 0:
                    # Log metrics to console instead of wandb
                    print(f"Step {global_step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
                    
                    if hasattr(model, 'last_diagnostics'):
                        diag = model.last_diagnostics
                        print(f"   Diagnostics: codebook_usage={diag.get('codebook_usage', 'N/A'):.3f}, "
                              f"entropy={diag.get('codebook_entropy', 'N/A'):.3f}, "
                              f"chunks={diag.get('num_chunks', 'N/A')}")
                    
                    # wandb.log(log_dict)  # Disabled for TA submission
                
                if batch_idx % (LOG_EVERY * 5) == 0:
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], "
                          f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
                    print(f"  Sample target: {target_sentences[0][:100]}...")
                    
            except Exception as e:
                print(f"Training step failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] finished. Average Training Loss: {avg_epoch_loss:.4f}")
        
        # === Evaluation ===
        if (epoch + 1) % EVAL_EVERY == 0:
            print(f"Starting evaluation for epoch {epoch+1}...")
            eval_metrics = evaluate_model(model, val_dataloader, DEVICE)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Metrics: {eval_metrics}")
            
            # Log to console instead of wandb
            print(f"📊 Epoch {epoch + 1} Summary:")
            print(f"   - Average train loss: {avg_epoch_loss:.4f}")
            for metric, value in eval_metrics.items():
                print(f"   - {metric}: {value:.4f}")
            print()
            
            # Save best model
            if eval_metrics["ROUGE-L"] > best_rouge:
                best_rouge = eval_metrics["ROUGE-L"]
                print(f"New best ROUGE-L: {best_rouge:.4f}")
                try:
                    # Save model state
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_rouge': best_rouge,
                        'global_step': global_step,
                        'config': {
                            'llm_name': LLM_NAME,
                            'visual_dim': VISUAL_DIM,
                            'motion_dim': MOTION_DIM,
                            'landmark_dim': LANDMARK_DIM,
                            'd_model': D_MODEL,
                            'num_fusion_layers': NUM_FUSION_LAYERS,
                            'num_query_tokens': NUM_QUERY_TOKENS,
                        }
                    }, "best_signvlm_fixed_checkpoint.pt")
                    print("✅ Saved best model checkpoint")
                except Exception as e:
                    print(f"❌ Failed to save checkpoint: {e}")
    
    # === Save Final Model ===
    print("Training finished. Saving final model state.")
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': NUM_EPOCHS,
            'global_step': global_step,
            'best_rouge': best_rouge,
            'config': {
                'llm_name': LLM_NAME,
                'visual_dim': VISUAL_DIM,
                'motion_dim': MOTION_DIM,
                'landmark_dim': LANDMARK_DIM,
                'd_model': D_MODEL,
                'num_fusion_layers': NUM_FUSION_LAYERS,
                'num_query_tokens': NUM_QUERY_TOKENS,
            }
        }, "final_signvlm_fixed_checkpoint.pt")
        print("✅ Saved final model checkpoint")
    except Exception as e:
        print(f"❌ Failed to save final checkpoint: {e}")
    
    # wandb.finish()  # Disabled for submission
    print("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()
