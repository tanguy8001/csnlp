# SignVLM: Sign Language Translation

A comprehensive framework for German sign language translation using multimodal vision-language models. This project implements multiple architectural variants for translating sign language videos to German text.

## 🚀 Overview

SignVLM combines visual features, optical flow motion data, and pose landmarks to generate accurate German translations of sign language videos. The project includes several architectural approaches ranging from simple concatenation to sophisticated cross-attention fusion mechanisms.

### Key Features

- **Multimodal Fusion**: Combines visual (CLIP), motion (optical flow), and landmark (pose) features
- **Multiple Architectures**: Different model variants with varying complexity

## 📁 Project Structure

```
├── sign_vlm.py              # Core model architectures
├── train.py                 # Training script with all configurations
├── datasets/
│   ├── phoenix_simplified.py    # Phoenix14T dataset loader
│   ├── asl_dataset.py           # Alternative dataset implementation (unused)
│   ├── constants.py             # Dataset constants
│   └── Phoenix14T/              # Dataset directory (to be downloaded)
├── fusion_model.py          # Cross-attention fusion components
├── final_llm.py            # LLM integration utilities
├── visual_encoder.py       # CLIP visual feature extraction
├── optical_flow_encoder.py # Dense optical flow computation
├── dynamic_chunker.py      # Dynamic sequence chunking
├── quantization_module.py  # Vector quantization
├── sign_adapter.py         # SignAdapter fusion modules (unused)
└── data_utils.py          # Data processing utilities
```

## 🏗️ Model Architectures

### 1. **SignVLM** (Base)
- Cross-modal attention fusion between visual, motion, and landmark features
- Vision-language connector with learnable query tokens
- LoRA fine-tuning of frozen LLM

### 2. **SignVLMDynamic** 
- Adds dynamic chunking and vector quantization
- Adaptive sequence compression based on content similarity
- Codebook learning for discrete representation

### 3. **SignVLMPooling**
- Simple adaptive pooling approach
- Fixed-length sequence compression
- Lightweight alternative to dynamic chunking

### 4. **SignVLMAdapter**
- Uses SignAdapter modules for multimodal fusion
- Efficient parameter sharing across modalities
- Lite and full adapter variants

### 5. **SignVLMDynamicNoVQ**
- Dynamic chunking without quantization
- Direct projection to LLM space
- Best performing variant in experiments

### 6. **SignVLMSimple**
- Concatenates all modality features
- Passes through MLP to LLM input space
- Special landmark emphasis (2x weighting)
- No discrete tokenization - clean and effective

### 7. **SignVLMLandmark**
- Landmark-only model for ablation studies
- Temporal transformer processing
- Focuses solely on pose information

### 8. **SignVLMT5Style** 🔥 **State-of-the-art**
- Based on Google's YouTubeASL paper
- T5 encoder-decoder architecture
- Full model fine-tuning (not LoRA)
- Multimodal enhancement with all three modalities
- Lower learning rate (1e-5) for stability

## 🛠️ Setup

### Prerequisites

```bash
# Create conda environment
conda create -n signvlm python=3.9
conda activate signvlm

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install required packages
pip install transformers
pip install peft  # For LoRA fine-tuning
pip install datasets
pip install opencv-python
pip install nltk
pip install rouge-score
pip install tqdm
pip install numpy
pip install Pillow

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Dataset Setup

1. **Download Phoenix14T Dataset**:
   ```bash
   # Create dataset directory
   mkdir -p datasets/Phoenix14T
   
   # Download from official source:
   # https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
   
   # The dataset should have this structure:
   # datasets/Phoenix14T/
   #   ├── features/
   #   │   ├── fullFrame-210x260px/  # Visual features
   #   │   ├── flow_data/            # Optical flow data  
   #   │   └── landmarks/            # Pose landmark data
   #   └── annotations/
   #       ├── manual/
   #       │   ├── train.corpus.csv
   #       │   ├── dev.corpus.csv
   #       │   └── test.corpus.csv
   ```

2. **Verify Dataset**:
   ```bash
   python -c "
   from datasets.phoenix_simplified import PhoenixSimplified
   dataset = PhoenixSimplified('datasets/Phoenix14T', split='train', max_samples=10)
   print(f'Dataset loaded: {len(dataset)} samples')
   print('Sample:', dataset[0].keys())
   "
   ```

## 🚂 Training

### Quick Start

```bash
# Train SignVLMSimple (recommended for beginners)
python train.py
```

### Architecture Selection

Edit the `APPROACH` variable in `train.py`:

```python
# Choose one of:
APPROACH = "simple"           # SignVLMSimple (recommended)
APPROACH = "t5_style"         # SignVLMT5Style (best performance)
APPROACH = "dynamic_no_vq"    # SignVLMDynamicNoVQ
APPROACH = "pooling"          # SignVLMPooling
APPROACH = "adapter"          # SignVLMAdapter
APPROACH = "dynamic"          # SignVLMDynamic
APPROACH = "landmark_only"    # SignVLMLandmark
```

### Key Training Parameters

```python
# Main configuration in train.py
BATCH_SIZE = 4              # Adjust based on GPU memory
NUM_EPOCHS = 200            # Training epochs
LEARNING_RATE = 1e-5        # Lower for T5, higher for LoRA models
MAX_FRAMES = 256            # Maximum video frames
DEVICE = 'cuda'             # Use GPU if available

# Model dimensions
VISUAL_DIM = 768           # CLIP visual features
LANDMARK_DIM = 129         # Pose landmarks (43 points × 3 coords)
MOTION_DIM = 2             # Optical flow (x, y displacement)
D_MODEL = 768              # Internal model dimension
```

### Resume Training

```python
# In train.py, set:
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_PATH = "best_signvlm_fixed_checkpoint.pt"
```

### Training Output

The training script provides comprehensive logging:
- Loss tracking per step and epoch
- BLEU and ROUGE-L metrics on validation set
- Model diagnostics (for applicable architectures)
- Checkpoint saving for best models
- Training examples for qualitative assessment

## 📊 Evaluation

### Automatic Metrics

The training script computes:
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram overlap with reference
- **ROUGE-L**: Longest common subsequence similarity
- **Loss**: Cross-entropy loss on validation set
