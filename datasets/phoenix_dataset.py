import glob
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from .phoenix_preprocessing import clean_phoenix_2014, clean_phoenix_2014_trans
from .asl_dataset import compute_dense_optical_flow, extract_landmarks
from visual_encoder import get_visual_encoder, preprocess_frames
from mediapipe_encoder import create_holistic_model
from data_utils import sample_frames


class defaultdict_with_warning(defaultdict):
    warned = set()
    warning_enabled = False

    def __getitem__(self, key):
        if key == "text" and key not in self.warned and self.warning_enabled:
            print(
                'Warning: using batch["text"] to obtain label is deprecated, '
                'please use batch["label"] instead.'
            )
            self.warned.add(key)
        return super().__getitem__(key)


class Phoenix14TCorpus:
    mean = [0.53724027, 0.5272855, 0.51954997]
    std = [1, 1, 1]

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.annotations_dir = os.path.join(self.root_dir, "annotations", "manual")
        self.features_dir = os.path.join(self.root_dir, "features")

    def load_data_frame(self, split):
        csv_path = os.path.join(self.annotations_dir, f"PHOENIX-2014-T.{split}.corpus.csv")
        df = pd.read_csv(csv_path, delimiter='|')
        df.rename(columns={'name': 'id'}, inplace=True)
        df.rename(columns={'speaker': 'signer'}, inplace=True)
        return df

    def get_frames(self, sample, split):
        video_id = sample["id"]
        frames_path_pattern = os.path.join(self.features_dir, split, video_id, "images*.png")
        frames = sorted(glob.glob(frames_path_pattern))
       #if not frames:
       #    # TODO: Fallback if the primary way doesn't find images (e.g. if image_folder_name was a sub-sub-directory)
       #    print(f"Warning: No frames found for {video_id} at {frames_path_pattern}")
        return frames

    def create_vocab(self, split="train"):
        df = self.load_data_frame(split)
        words = set()
        for text_sentence in df["translation"]:
            if pd.isna(text_sentence):
                continue
            cleaned_sentence = clean_phoenix_2014_trans(text_sentence)
            words.update(cleaned_sentence.lower().split())
        
        vocab = {word: i for i, word in enumerate(sorted(list(words)))}
        vocab["<unk>"] = len(vocab)
        vocab["<pad>"] = len(vocab)
        return vocab


class PhoenixVideoTextDataset(Dataset):
    Corpus = Phoenix14TCorpus

    def __init__(
        self,
        root="/work/courses/csnlp/Team3/slt/datasets/Phoenix14T",
        split="train",
        random_crop=True,
        base_size=[256, 256],
        crop_size=[224, 224],
        llm_tokenizer=None,
        flow_stride=1,
        max_frames=None,
        max_samples=None,
        max_target_text_len=128,
    ):
        assert (
            self.Corpus is not None
        ), f"Corpus is not defined in the derived class {self.__class__.__name__}."
        assert llm_tokenizer is not None, "LLM tokenizer must be provided."

        self.corpus = self.Corpus(root)
        self.split = split
        self.llm_tokenizer = llm_tokenizer
        self.max_target_text_len = max_target_text_len
        self.flow_stride = flow_stride
        self.max_frames = max_frames

        self.main_cache_dir = os.path.join(root, "cache")
        os.makedirs(self.main_cache_dir, exist_ok=True)
        self.videos_specific_cache_dir = os.path.join(self.main_cache_dir, self.split)

        self.visual_encoder = get_visual_encoder()
        self.holistic_model = create_holistic_model()

        self.data_frame = self.corpus.load_data_frame(split)
        self.german_vocab = self.corpus.create_vocab(split="train")
        self.transform = transforms.Compose(
            [
                transforms.Resize(base_size),
                transforms.RandomCrop(crop_size)
                if random_crop
                else transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.Corpus.mean, self.Corpus.std),
            ]
        )
        
        # Pre-filter dataframe for entries that have frames to avoid errors during __getitem__
        # This is a bit slow at initialization but safer.
        valid_indices = []
        for i in range(len(self.data_frame)):
            sample = self.data_frame.iloc[i].to_dict()
            frames = self.corpus.get_frames(sample, self.split)
            if frames:
                valid_indices.append(i)
        
        if len(valid_indices) < len(self.data_frame):
            print(f"Warning: Dropped {len(self.data_frame) - len(valid_indices)} samples from '{self.split}' split due to missing frames.")
        self.data_frame = self.data_frame.iloc[valid_indices].reset_index(drop=True)

        if max_samples is not None and max_samples < len(self.data_frame):
            self.data_frame = self.data_frame.head(max_samples)
            print(f"Limiting dataset to {max_samples} samples.")


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        sample_info = self.data_frame.iloc[index].to_dict()
        
        video_id = sample_info['id']
        video_specific_cache_dir = os.path.join(self.main_cache_dir, self.split, video_id)
        os.makedirs(video_specific_cache_dir, exist_ok=True)
        
        frames_paths = self.corpus.get_frames(sample_info, self.split)

        if self.max_frames is not None and len(frames_paths) > self.max_frames:
            indices = np.linspace(0, len(frames_paths) - 1, self.max_frames, dtype=int)
            sampled_frames_paths = [frames_paths[i] for i in indices]
        else:
            sampled_frames_paths = frames_paths
        frames_pil = []
        for p in sampled_frames_paths:
            img = Image.open(p).convert('RGB')
            frames_pil.append(img)
        
        if not frames_pil:
            print(f"Error: All selected frames failed to load for {sample_info['id']}. Returning dummy.")
            return None
        else:
            frames_transformed = [self.transform(f) for f in frames_pil]
            frames_tensor = torch.stack(frames_transformed) # (T, C, H, W) - main video tensor
        # Process German translation for LLM target
        german_text_str = sample_info.get("translation", "")
        if pd.isna(german_text_str):
            german_text_str = ""
        
        cleaned_german_text_str = clean_phoenix_2014_trans(german_text_str)
        cleaned_german_text_str_lower = cleaned_german_text_str.lower()
        # Tokenize German text using the dataset's German vocab (for reference or other uses)
        german_vocab_label = [self.german_vocab.get(word, self.german_vocab.get("<unk>", 0)) for word in cleaned_german_text_str_lower.split()]
        if not german_vocab_label: # Handle empty translation string or all words are OOV and no <unk>
            german_vocab_label = [self.german_vocab.get("<pad>", self.german_vocab.get("<unk>", 0))]
        # Tokenize German text using the LLM's tokenizer for LLM targets
        llm_tokenized = self.llm_tokenizer(
            cleaned_german_text_str_lower,
            add_special_tokens=True, # Adds BOS and EOS for Llama tokenizer by default
            padding='do_not_pad', # We will pad in collate_fn
            truncation=True,
            max_length=self.max_target_text_len,
            return_attention_mask=False
        )
        llm_target_token_ids = llm_tokenized['input_ids']
        if not llm_target_token_ids: # If text is empty or only special tokens after truncation
            # Use pad token if empty after tokenization (e.g. empty string becomes [BOS, EOS] then gets truncated if max_length is small)
            llm_target_token_ids = [self.llm_tokenizer.pad_token_id if self.llm_tokenizer.pad_token_id is not None else self.llm_tokenizer.eos_token_id]
        # Keep original gloss annotation (DGS)
        gloss_annotation_str = sample_info.get("annotation", "")
        if pd.isna(gloss_annotation_str): # Handle potential NaN
            gloss_annotation_str = ""
        
        cleaned_gloss_annotation_str = clean_phoenix_2014(gloss_annotation_str)


        # === Motion Encoder (Optical Flow)
        flow_cache_filename = f"flow_stride{self.flow_stride}_max_frames{self.max_frames}.pt"
        flow_cache = os.path.join(video_specific_cache_dir, flow_cache_filename)
        if os.path.exists(flow_cache):
            flow_feats = torch.load(flow_cache, weights_only=True)
        else:
            frames_np = [np.array(img) for img in frames_pil]
            flow_feats_computed = compute_dense_optical_flow(frames_np, stride=self.flow_stride, max_len=self.max_frames)
            flow_feats = torch.tensor(flow_feats_computed).float()
            torch.save(flow_feats, flow_cache)


        # === Visual Encoder
        visual_cache_filename = f"visual_feats_max_frames{self.max_frames}.pt"
        visual_cache = os.path.join(video_specific_cache_dir, visual_cache_filename)
        if os.path.exists(visual_cache):
            visual_feats = torch.load(visual_cache, weights_only=True)
            if visual_feats.ndim == 3 and visual_feats.shape[0] == 1:
                # Squeeze if it's an old cached file with shape (1, T, D)
                visual_feats = visual_feats.squeeze(0)
        else:
            batched_frames_for_visual_encoder = frames_tensor # (T, C, H, W) from self.transform
            with torch.no_grad():
                device = next(self.visual_encoder.parameters()).device
                batched_frames_for_visual_encoder = batched_frames_for_visual_encoder.to(dtype=torch.float32, device=device)
                visual_output = self.visual_encoder.vision_model(pixel_values=batched_frames_for_visual_encoder)
                visual_feats = visual_output.last_hidden_state[:, 0, :] # Shape: (T, D)
            torch.save(visual_feats, visual_cache)

            
        # === Skeleton Encoder
        skeleton_cache_filename = f"skeleton_feats_max_frames{self.max_frames}.pt"
        skeleton_cache = os.path.join(video_specific_cache_dir, skeleton_cache_filename)
        if os.path.exists(skeleton_cache):
            skeleton_feats = torch.load(skeleton_cache, weights_only=True)
        else:
            frames_np = [np.array(img) for img in frames_pil]
            landmarks = extract_landmarks(frames_np, self.holistic_model)
            skeleton_feats = torch.tensor(landmarks).float()  # (T, D)
            torch.save(skeleton_feats, skeleton_cache)


        sample_output = defaultdict_with_warning()
        sample_output.update(sample_info)
        
        sample_output["video"] = frames_tensor
        sample_output["label"] = torch.tensor(german_vocab_label).long() # Tokenized German words (custom vocab)
        sample_output["llm_target_token_ids"] = torch.tensor(llm_target_token_ids).long() # NEW LLM-tokenized targets
        sample_output["translation_text"] = cleaned_german_text_str_lower  # Store cleaned, lowercased German text
        sample_output["gloss_annotation"] = cleaned_gloss_annotation_str # Store cleaned gloss string
        sample_output["video_dataset_instance"] = self
        sample_output["flow_feats"] = flow_feats
        sample_output["skeleton_feats"] = skeleton_feats
        sample_output["visual_feats"] = visual_feats
        
        return sample_output

    @staticmethod
    def collate_fn(batch):
        collated = defaultdict_with_warning(list)
        
        max_video_len = 0
        max_visual_feat_len = 0
        max_skeleton_feat_len = 0
        max_flow_feat_len = 0
        max_llm_target_len = 0

        for sample in batch:
            video_tensor = sample["video"]
            if isinstance(video_tensor, torch.Tensor):
                 max_video_len = max(max_video_len, video_tensor.shape[0])
            
            visual_feat_tensor = sample.get("visual_feats")
            if isinstance(visual_feat_tensor, torch.Tensor):
                max_visual_feat_len = max(max_visual_feat_len, visual_feat_tensor.shape[0])

            skeleton_feat_tensor = sample.get("skeleton_feats")
            if isinstance(skeleton_feat_tensor, torch.Tensor):
                max_skeleton_feat_len = max(max_skeleton_feat_len, skeleton_feat_tensor.shape[0])

            flow_feat_tensor = sample.get("flow_feats")
            if isinstance(flow_feat_tensor, torch.Tensor):
                max_flow_feat_len = max(max_flow_feat_len, flow_feat_tensor.shape[0])

            llm_target_tensor = sample.get("llm_target_token_ids")
            if isinstance(llm_target_tensor, torch.Tensor):
                max_llm_target_len = max(max_llm_target_len, llm_target_tensor.shape[0])


        dummy_c, dummy_h, dummy_w = 3, 224, 224 # Default for video
        try:
            valid_sample_video = next(s["video"] for s in batch if isinstance(s["video"], torch.Tensor) and s["video"].ndim == 4)
            dummy_c, dummy_h, dummy_w = valid_sample_video.shape[1:]
        except StopIteration:
            pass 
        
        visual_dim = batch[0].get("visual_feats").shape[1] if batch and isinstance(batch[0].get("visual_feats"), torch.Tensor) else 768
        skeleton_dim = batch[0].get("skeleton_feats").shape[1] if batch and isinstance(batch[0].get("skeleton_feats"), torch.Tensor) else 129
        flow_dim = batch[0].get("flow_feats").shape[1] if batch and isinstance(batch[0].get("flow_feats"), torch.Tensor) else 2


        for sample in batch:
            # Pad video tensor
            video_tensor = sample["video"]
            if isinstance(video_tensor, torch.Tensor):
                c, h, w = video_tensor.shape[1:]
                padding_len = max_video_len - video_tensor.shape[0]
                if padding_len > 0:
                    padding = torch.zeros((padding_len, c, h, w), dtype=video_tensor.dtype)
                    video_tensor = torch.cat([video_tensor, padding], dim=0)
            else:
                video_tensor = torch.zeros((max_video_len, dummy_c, dummy_h, dummy_w))
            collated["video"].append(video_tensor)

            # Pad visual_feats (T, D_vis)
            vf_tensor = sample.get("visual_feats")
            if isinstance(vf_tensor, torch.Tensor):
                d_vis = vf_tensor.shape[1]
                padding_len = max_visual_feat_len - vf_tensor.shape[0]
                if padding_len > 0:
                    padding = torch.zeros((padding_len, d_vis), dtype=vf_tensor.dtype)
                    vf_tensor = torch.cat([vf_tensor, padding], dim=0)
            else: # Should not happen if __getitem__ is correct
                vf_tensor = torch.zeros((max_visual_feat_len, visual_dim))
            collated["visual_feats"].append(vf_tensor)

            # Pad skeleton_feats (T, D_skel)
            sf_tensor = sample.get("skeleton_feats")
            if isinstance(sf_tensor, torch.Tensor):
                d_skel = sf_tensor.shape[1]
                padding_len = max_skeleton_feat_len - sf_tensor.shape[0]
                if padding_len > 0:
                    padding = torch.zeros((padding_len, d_skel), dtype=sf_tensor.dtype)
                    sf_tensor = torch.cat([sf_tensor, padding], dim=0)
            else:
                sf_tensor = torch.zeros((max_skeleton_feat_len, skeleton_dim))
            collated["skeleton_feats"].append(sf_tensor)

            # Pad flow_feats (T, D_flow)
            ff_tensor = sample.get("flow_feats")
            if isinstance(ff_tensor, torch.Tensor):
                d_flow = ff_tensor.shape[1]
                padding_len = max_flow_feat_len - ff_tensor.shape[0]
                if padding_len > 0:
                    padding = torch.zeros((padding_len, d_flow), dtype=ff_tensor.dtype)
                    ff_tensor = torch.cat([ff_tensor, padding], dim=0)
            else:
                ff_tensor = torch.zeros((max_flow_feat_len, flow_dim))
            collated["flow_feats"].append(ff_tensor)
            
            # label: (L_target_tokens) - this is now tokenized German with custom vocab
            collated["label_unpadded"].append(sample["label"]) # Store original for reference if needed
            collated["llm_target_token_ids_unpadded"].append(sample["llm_target_token_ids"])
            collated["signer"].append(sample.get("signer", "UnknownSigner"))
            # Updated keys:
            collated["translation_text"].append(sample.get("translation_text", "")) # Original German text
            collated["gloss_annotation"].append(sample.get("gloss_annotation", "")) # Original gloss annotation
            collated["id"].append(sample.get("id", "UnknownID"))

        # Pad labels
        # Assuming vocab["<pad>"] exists and is the padding index
        # This part should be robust to vocab not having <pad> but it's standard practice
        pad_idx = batch[0]["video_dataset_instance"].german_vocab.get("<pad>", -1) if batch and hasattr(batch[0].get("video_dataset_instance"), "german_vocab") else -1
        if pad_idx == -1:
             # Attempt to get pad_idx from the first sample's vocab if video_dataset_instance was not passed
            try:
                first_sample_vocab = batch[0].default_factory().__self__.german_vocab
                pad_idx = first_sample_vocab.get("<pad>", -1)
            except:
                 print("Warning: <pad> token not found in vocab or vocab inaccessible. Using 0 for padding labels.")
                 pad_idx = 0


        max_label_len = 0
        for label_tensor in collated["label_unpadded"]:
            max_label_len = max(max_label_len, len(label_tensor))
        
        for label_tensor in collated["label_unpadded"]:
            padding_len = max_label_len - len(label_tensor)
            if padding_len > 0:
                padding = torch.full((padding_len,), pad_idx, dtype=label_tensor.dtype)
                padded_label = torch.cat([label_tensor, padding], dim=0)
            else:
                padded_label = label_tensor
            collated["label"].append(padded_label)
            collated["text"].append(padded_label) # for backward compatibility

        # Pad llm_target_token_ids
        llm_pad_idx = batch[0]["video_dataset_instance"].llm_tokenizer.pad_token_id
        if llm_pad_idx is None: # Fallback if pad_token_id is not explicitly set
            llm_pad_idx = batch[0]["video_dataset_instance"].llm_tokenizer.eos_token_id
        if llm_pad_idx is None: # Ultimate fallback if neither are set (highly unlikely for good tokenizers)
             print("Warning: LLM tokenizer has no pad_token_id or eos_token_id. Using 0 for padding LLM target tokens.")
             llm_pad_idx = 0
        
        for label_tensor in collated["llm_target_token_ids_unpadded"]:
            padding_len = max_llm_target_len - len(label_tensor)
            if padding_len > 0:
                padding = torch.full((padding_len,), llm_pad_idx, dtype=label_tensor.dtype)
                padded_label = torch.cat([label_tensor, padding], dim=0)
            else:
                padded_label = label_tensor
            collated["llm_target_token_ids"].append(padded_label)


        # Stack tensors
        collated["video"] = torch.stack(collated["video"])
        collated["label"] = torch.stack(collated["label"]) # Custom German vocab tokens
        collated["text"] = torch.stack(collated["text"])   # Backward compatibility for "label"
        collated["llm_target_token_ids"] = torch.stack(collated["llm_target_token_ids"]) # LLM specific tokens
        collated["visual_feats"] = torch.stack(collated["visual_feats"])
        collated["skeleton_feats"] = torch.stack(collated["skeleton_feats"])
        collated["flow_feats"] = torch.stack(collated["flow_feats"])
        
        collated.warning_enabled = True
        
        # Add video_dataset_instance to each sample in __getitem__ to access vocab in collate_fn
        # This is a way to pass dataset-level info like vocab to collate_fn
        # Modify __getitem__ to: sample_output["video_dataset_instance"] = self
        # For now, this part of collate_fn regarding pad_idx might need direct vocab access or passing vocab.

        del collated["label_unpadded"] # cleanup
        del collated["llm_target_token_ids_unpadded"] # cleanup
        return dict(collated)

# Example usage
if __name__ == '__main__':
    # Path relative to the directory where `python -m datasets.phoenix_dataset` is run (e.g., slt/)
    root_dir = "datasets/Phoenix14T" 
    
    # Check if the root directory and necessary subdirectories exist
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} not found. Skipping example usage.")
    elif not os.path.exists(os.path.join(root_dir, "annotations", "manual")):
        print(f"Annotations directory not found in {root_dir}. Skipping example usage.")
    elif not os.path.exists(os.path.join(root_dir, "features", "train")): # Check for train features
        print(f"Features train directory not found in {root_dir}. Skipping example usage.")
    else:
        print(f"Attempting to load Phoenix14T dataset from: {root_dir}")
        try:
            # Create dataset instance
            # For testing __main__, we need a dummy tokenizer or load one.
            # This part will error if not run via train.py which supplies the tokenizer.
            # For a quick standalone test, one might do:
            # from transformers import AutoTokenizer
            # dummy_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B") # or any other
            # train_dataset = PhoenixVideoTextDataset(root=root_dir, split="train", flow_stride=1, max_frames=64, llm_tokenizer=dummy_tokenizer)
            
            # Assuming this script is NOT run standalone for this part, so llm_tokenizer would be provided by train.py
            # The following lines are for when llm_tokenizer IS provided:
            # train_dataset = PhoenixVideoTextDataset(root=root_dir, split="train", flow_stride=1, max_frames=64, llm_tokenizer=llm_tokenizer_instance_from_train_py)

            # To make the existing __main__ runnable with a placeholder for llm_tokenizer for now:
            print("Note: For __main__ example usage, a placeholder LLM tokenizer will be used if not supplied via imports.")
            try:
                from transformers import AutoTokenizer
                example_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B") # Example
            except ImportError:
                print("transformers library not found, cannot create example tokenizer for __main__.")
                example_tokenizer = None
            
            if example_tokenizer:
                 train_dataset = PhoenixVideoTextDataset(root=root_dir, split="train", flow_stride=1, max_frames=64, llm_tokenizer=example_tokenizer)
            
                 if len(train_dataset) == 0:
                    print("Dataset is empty. Check paths and data.")
                 else:
                    print(f"Dataset loaded. Number of train samples: {len(train_dataset)}")
                    print(f"Vocab size: {len(train_dataset.german_vocab)}")
                    # print(f"Vocab: {train_dataset.german_vocab}")

                    # Create DataLoader
                    # To use the custom collate_fn, it should be passed to DataLoader
                    # However, the current collate_fn has a slight issue accessing vocab for pad_idx
                    # Let's refine __getitem__ to pass 'self' and collate_fn to use it
                    # For now, let's test __getitem__ directly first.
                    
                    sample_idx = 0
                    print(f"Fetching sample {sample_idx}...")
                    try:
                        sample = train_dataset[sample_idx]
                        print(f"Sample ID: {sample['id']}")
                        print(f"Video tensor shape: {sample['video'].shape}") # T, C, H, W
                        print(f"Label tensor (tokenized German, custom vocab): {sample['label']}")
                        print(f"LLM Target Tokens (LLM vocab): {sample['llm_target_token_ids']}")
                        print(f"Original German translation: {sample['translation_text']}")
                        print(f"Original Gloss annotation: {sample['gloss_annotation']}")
                        print(f"Signer: {sample['signer']}")

                        # Test collate_fn with a small batch
                        print("\nTesting collate_fn...")
                        # To properly test collate_fn, it needs access to dataset's vocab for pad_idx
                        # Modify __getitem__ to include a reference to the dataset instance
                        # Add this to PhoenixVideoTextDataset.__getitem__ before return:
                        # sample_output["video_dataset_instance"] = self
                        
                        # --- Modification for testing collate_fn ---
                        # Temporarily add this to __getitem__ for the test run:
                        # sample_output["video_dataset_instance"] = self 
                        # This is a common pattern if collate_fn needs dataset properties.
                        
                        # Re-fetch sample with the (assumed) modification if you were to test collate directly here
                        # For now, we'll construct a dummy batch for structure testing

                        if len(train_dataset) >= 2:
                            raw_batch = [train_dataset[i] for i in range(2)] # Get two samples
                            
                            # Manually add dataset instance to each sample for this test
                            for s_item in raw_batch:
                                s_item["video_dataset_instance"] = train_dataset

                            collated_batch = PhoenixVideoTextDataset.collate_fn(raw_batch)
                            print("Collated batch keys:", collated_batch.keys())
                            print("Collated video shape:", collated_batch["video"].shape) # B, T_padded, C, H, W
                            print("Collated label shape:", collated_batch["label"].shape) # B, L_padded
                            print("Collated first ID:", collated_batch["id"][0])
                        else:
                            print("Not enough samples in dataset to test collate_fn with a batch of 2.")

                    except IndexError:
                        print(f"Could not retrieve sample {sample_idx}. Dataset might be smaller than expected or filtering removed it.")
                    except Exception as e:
                        print(f"Error during sample fetching or collate_fn test: {e}")
                        import traceback
                        traceback.print_exc()

            # Add a test for max_samples
            if os.path.exists(root_dir) and \
               os.path.exists(os.path.join(root_dir, "annotations", "manual")) and \
               os.path.exists(os.path.join(root_dir, "features", "train")):
                print("\nTesting with max_samples=2...")
                try:
                    # Create dataset instance with max_samples
                    test_dataset = PhoenixVideoTextDataset(root=root_dir, split="train", flow_stride=1, max_frames=64, max_samples=2, llm_tokenizer=example_tokenizer)
                    
                    if len(test_dataset) == 0:
                        print("Max_samples test: Dataset is empty. Check paths and data.")
                    else:
                        print(f"Max_samples test: Dataset loaded. Number of train samples: {len(test_dataset)}")
                        assert len(test_dataset) <= 2, "Dataset length exceeds max_samples"
                        
                        if len(test_dataset) > 0:
                            sample_idx = 0
                            print(f"Fetching sample {sample_idx} from max_samples limited dataset...")
                            sample = test_dataset[sample_idx]
                            print(f"Sample ID: {sample['id']}")

                except FileNotFoundError as e:
                    print(f"Error initializing dataset for max_samples test: {e}.")
                except Exception as e:
                    print(f"An unexpected error occurred during max_samples test: {e}")
                    import traceback
                    traceback.print_exc()
        
        except FileNotFoundError as e:
            print(f"Error initializing dataset: {e}. Check dataset paths and structure.")
        except Exception as e:
            print(f"An unexpected error occurred during dataset initialization or testing: {e}")
            import traceback
            traceback.print_exc() 