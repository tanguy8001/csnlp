"""
SignVLM: A Vision-Language Model for Sign Language Translation
Using proper vision feature prepending instead of token replacement

Architecture:
1. Multimodal Feature Extractors (Visual + Motion + Skeleton)
2. Cross-Modal Fusion with attention  
3. Vision-Language Connector
4. Prepend vision tokens to text sequence (like final_llm.py)
5. Causal Language Model for generation with proper loss masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from typing import Optional, List, Dict, Any
import math

from dynamic_chunker import DynamicChunker
from quantization_module import VectorQuantizer
from sign_adapter import SignAdapter, SignAdapterLite


def normalize_landmarks_to_unit_box(landmarks):
    """
    Normalize landmarks to fit in a unit bounding box [0,1] x [0,1] across the entire clip.
    
    Args:
        landmarks: (B, T, D) where D contains x,y coordinates
                   Assumes D = 129 = 43 landmarks * 3 (x,y,visibility)
    
    Returns:
        normalized_landmarks: (B, T, D) with x,y coordinates normalized to [0,1]
    """
    B, T, D = landmarks.shape
    

    landmarks_reshaped = landmarks.view(B, T, -1, 3)
    
    x_coords = landmarks_reshaped[:, :, :, 0]
    y_coords = landmarks_reshaped[:, :, :, 1]
    visibility = landmarks_reshaped[:, :, :, 2]
    
    normalized_landmarks = landmarks_reshaped.clone()
    
    for b in range(B):
        valid_mask = visibility[b] > 0.5
        
        if valid_mask.any():
            x_valid = x_coords[b][valid_mask]
            y_valid = y_coords[b][valid_mask]
            
            x_min, x_max = x_valid.min(), x_valid.max()
            y_min, y_max = y_valid.min(), y_valid.max()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            if x_range > 1e-6:
                normalized_landmarks[b, :, :, 0] = (x_coords[b] - x_min) / x_range
            else:
                normalized_landmarks[b, :, :, 0] = 0.5
                
            if y_range > 1e-6:
                normalized_landmarks[b, :, :, 1] = (y_coords[b] - y_min) / y_range
            else:
                normalized_landmarks[b, :, :, 1] = 0.5
        else:
            pass
    
    return normalized_landmarks.view(B, T, D)


class TemporalPositionalEncoding(nn.Module):
    """Learned temporal positional encoding for video sequences"""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x):
        """x: (B, T, D)"""
        B, T, D = x.shape
        pos_emb = self.pos_embedding[:, :T, :]
        return x + pos_emb


class MultimodalFeatureExtractor(nn.Module):
    """Extract and process features from multiple modalities"""
    def __init__(self, visual_dim: int = 768, motion_dim: int = 2, landmark_dim: int = 129, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
           
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, d_model // 4),  
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Dropout(0.1)
        )
        
        self.landmark_proj = nn.Sequential(
            nn.Linear(landmark_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.visual_pos = TemporalPositionalEncoding(d_model)
        self.motion_pos = TemporalPositionalEncoding(d_model)
        self.landmark_pos = TemporalPositionalEncoding(d_model)
        
        self.visual_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.motion_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.landmark_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(self, visual_feats, motion_feats, landmark_feats):
        """
        Args:
            visual_feats: (B, T_v, D_v)
            motion_feats: (B, T_m, D_m) 
            landmark_feats: (B, T_l, D_l)
        Returns:
            Dict with processed features for each modality
        """
        B = visual_feats.shape[0]
        
        normalized_landmark_feats = normalize_landmarks_to_unit_box(landmark_feats)
        
        visual_proj = self.visual_proj(visual_feats) 
        motion_proj = self.motion_proj(motion_feats) 
        landmark_proj = self.landmark_proj(normalized_landmark_feats)  
        
        visual_with_pos = self.visual_pos(visual_proj)
        motion_with_pos = self.motion_pos(motion_proj)
        landmark_with_pos = self.landmark_pos(landmark_proj)
        
        visual_final = visual_with_pos + self.visual_type_emb.expand(B, visual_with_pos.shape[1], -1)
        motion_final = motion_with_pos + self.motion_type_emb.expand(B, motion_with_pos.shape[1], -1)
        landmark_final = landmark_with_pos + self.landmark_type_emb.expand(B, landmark_with_pos.shape[1], -1)
        
        return {
            'visual': visual_final,
            'motion': motion_final,
            'landmarks': landmark_final
        }


class CrossModalFusionLayer(nn.Module):
    """Cross-modal attention layer for fusing different modalities"""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.visual_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.motion_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.landmark_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.visual_motion_cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.visual_landmark_cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.motion_landmark_cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.landmark_motion_cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.visual_ffn = self._make_ffn(d_model, dropout)
        self.motion_ffn = self._make_ffn(d_model, dropout)
        self.landmark_ffn = self._make_ffn(d_model, dropout)
        
        self.visual_norm1 = nn.LayerNorm(d_model)
        self.visual_norm2 = nn.LayerNorm(d_model)
        self.visual_norm3 = nn.LayerNorm(d_model)
        
        self.motion_norm1 = nn.LayerNorm(d_model)
        self.motion_norm2 = nn.LayerNorm(d_model)
        self.motion_norm3 = nn.LayerNorm(d_model)
        
        self.landmark_norm1 = nn.LayerNorm(d_model)
        self.landmark_norm2 = nn.LayerNorm(d_model)
        self.landmark_norm3 = nn.LayerNorm(d_model)
        
        # Learnable fusion weights instead of fixed 0.5
        self.visual_motion_gate = nn.Parameter(torch.tensor(0.5))
        self.visual_landmark_gate = nn.Parameter(torch.tensor(0.5))
        self.motion_gate = nn.Parameter(torch.tensor(0.5))
        self.landmark_gate = nn.Parameter(torch.tensor(0.5))
        
    def _make_ffn(self, d_model: int, dropout: float):
        return nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, modal_feats: Dict[str, torch.Tensor]):
        """
        Args:
            modal_feats: Dict with keys 'visual', 'motion', 'landmarks'
        Returns:
            Enhanced features for each modality
        """
        visual = modal_feats['visual']
        motion = modal_feats['motion'] 
        landmarks = modal_feats['landmarks']
        
        visual_self, _ = self.visual_self_attn(visual, visual, visual)
        visual = self.visual_norm1(visual + visual_self)
        
        motion_self, _ = self.motion_self_attn(motion, motion, motion)
        motion = self.motion_norm1(motion + motion_self)
        
        landmark_self, _ = self.landmark_self_attn(landmarks, landmarks, landmarks)
        landmarks = self.landmark_norm1(landmarks + landmark_self)
        
        visual_motion, _ = self.visual_motion_cross(visual, motion, motion)
        visual_landmark, _ = self.visual_landmark_cross(visual, landmarks, landmarks)
        
        # Enhanced visual features with cross-modal information using learnable gates
        visual_enhanced = visual + torch.sigmoid(self.visual_motion_gate) * visual_motion + torch.sigmoid(self.visual_landmark_gate) * visual_landmark
        visual_enhanced = self.visual_norm2(visual_enhanced)
        
        motion_visual, _ = self.motion_landmark_cross(motion, visual, visual)
        motion_enhanced = motion + torch.sigmoid(self.motion_gate) * motion_visual
        motion_enhanced = self.motion_norm2(motion_enhanced)
        
        landmark_motion, _ = self.landmark_motion_cross(landmarks, motion, motion)  # Dedicated landmark to motion attention
        landmark_enhanced = landmarks + torch.sigmoid(self.landmark_gate) * landmark_motion
        landmark_enhanced = self.landmark_norm2(landmark_enhanced)
        
        # Feed-forward networks
        visual_final = self.visual_norm3(visual_enhanced + self.visual_ffn(visual_enhanced))
        motion_final = self.motion_norm3(motion_enhanced + self.motion_ffn(motion_enhanced))
        landmark_final = self.landmark_norm3(landmark_enhanced + self.landmark_ffn(landmark_enhanced))
        
        return {
            'visual': visual_final,
            'motion': motion_final, 
            'landmarks': landmark_final
        }


class VisionLanguageConnector(nn.Module):
    """Connect multimodal features to language model input space"""
    def __init__(self, d_model: int, llm_dim: int, num_query_tokens: int = 16):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, d_model) * 0.02)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, modal_feats: Dict[str, torch.Tensor]):
        """
        Args:
            modal_feats: Dict with processed multimodal features
        Returns:
            vision_tokens: (B, num_query_tokens, llm_dim) - tokens for LLM input
        """
        B = list(modal_feats.values())[0].shape[0]
        
        all_feats = torch.cat([
            modal_feats['visual'],
            modal_feats['motion'],
            modal_feats['landmarks']
        ], dim=1)
        
        queries = self.query_tokens.expand(B, -1, -1)  
        
        cross_out, _ = self.cross_attn(queries, all_feats, all_feats)
        queries = self.norm1(queries + cross_out)
        
        self_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + self_out)
        
        vision_tokens = self.projection(queries)  
        
        return vision_tokens





class SignVLM(nn.Module):
    """
    Fixed Sign Language Vision-Language Model using prepending approach
    """
    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-3.2-1B",
        visual_dim: int = 768,
        motion_dim: int = 2,
        landmark_dim: int = 129,
        d_model: int = 768,
        num_fusion_layers: int = 3,
        num_query_tokens: int = 16,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        base_llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        
        for param in base_llm.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(base_llm, lora_config)
        
        
        self.feature_extractor = MultimodalFeatureExtractor(
            visual_dim=visual_dim,
            motion_dim=motion_dim, 
            landmark_dim=landmark_dim,
            d_model=d_model
        )
        
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(d_model) for _ in range(num_fusion_layers)
        ])
        
        self.vision_connector = VisionLanguageConnector(
            d_model=d_model,
            llm_dim=self.llm.config.hidden_size,
            num_query_tokens=num_query_tokens
        )
        
        self.prompt_text = "Translate this sign language video to German:"
        
        self.to(device)
    
    def _get_prompt_embeddings(self, batch_size: int):
        """Get embeddings for the prompt text"""
        prompts = [self.prompt_text] * batch_size
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            prompt_embeddings = self.llm.get_input_embeddings()(encoded.input_ids)
        
        return prompt_embeddings, encoded.attention_mask
    
    def forward(self, visual_feats, motion_feats, landmark_feats, target_texts=None, max_new_tokens=50):
        """
        Args:
            visual_feats: (B, T_v, D_v)
            motion_feats: (B, T_m, D_m)
            landmark_feats: (B, T_l, D_l) 
            target_texts: List[str] for training, None for inference
        """
        B = visual_feats.shape[0]
        
        modal_feats = self.feature_extractor(visual_feats, motion_feats, landmark_feats)
        
        for fusion_layer in self.fusion_layers:
            modal_feats = fusion_layer(modal_feats)
        
        vision_tokens = self.vision_connector(modal_feats) 

        if target_texts is not None:
            return self._forward_training(vision_tokens, target_texts)
        else:
            return self._forward_inference(vision_tokens, max_new_tokens)
    
    def _forward_training(self, vision_tokens, target_texts):
        """Training forward pass using prepending approach"""
        B = vision_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        P_len = prompt_embeddings.shape[1]
        V_len = vision_tokens.shape[1]
        
        target_encoded = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True  
        ).to(self.device)
        
        target_ids = target_encoded.input_ids
        target_mask = target_encoded.attention_mask
        
        text_input_ids = target_ids[:, :-1]
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)   
        T_len = text_embeddings.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        vision_tokens = vision_tokens.to(embed_dtype)
        text_embeddings = text_embeddings.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings, 
            vision_tokens,      
            text_embeddings  
        ], dim=1) 
        
        attention_mask = torch.cat([
            prompt_mask,                                           
            torch.ones(B, V_len, dtype=torch.long, device=self.device),  
            target_mask[:, :-1]                                     
        ], dim=1)  
        
       
        target_labels = target_ids[:, 1:]  
        
        target_labels = target_labels.masked_fill(target_mask[:, 1:] == 0, -100)
        
        labels = torch.cat([
            torch.full((B, P_len), -100, dtype=torch.long, device=self.device), 
            torch.full((B, V_len), -100, dtype=torch.long, device=self.device), 
            target_labels                                                       
        ], dim=1)  
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def _forward_inference(self, vision_tokens, max_new_tokens):
        """Inference forward pass using prepending approach"""
        B = vision_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        V_len = vision_tokens.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        vision_tokens = vision_tokens.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            vision_tokens,
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
        ], dim=1)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        input_length = inputs_embeds.shape[1]
        
        for i, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_loss(self, visual_feats, motion_feats, landmark_feats, target_texts):
        """Compute training loss"""
        outputs = self.forward(visual_feats, motion_feats, landmark_feats, target_texts)
        return outputs.loss
    
    def generate(self, visual_feats, motion_feats, landmark_feats, max_new_tokens=50):
        """Generate translations"""
        return self.forward(visual_feats, motion_feats, landmark_feats, None, max_new_tokens)


class SignVLMDynamic(nn.Module):
    """
    SignVLM with Dynamic Chunking + Quantization approach
    """
    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-3.2-1B",
        visual_dim: int = 768,
        motion_dim: int = 2,
        landmark_dim: int = 129,
        d_model: int = 768,
        num_fusion_layers: int = 3,
        num_query_tokens: int = 16,
        codebook_size: int = 512,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        
        for param in base_llm.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(base_llm, lora_config)
        
        self.feature_extractor = MultimodalFeatureExtractor(
            visual_dim=visual_dim,
            motion_dim=motion_dim, 
            landmark_dim=landmark_dim,
            d_model=d_model
        )
        
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(d_model) for _ in range(num_fusion_layers)
        ])
        
        self.dynamic_chunker = DynamicChunker(
            dim=d_model,
            max_chunk_len=32,
            nhead=8
        )
        
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            dim=d_model,
            tau=2.0,
            commitment_beta=1.0
        )
        
        self.token_to_llm = nn.Linear(d_model, self.llm.config.hidden_size)
        
        # Chunk positional embeddings to preserve temporal order after chunking
        self.chunk_pos_embedding = nn.Parameter(torch.randn(1, 128, self.llm.config.hidden_size) * 0.02)
        
        self.prompt_text = "Translate this sign language video to German:"
        
        self.to(device)
    
    def _get_prompt_embeddings(self, batch_size: int):
        """Get embeddings for the prompt text"""
        prompts = [self.prompt_text] * batch_size
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            prompt_embeddings = self.llm.get_input_embeddings()(encoded.input_ids)
        
        return prompt_embeddings, encoded.attention_mask
    
    def forward(self, visual_feats, motion_feats, landmark_feats, target_texts=None, max_new_tokens=50):
        """
        Args:
            visual_feats: (B, T_v, D_v)
            motion_feats: (B, T_m, D_m)
            landmark_feats: (B, T_l, D_l) 
            target_texts: List[str] for training, None for inference
        """
        B = visual_feats.shape[0]
        
        modal_feats = self.feature_extractor(visual_feats, motion_feats, landmark_feats)
        
        for fusion_layer in self.fusion_layers:
            modal_feats = fusion_layer(modal_feats)
        
        primary_feats = modal_feats['visual']
        
        chunk_embeddings = self.dynamic_chunker(primary_feats, threshold=0.7)
        
        max_allowed_chunks = 128
        if chunk_embeddings.shape[1] > max_allowed_chunks:
            print(f"Warning: Chunker produced {chunk_embeddings.shape[1]} chunks, limiting to {max_allowed_chunks}")
            chunk_embeddings = chunk_embeddings[:, :max_allowed_chunks, :]
        
        quantized_embeddings, indices, codebook_loss, commitment_loss, entropy = self.quantizer(chunk_embeddings)
        
        self.last_diagnostics = {
            'codebook_usage': len(torch.unique(indices)) / self.quantizer.codebook_size,
            'codebook_entropy': entropy.mean().item(),
            'num_chunks': chunk_embeddings.shape[1],
            'commitment_loss': commitment_loss.item(),
            'codebook_loss': codebook_loss.item()
        }
        
        llm_tokens = self.token_to_llm(quantized_embeddings)
        
        # Add chunk positional embeddings to preserve temporal order
        B, num_chunks, llm_dim = llm_tokens.shape
        
        # Expand positional embeddings if needed
        if num_chunks > self.chunk_pos_embedding.shape[1]:
            max_pos = self.chunk_pos_embedding.shape[1]
            extra_pos = self.chunk_pos_embedding[:, -1:, :].expand(1, num_chunks - max_pos, -1)
            chunk_pos_full = torch.cat([self.chunk_pos_embedding, extra_pos], dim=1)
            chunk_pos = chunk_pos_full[:, :num_chunks, :].expand(B, -1, -1)
        else:
            chunk_pos = self.chunk_pos_embedding[:, :num_chunks, :].expand(B, -1, -1)
        
        llm_tokens = llm_tokens + chunk_pos
        
        if target_texts is not None:
            return self._forward_training(llm_tokens, target_texts, codebook_loss, commitment_loss)
        else:
            return self._forward_inference(llm_tokens, max_new_tokens)
    
    def _forward_training(self, llm_tokens, target_texts, codebook_loss, commitment_loss):
        """Training forward pass with quantization losses"""
        B = llm_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        P_len = prompt_embeddings.shape[1]
        V_len = llm_tokens.shape[1]
        
        target_encoded = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True
        ).to(self.device)
        
        target_ids = target_encoded.input_ids
        target_mask = target_encoded.attention_mask
        
        text_input_ids = target_ids[:, :-1]
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)
        T_len = text_embeddings.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        llm_tokens = llm_tokens.to(embed_dtype)
        text_embeddings = text_embeddings.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            llm_tokens,
            text_embeddings
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
            target_mask[:, :-1]
        ], dim=1)
        
        target_labels = target_ids[:, 1:]
        target_labels = target_labels.masked_fill(target_mask[:, 1:] == 0, -100)
        
        labels = torch.cat([
            torch.full((B, P_len), -100, dtype=torch.long, device=self.device),
            torch.full((B, V_len), -100, dtype=torch.long, device=self.device),
            target_labels
        ], dim=1)
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        total_loss = outputs.loss + 0.1 * codebook_loss + 0.1 * commitment_loss
        
        outputs.loss = total_loss
        return outputs
    
    def _forward_inference(self, llm_tokens, max_new_tokens):
        """Inference forward pass"""
        B = llm_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        V_len = llm_tokens.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        llm_tokens = llm_tokens.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            llm_tokens,
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
        ], dim=1)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        input_length = inputs_embeds.shape[1]
        
        for i, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_loss(self, visual_feats, motion_feats, landmark_feats, target_texts):
        """Compute training loss"""
        outputs = self.forward(visual_feats, motion_feats, landmark_feats, target_texts)
        return outputs.loss
    
    def generate(self, visual_feats, motion_feats, landmark_feats, max_new_tokens=50):
        """Generate translations"""
        return self.forward(visual_feats, motion_feats, landmark_feats, None, max_new_tokens)


class SignVLMPooling(nn.Module):
    """
    SignVLM with Simple Mean Pooling approach
    Replaces dynamic chunking + quantization with straightforward pooling
    """
    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-3.2-1B",
        visual_dim: int = 768,
        motion_dim: int = 2,
        landmark_dim: int = 129,
        d_model: int = 768,
        num_fusion_layers: int = 3,
        num_pooled_tokens: int = 16,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.num_pooled_tokens = num_pooled_tokens
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        
        for param in base_llm.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(base_llm, lora_config)
        
        self.feature_extractor = MultimodalFeatureExtractor(
            visual_dim=visual_dim,
            motion_dim=motion_dim, 
            landmark_dim=landmark_dim,
            d_model=d_model
        )
        
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(d_model) for _ in range(num_fusion_layers)
        ])
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_pooled_tokens)  # Pool to fixed number of tokens
        
        self.to_llm = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.llm.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        )
        
        self.pooled_pos_embedding = nn.Parameter(torch.randn(1, num_pooled_tokens, self.llm.config.hidden_size) * 0.02)

        self.prompt_text = "Translate this sign language video to German:"
        
        self.to(device)
    
    def _get_prompt_embeddings(self, batch_size: int):
        """Get embeddings for the prompt text"""
        prompts = [self.prompt_text] * batch_size
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            prompt_embeddings = self.llm.get_input_embeddings()(encoded.input_ids)
        
        return prompt_embeddings, encoded.attention_mask
    
    def forward(self, visual_feats, motion_feats, landmark_feats, target_texts=None, max_new_tokens=50):
        """
        Args:
            visual_feats: (B, T_v, D_v)
            motion_feats: (B, T_m, D_m)
            landmark_feats: (B, T_l, D_l) 
            target_texts: List[str] for training, None for inference
        """
        B = visual_feats.shape[0]
        
        modal_feats = self.feature_extractor(visual_feats, motion_feats, landmark_feats)
        
        for fusion_layer in self.fusion_layers:
            modal_feats = fusion_layer(modal_feats)
        
        all_feats = torch.cat([
            modal_feats['visual'],
            modal_feats['motion'],
            modal_feats['landmarks']
        ], dim=1)
        
        # pooling
        pooled_feats = self.adaptive_pool(all_feats.transpose(1, 2))
        pooled_feats = pooled_feats.transpose(1, 2)
        
        llm_tokens = self.to_llm(pooled_feats)
        
        llm_tokens = llm_tokens + self.pooled_pos_embedding.expand(B, -1, -1)
        
        self.last_diagnostics = {
            'num_pooled_tokens': self.num_pooled_tokens,
            'input_sequence_length': all_feats.shape[1],
            'compression_ratio': all_feats.shape[1] / self.num_pooled_tokens
        }
        
        if target_texts is not None:
            return self._forward_training(llm_tokens, target_texts)
        else:
            return self._forward_inference(llm_tokens, max_new_tokens)
    
    def _forward_training(self, llm_tokens, target_texts):
        """Training forward pass"""
        B = llm_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        P_len = prompt_embeddings.shape[1]
        V_len = llm_tokens.shape[1]
        
        target_encoded = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True
        ).to(self.device)
        
        target_ids = target_encoded.input_ids
        target_mask = target_encoded.attention_mask
        
        # Get text embeddings for input (all but last token)
        text_input_ids = target_ids[:, :-1]
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)
        T_len = text_embeddings.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        llm_tokens = llm_tokens.to(embed_dtype)
        text_embeddings = text_embeddings.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            llm_tokens,
            text_embeddings
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
            target_mask[:, :-1]
        ], dim=1)
        
        target_labels = target_ids[:, 1:]
        target_labels = target_labels.masked_fill(target_mask[:, 1:] == 0, -100)
        
        labels = torch.cat([
            torch.full((B, P_len), -100, dtype=torch.long, device=self.device),
            torch.full((B, V_len), -100, dtype=torch.long, device=self.device),
            target_labels
        ], dim=1)
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def _forward_inference(self, llm_tokens, max_new_tokens):
        """Inference forward pass (same as other approaches)"""
        B = llm_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        V_len = llm_tokens.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        llm_tokens = llm_tokens.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            llm_tokens,
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
        ], dim=1)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        input_length = inputs_embeds.shape[1]
        
        for i, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_loss(self, visual_feats, motion_feats, landmark_feats, target_texts):
        """Compute training loss (much simpler - no quantization losses)"""
        outputs = self.forward(visual_feats, motion_feats, landmark_feats, target_texts)
        return outputs.loss
    
    def generate(self, visual_feats, motion_feats, landmark_feats, max_new_tokens=50):
        """Generate translations"""
        return self.forward(visual_feats, motion_feats, landmark_feats, None, max_new_tokens)


class SignVLMAdapter(nn.Module):
    """
    SignVLM using SignAdapter for multimodal fusion instead of cross-attention
    """
    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-3.2-1B",
        visual_dim: int = 768,
        motion_dim: int = 2,
        landmark_dim: int = 129,
        d_model: int = 768,
        num_query_tokens: int = 16,
        use_lite_adapter: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        
        for param in base_llm.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(base_llm, lora_config)
        
        # Use SignAdapter instead of separate feature extractor + fusion layers
        if use_lite_adapter:
            self.multimodal_fusion = SignAdapterLite(
                d_spatial=visual_dim,
                d_motion=motion_dim,
                d_landmarks=landmark_dim,
                d_model=d_model
            )
        else:
            self.multimodal_fusion = SignAdapter(
                d_spatial=visual_dim,
                d_motion=motion_dim,
                d_landmarks=landmark_dim,
                d_model=d_model
            )
        
        # Vision-Language Connector
        self.vision_connector = VisionLanguageConnector(
            d_model=d_model,
            llm_dim=self.llm.config.hidden_size,
            num_query_tokens=num_query_tokens
        )
        
        self.prompt_text = "Translate this sign language video to German:"
        
        self.to(device)
    
    def _get_prompt_embeddings(self, batch_size: int):
        """Get embeddings for the prompt text"""
        prompts = [self.prompt_text] * batch_size
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            prompt_embeddings = self.llm.get_input_embeddings()(encoded.input_ids)
        
        return prompt_embeddings, encoded.attention_mask
    
    def forward(self, visual_feats, motion_feats, landmark_feats, target_texts=None, max_new_tokens=50):
        """
        Args:
            visual_feats: (B, T_v, D_v)
            motion_feats: (B, T_m, D_m)
            landmark_feats: (B, T_l, D_l) 
            target_texts: List[str] for training, None for inference
        """
        B = visual_feats.shape[0]
        
        # Use SignAdapter for fusion
        fused_feats = self.multimodal_fusion(visual_feats, motion_feats, landmark_feats)
        
        modal_feats = {
            'visual': fused_feats,
            'motion': torch.zeros_like(fused_feats[:, :0, :]),
            'landmarks': torch.zeros_like(fused_feats[:, :0, :])
        }
        
        vision_tokens = self.vision_connector(modal_feats)
        
        if target_texts is not None:
            return self._forward_training(vision_tokens, target_texts)
        else:
            return self._forward_inference(vision_tokens, max_new_tokens)
    
    def _forward_training(self, vision_tokens, target_texts):
        """Training forward pass using prepending approach"""
        B = vision_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        P_len = prompt_embeddings.shape[1]
        V_len = vision_tokens.shape[1]
        
        target_encoded = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True
        ).to(self.device)
        
        target_ids = target_encoded.input_ids
        target_mask = target_encoded.attention_mask
        
        text_input_ids = target_ids[:, :-1]
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)
        T_len = text_embeddings.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        vision_tokens = vision_tokens.to(embed_dtype)
        text_embeddings = text_embeddings.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings, 
            vision_tokens,      
            text_embeddings  
        ], dim=1) 
        
        attention_mask = torch.cat([
            prompt_mask,                                           
            torch.ones(B, V_len, dtype=torch.long, device=self.device),  
            target_mask[:, :-1]                                     
        ], dim=1)  
        
        target_labels = target_ids[:, 1:]  
        target_labels = target_labels.masked_fill(target_mask[:, 1:] == 0, -100)
        
        labels = torch.cat([
            torch.full((B, P_len), -100, dtype=torch.long, device=self.device), 
            torch.full((B, V_len), -100, dtype=torch.long, device=self.device), 
            target_labels                                                       
        ], dim=1)  
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def _forward_inference(self, vision_tokens, max_new_tokens):
        """Inference forward pass using prepending approach"""
        B = vision_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        V_len = vision_tokens.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        vision_tokens = vision_tokens.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            vision_tokens,
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
        ], dim=1)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        input_length = inputs_embeds.shape[1]
        
        for i, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_loss(self, visual_feats, motion_feats, landmark_feats, target_texts):
        """Compute training loss"""
        outputs = self.forward(visual_feats, motion_feats, landmark_feats, target_texts)
        return outputs.loss
    
    def generate(self, visual_feats, motion_feats, landmark_feats, max_new_tokens=50):
        """Generate translations"""
        return self.forward(visual_feats, motion_feats, landmark_feats, None, max_new_tokens)


class SignVLMDynamicNoVQ(nn.Module):
    """
    SignVLM with Dynamic Chunking, NO Quantization, and direct projection
    """
    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-3.2-1B",
        visual_dim: int = 768,
        motion_dim: int = 2,
        landmark_dim: int = 129,
        d_model: int = 768,
        num_fusion_layers: int = 3,
        max_allowed_chunks: int = 128,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.max_allowed_chunks = max_allowed_chunks
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        
        for param in base_llm.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(base_llm, lora_config)
        
        self.feature_extractor = MultimodalFeatureExtractor(
            visual_dim=visual_dim,
            motion_dim=motion_dim, 
            landmark_dim=landmark_dim,
            d_model=d_model
        )
        
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(d_model) for _ in range(num_fusion_layers)
        ])
        
        self.dynamic_chunker = DynamicChunker(
            dim=d_model,
            max_chunk_len=32,
            nhead=8
        )
        
        self.chunk_to_llm_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.llm.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        )
        

        self.chunk_pos_embedding = nn.Parameter(torch.randn(1, max_allowed_chunks, self.llm.config.hidden_size) * 0.02)
        
        self.prompt_text = "Translate this sign language video to German:"
        
        self.to(device)
    
    def _get_prompt_embeddings(self, batch_size: int):
        """Get embeddings for the prompt text"""
        prompts = [self.prompt_text] * batch_size
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            prompt_embeddings = self.llm.get_input_embeddings()(encoded.input_ids)
        
        return prompt_embeddings, encoded.attention_mask
    
    def forward(self, visual_feats, motion_feats, landmark_feats, target_texts=None, max_new_tokens=50):
        """
        Args:
            visual_feats: (B, T_v, D_v)
            motion_feats: (B, T_m, D_m)
            landmark_feats: (B, T_l, D_l) 
            target_texts: List[str] for training, None for inference
        """
        B = visual_feats.shape[0]
        
        modal_feats = self.feature_extractor(visual_feats, motion_feats, landmark_feats)
        
        for fusion_layer in self.fusion_layers:
            modal_feats = fusion_layer(modal_feats)
        
        primary_feats = modal_feats['visual']
        
        # Dynamic chunking
        chunk_embeddings = self.dynamic_chunker(primary_feats, threshold=0.7)
        
        if chunk_embeddings.shape[1] > self.max_allowed_chunks:
            chunk_embeddings = chunk_embeddings[:, :self.max_allowed_chunks, :]

        llm_tokens = self.chunk_to_llm_proj(chunk_embeddings)
        
        B, num_chunks, llm_dim = llm_tokens.shape
        
        chunk_pos = self.chunk_pos_embedding[:, :num_chunks, :].expand(B, -1, -1)
        llm_tokens = llm_tokens + chunk_pos
        
        if target_texts is not None:
            return self._forward_training(llm_tokens, target_texts)
        else:
            return self._forward_inference(llm_tokens, max_new_tokens)
    
    def _forward_training(self, llm_tokens, target_texts):
        """Training forward pass using prepending approach"""
        B = llm_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        P_len = prompt_embeddings.shape[1]
        V_len = llm_tokens.shape[1]
        
        target_encoded = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True 
        ).to(self.device)
        
        target_ids = target_encoded.input_ids
        target_mask = target_encoded.attention_mask
        
        text_input_ids = target_ids[:, :-1]
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids) 

        

        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        llm_tokens = llm_tokens.to(embed_dtype)
        text_embeddings = text_embeddings.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            llm_tokens,
            text_embeddings
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
            target_mask[:, :-1]
        ], dim=1)
        
        target_labels = target_ids[:, 1:]
        target_labels = target_labels.masked_fill(target_mask[:, 1:] == 0, -100)
        
        labels = torch.cat([
            torch.full((B, P_len), -100, dtype=torch.long, device=self.device),
            torch.full((B, V_len), -100, dtype=torch.long, device=self.device),
            target_labels
        ], dim=1)
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def _forward_inference(self, llm_tokens, max_new_tokens):
        """Inference forward pass"""
        B = llm_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        V_len = llm_tokens.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        llm_tokens = llm_tokens.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            llm_tokens,
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
        ], dim=1)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        input_length = inputs_embeds.shape[1]
        
        for i, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_loss(self, visual_feats, motion_feats, landmark_feats, target_texts):
        """Compute training loss"""
        outputs = self.forward(visual_feats, motion_feats, landmark_feats, target_texts)
        return outputs.loss
    
    def generate(self, visual_feats, motion_feats, landmark_feats, max_new_tokens=50):
        """Generate translations"""
        return self.forward(visual_feats, motion_feats, landmark_feats, None, max_new_tokens)


class SignVLMSimple(nn.Module):
    """
    Simple SignVLM that concatenates all three modality features and passes through MLP
    No discrete tokenization, no complex attention - just concatenation + MLP
    Special emphasis on landmarks as requested
    """
    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-3.2-1B",
        visual_dim: int = 768,
        motion_dim: int = 2,
        landmark_dim: int = 129,
        d_model: int = 768,
        max_sequence_length: int = 256,
        landmark_weight: float = 2.0,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.landmark_weight = landmark_weight
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        
        for param in base_llm.parameters():
            param.requires_grad = False
            
        # Add LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(base_llm, lora_config)
        
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
           
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, d_model // 4),  
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Dropout(0.1)
        )
        
        self.landmark_proj = nn.Sequential(
            nn.Linear(landmark_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.visual_pos = TemporalPositionalEncoding(d_model)
        self.motion_pos = TemporalPositionalEncoding(d_model)
        self.landmark_pos = TemporalPositionalEncoding(d_model)
        
        self.visual_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.motion_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.landmark_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_sequence_length)
        
        self.multimodal_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, self.llm.config.hidden_size),
            nn.LayerNorm(self.llm.config.hidden_size)
        )
        
        self.final_pos_embedding = nn.Parameter(
            torch.randn(1, max_sequence_length, self.llm.config.hidden_size) * 0.02
        )
        
        self.prompt_text = "Translate this sign language video to German:"
        
        self.to(device)
    
    def _get_prompt_embeddings(self, batch_size: int):
        """Get embeddings for the prompt text"""
        prompts = [self.prompt_text] * batch_size
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            prompt_embeddings = self.llm.get_input_embeddings()(encoded.input_ids)
        
        return prompt_embeddings, encoded.attention_mask
    
    def forward(self, visual_feats, motion_feats, landmark_feats, target_texts=None, max_new_tokens=50):
        """
        Args:
            visual_feats: (B, T_v, D_v)
            motion_feats: (B, T_m, D_m)
            landmark_feats: (B, T_l, D_l) 
            target_texts: List[str] for training, None for inference
        """
        B = visual_feats.shape[0]
        
        normalized_landmark_feats = normalize_landmarks_to_unit_box(landmark_feats)
        
        visual_proj = self.visual_proj(visual_feats)
        motion_proj = self.motion_proj(motion_feats)
        landmark_proj = self.landmark_proj(normalized_landmark_feats)
        visual_with_pos = self.visual_pos(visual_proj)
        motion_with_pos = self.motion_pos(motion_proj)
        landmark_with_pos = self.landmark_pos(landmark_proj)
        
        visual_final = visual_with_pos + self.visual_type_emb.expand(B, visual_with_pos.shape[1], -1)
        motion_final = motion_with_pos + self.motion_type_emb.expand(B, motion_with_pos.shape[1], -1)
        landmark_final = landmark_with_pos + self.landmark_type_emb.expand(B, landmark_with_pos.shape[1], -1)
        
        landmark_final = landmark_final * self.landmark_weight
        
        all_feats = torch.cat([
            visual_final,
            motion_final,
            landmark_final
        ], dim=1)
        
        # Pool to fixed sequence length
        pooled_feats = self.adaptive_pool(all_feats.transpose(1, 2))  
        pooled_feats = pooled_feats.transpose(1, 2) 
        
        # Now we have aligned sequences, concatenate along feature dimension
        # For simplicity, split the pooled sequence back into three parts
        seq_len = pooled_feats.shape[1]
        third = seq_len // 3
        
        visual_pooled = pooled_feats[:, :third, :]
        motion_pooled = pooled_feats[:, third:2*third, :]
        landmark_pooled = pooled_feats[:, 2*third:, :]
        
        if landmark_pooled.shape[1] < third:
            pad_size = third - landmark_pooled.shape[1]
            padding = torch.zeros(B, pad_size, d_model, device=self.device, dtype=landmark_pooled.dtype)
            landmark_pooled = torch.cat([landmark_pooled, padding], dim=1)
        elif landmark_pooled.shape[1] > third:
            landmark_pooled = landmark_pooled[:, :third, :]
            
        concatenated_feats = torch.cat([
            visual_pooled,
            motion_pooled,
            landmark_pooled
        ], dim=-1)

        llm_tokens = self.multimodal_mlp(concatenated_feats)
        
        final_seq_len = llm_tokens.shape[1]
        llm_tokens = llm_tokens + self.final_pos_embedding[:, :final_seq_len, :].expand(B, -1, -1)
        
        self.last_diagnostics = {
            'input_sequence_length': all_feats.shape[1],
            'final_sequence_length': final_seq_len,
            'compression_ratio': all_feats.shape[1] / final_seq_len,
            'landmark_weight': self.landmark_weight
        }
        
        if target_texts is not None:
            return self._forward_training(llm_tokens, target_texts)
        else:
            return self._forward_inference(llm_tokens, max_new_tokens)
    
    def _forward_training(self, llm_tokens, target_texts):
        """Training forward pass using prepending approach"""
        B = llm_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        P_len = prompt_embeddings.shape[1]
        V_len = llm_tokens.shape[1]
        
        target_encoded = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True
        ).to(self.device)
        
        target_ids = target_encoded.input_ids
        target_mask = target_encoded.attention_mask
        
        # Get text embeddings for input (all but last token)
        text_input_ids = target_ids[:, :-1]
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        llm_tokens = llm_tokens.to(embed_dtype)
        text_embeddings = text_embeddings.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            llm_tokens,
            text_embeddings
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
            target_mask[:, :-1]
        ], dim=1)
        
        target_labels = target_ids[:, 1:]
        target_labels = target_labels.masked_fill(target_mask[:, 1:] == 0, -100)
        
        labels = torch.cat([
            torch.full((B, P_len), -100, dtype=torch.long, device=self.device),
            torch.full((B, V_len), -100, dtype=torch.long, device=self.device),
            target_labels
        ], dim=1)
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def _forward_inference(self, llm_tokens, max_new_tokens):
        """Inference forward pass"""
        B = llm_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        V_len = llm_tokens.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        llm_tokens = llm_tokens.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            llm_tokens,
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
        ], dim=1)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        generated_texts = []
        input_length = inputs_embeds.shape[1]
        
        for i, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_loss(self, visual_feats, motion_feats, landmark_feats, target_texts):
        """Compute training loss"""
        outputs = self.forward(visual_feats, motion_feats, landmark_feats, target_texts)
        return outputs.loss
    
    def generate(self, visual_feats, motion_feats, landmark_feats, max_new_tokens=50):
        """Generate translations"""
        return self.forward(visual_feats, motion_feats, landmark_feats, None, max_new_tokens)


class SignVLMLandmark(nn.Module):
    """
    A simple yet powerful model inspired by recent findings, focusing exclusively on landmarks.
    It normalizes, temporally processes, and projects landmark data to the LLM.
    This architecture avoids potentially noisy visual/motion features and complex fusion.
    """
    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-3.2-1B",
        landmark_dim: int = 129,
        d_model: int = 512,
        num_temporal_layers: int = 2,
        nhead: int = 8,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        
        for param in base_llm.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(base_llm, lora_config)
        
        self.landmark_input_norm = nn.LayerNorm(landmark_dim)
        
        self.landmark_proj = nn.Linear(landmark_dim, d_model)
        
        self.pos_encoder = TemporalPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_temporal_layers)
        
        self.to_llm_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.llm.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        )
        
        self.prompt_text = "Translate this sign language video to German:"
        
        self.to(device)
    
    def _get_prompt_embeddings(self, batch_size: int):
        """Get embeddings for the prompt text"""
        prompts = [self.prompt_text] * batch_size
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            prompt_embeddings = self.llm.get_input_embeddings()(encoded.input_ids)
        
        return prompt_embeddings, encoded.attention_mask
    
    def forward(self, visual_feats, motion_feats, landmark_feats, target_texts=None, max_new_tokens=50):
        """
        Args:
            landmark_feats: (B, T, D) - Only landmarks used
            visual_feats: Ignored.
            motion_feats: Ignored.
            target_texts: List[str] for training, None for inference.
        """
        B, T, _ = landmark_feats.shape
        
        spatially_normalized_landmarks = normalize_landmarks_to_unit_box(landmark_feats)
        
        feature_normalized_landmarks = self.landmark_input_norm(spatially_normalized_landmarks)
        projected_landmarks = self.landmark_proj(feature_normalized_landmarks)
        
        pos_encoded_landmarks = self.pos_encoder(projected_landmarks)
        temporally_processed_landmarks = self.temporal_encoder(pos_encoded_landmarks)
        
        landmark_tokens = self.to_llm_proj(temporally_processed_landmarks)
        
        if target_texts is not None:
            return self._forward_training(landmark_tokens, target_texts)
        else:
            return self._forward_inference(landmark_tokens, max_new_tokens)

    def _forward_training(self, vision_tokens, target_texts):
        """Training forward pass using prepending approach"""
        B = vision_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        P_len = prompt_embeddings.shape[1]
        V_len = vision_tokens.shape[1]
        
        target_encoded = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True
        ).to(self.device)
        
        target_ids = target_encoded.input_ids
        target_mask = target_encoded.attention_mask
        
        text_input_ids = target_ids[:, :-1]
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        vision_tokens = vision_tokens.to(embed_dtype)
        text_embeddings = text_embeddings.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings, 
            vision_tokens,      
            text_embeddings  
        ], dim=1) 
        
        attention_mask = torch.cat([
            prompt_mask,                                           
            torch.ones(B, V_len, dtype=torch.long, device=self.device),  
            target_mask[:, :-1]                                     
        ], dim=1)  
        
        target_labels = target_ids[:, 1:]  
        
        target_labels = target_labels.masked_fill(target_mask[:, 1:] == 0, -100)
        
        labels = torch.cat([
            torch.full((B, P_len), -100, dtype=torch.long, device=self.device), 
            torch.full((B, V_len), -100, dtype=torch.long, device=self.device), 
            target_labels                                                     
        ], dim=1)  
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def _forward_inference(self, vision_tokens, max_new_tokens):
        """Inference forward pass"""
        B = vision_tokens.shape[0]
        
        prompt_embeddings, prompt_mask = self._get_prompt_embeddings(B)
        V_len = vision_tokens.shape[1]
        
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        prompt_embeddings = prompt_embeddings.to(embed_dtype)
        vision_tokens = vision_tokens.to(embed_dtype)
        
        inputs_embeds = torch.cat([
            prompt_embeddings,
            vision_tokens,
        ], dim=1)
        
        attention_mask = torch.cat([
            prompt_mask,
            torch.ones(B, V_len, dtype=torch.long, device=self.device),
        ], dim=1)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        input_length = inputs_embeds.shape[1]
        
        for i, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_loss(self, visual_feats, motion_feats, landmark_feats, target_texts):
        """Compute training loss. Ignores visual and motion features."""
        outputs = self.forward(visual_feats, motion_feats, landmark_feats, target_texts)
        return outputs.loss
    
    def generate(self, visual_feats, motion_feats, landmark_feats, max_new_tokens=50):
        """Generate translations. Ignores visual and motion features."""
        return self.forward(visual_feats, motion_feats, landmark_feats, None, max_new_tokens)


class SignVLMT5Style(nn.Module):
    """
    Implementation following YouTubeASL paper approach with multimodal enhancement:
    - Concatenate visual + motion + landmarks before projection to encoder
    - Decoder uses standard text token embeddings
    - Clean separation between multimodal processing and text generation
    """
    def __init__(
        self,
        llm_name: str = "google/flan-t5-base",
        visual_dim: int = 768,
        motion_dim: int = 2,
        landmark_dim: int = 129,
        max_frames: int = 256,
        max_text_length: int = 128,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.max_frames = max_frames
        self.max_text_length = max_text_length
        self.visual_dim = visual_dim
        self.motion_dim = motion_dim
        self.landmark_dim = landmark_dim
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        
        # KEY DIFFERENCE: Full T5 fine-tuning (not frozen + LoRA)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_name, torch_dtype=torch.float32)
        
        # Make ALL parameters trainable
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.encoder_dim = self.model.config.d_model
        
        # MULTIMODAL ENHANCEMENT: Process all modalities
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, self.encoder_dim // 3), 
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(motion_dim),
            nn.Linear(motion_dim, self.encoder_dim // 6),
            nn.GELU(),
            nn.Linear(self.encoder_dim // 6, self.encoder_dim // 3),
            nn.Dropout(0.1)
        )
        
        self.landmark_proj = nn.Sequential(
            nn.LayerNorm(landmark_dim),
            nn.Linear(landmark_dim, self.encoder_dim // 3),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        concatenated_dim = (self.encoder_dim // 3) * 3
        self.multimodal_to_encoder = nn.Sequential(
            nn.LayerNorm(concatenated_dim),
            nn.Linear(concatenated_dim, self.encoder_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder_dim, self.encoder_dim)
        )
        
        self.multimodal_pos_embeddings = nn.Parameter(
            torch.randn(1, max_frames, self.encoder_dim) * 0.02
        )
        
        self.to(device)
    
    def forward(self, visual_feats, motion_feats, landmark_feats, target_texts=None, max_new_tokens=50):
        """
        Args:
            visual_feats: (B, T_v, D_v) - Visual features
            motion_feats: (B, T_m, D_m) - Motion features  
            landmark_feats: (B, T_l, D_l) - Landmark features
            target_texts: List[str] for training, None for inference
        """
        B = visual_feats.shape[0]
        
        normalized_landmarks = normalize_landmarks_to_unit_box(landmark_feats)
        
        min_frames = min(visual_feats.shape[1], motion_feats.shape[1], landmark_feats.shape[1])
        min_frames = min(min_frames, self.max_frames)
        
        visual_aligned = visual_feats[:, :min_frames, :].contiguous()
        motion_aligned = motion_feats[:, :min_frames, :].contiguous()  
        landmark_aligned = normalized_landmarks[:, :min_frames, :].contiguous()
        
        visual_proj = self.visual_proj(visual_aligned)      
        motion_proj = self.motion_proj(motion_aligned)
        landmark_proj = self.landmark_proj(landmark_aligned)
        
        multimodal_feats = torch.cat([
            visual_proj,
            motion_proj,
            landmark_proj
        ], dim=-1)
        
        multimodal_embeddings = self.multimodal_to_encoder(multimodal_feats)
        
        T = min_frames
        multimodal_embeddings = multimodal_embeddings + self.multimodal_pos_embeddings[:, :T, :].expand(B, -1, -1)
        
        attention_mask = torch.ones(B, T, dtype=torch.long, device=self.device)
        
        if target_texts is not None:
            return self._forward_training(multimodal_embeddings, attention_mask, target_texts)
        else:
            return self._forward_inference(multimodal_embeddings, attention_mask, max_new_tokens)
    
    def _forward_training(self, multimodal_embeddings, attention_mask, target_texts):
        """Training using T5 encoder-decoder architecture"""
        B = multimodal_embeddings.shape[0]
        
        encoder_outputs = self.model.encoder(
            inputs_embeds=multimodal_embeddings,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        target_encoded = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=True
        ).to(self.device)
        
        decoder_input_ids = target_encoded.input_ids[:, :-1].contiguous()
        labels = target_encoded.input_ids[:, 1:].contiguous()
        
        labels = labels.masked_fill(target_encoded.attention_mask[:, 1:] == 0, -100)
        
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=target_encoded.attention_mask[:, :-1],
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def _forward_inference(self, multimodal_embeddings, attention_mask, max_new_tokens):
        """Inference using T5 encoder-decoder"""
        B = multimodal_embeddings.shape[0]
        
        # ENCODER: Process multimodal features
        encoder_outputs = self.model.encoder(
            inputs_embeds=multimodal_embeddings,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # DECODER: Generate text
        with torch.no_grad():
            output_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        for output in output_ids:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_loss(self, visual_feats, motion_feats, landmark_feats, target_texts):
        """Compute training loss"""
        outputs = self.forward(visual_feats, motion_feats, landmark_feats, target_texts)
        return outputs.loss
    
    def generate(self, visual_feats, motion_feats, landmark_feats, max_new_tokens=50):
        """Generate translations"""
        return self.forward(visual_feats, motion_feats, landmark_feats, None, max_new_tokens)
