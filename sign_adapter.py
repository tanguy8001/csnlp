import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """
    1D Temporal Convolutional Block with residual connections.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        kernel_size: Kernel size for temporal convolution
        dropout: Dropout probability
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, T) - batch, channels, time
        Returns:
            (B, C, T) - same shape
        """
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out


class SignAdapter(nn.Module):
    """
    Enhanced SignAdapter that fuses spatial, motion, and landmark streams.

    Inputs
    ------
    spatial : Tensor (B, T_spatial, D_spatial)
    motion  : Tensor (B, T_motion, D_motion)
    landmarks : Tensor (B, T_landmarks, D_landmarks)

    Output
    ------
    fused   : Tensor (B, T_out, d_model)
    """
    def __init__(self,
                 d_spatial: int,
                 d_motion: int,
                 d_landmarks: int,
                 d_model: int = 768,
                 d_hidden: int = 1024,
                 tcn_kernel: int = 3,
                 tcn_dropout: float = 0.1,
                 num_tcn_blocks: int = 2):
        super().__init__()

        # Project each modality to common dimension d_model
        self.spatial_proj = nn.Linear(d_spatial, d_model)
        self.motion_proj = nn.Linear(d_motion, d_model)
        self.landmarks_proj = nn.Linear(d_landmarks, d_model)

        # Temporal convolution blocks (expects input as (B, C, T))
        self.tcn_blocks = nn.ModuleList([
            TemporalConvBlock(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=tcn_kernel,
                dropout=tcn_dropout
            ) for _ in range(num_tcn_blocks)
        ])

        # Cross-modal attention to better fuse the modalities
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=tcn_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Cross-modal MLP (Linear -> GELU -> Linear)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(tcn_dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(tcn_dropout)
        )

    def forward(self, spatial, motion, landmarks):
        """
        Args:
            spatial: (B, T_spatial, D_spatial)
            motion: (B, T_motion, D_motion)  
            landmarks: (B, T_landmarks, D_landmarks)
        Returns:
            fused: (B, T_out, d_model)
        """
        # 1. Project each modality to common dimension
        s = self.spatial_proj(spatial)      # (B, T_spatial, d_model)
        m = self.motion_proj(motion)        # (B, T_motion, d_model)
        l = self.landmarks_proj(landmarks)  # (B, T_landmarks, d_model)

        # 2. Concatenate over time dimension
        # Note: We concatenate all modalities along time axis
        Z_cat = torch.cat([s, m, l], dim=1)  # (B, T_spatial + T_motion + T_landmarks, d_model)

        # 3. Apply temporal convolution blocks
        # TCN expects (B, C, T) format
        x = Z_cat.transpose(1, 2)  # (B, d_model, T_total)
        
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)  # (B, d_model, T_total)
        
        x = x.transpose(1, 2)  # (B, T_total, d_model)

        # 4. Self-attention for temporal refinement
        x_norm = self.norm1(x)
        attn_out, _ = self.cross_attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # 5. Cross-modal MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        out = x + mlp_out  # (B, T_total, d_model)

        return out


class SignAdapterLite(nn.Module):
    """
    Lightweight version of SignAdapter with fewer parameters.
    """
    def __init__(self,
                 d_spatial: int,
                 d_motion: int,
                 d_landmarks: int,
                 d_model: int = 768,
                 tcn_kernel: int = 3,
                 tcn_dropout: float = 0.1):
        super().__init__()

        # Project each modality to common dimension d_model
        self.spatial_proj = nn.Linear(d_spatial, d_model)
        self.motion_proj = nn.Linear(d_motion, d_model)
        self.landmarks_proj = nn.Linear(d_landmarks, d_model)

        # Single temporal convolution block
        self.tcn = TemporalConvBlock(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout
        )

        # Simple MLP for final fusion
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(tcn_dropout),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, spatial, motion, landmarks):
        """
        Args:
            spatial: (B, T_spatial, D_spatial)
            motion: (B, T_motion, D_motion)  
            landmarks: (B, T_landmarks, D_landmarks)
        Returns:
            fused: (B, T_out, d_model)
        """
        # Project and concatenate
        s = self.spatial_proj(spatial)
        m = self.motion_proj(motion)
        l = self.landmarks_proj(landmarks)
        
        Z_cat = torch.cat([s, m, l], dim=1)  # (B, T_total, d_model)

        # Temporal convolution
        x = Z_cat.transpose(1, 2)  # (B, d_model, T_total)
        x = self.tcn(x)            # (B, d_model, T_total)
        x = x.transpose(1, 2)      # (B, T_total, d_model)

        # Final fusion
        out = self.fusion_mlp(x)   # (B, T_total, d_model)
        
        return out 