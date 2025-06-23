import torch
import torch.nn as nn

class DynamicChunker(nn.Module):
    def __init__(self, dim, max_chunk_len=32, nhead=4):
        super().__init__()
        self.dim = dim
        self.max_chunk_len = max_chunk_len

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            batch_first=True
        )

        # Predict end‐of‐chunk probability
        self.eoc_classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, frames, threshold=0.5):
        """
        Args:
            frames: Tensor of shape (B, T, D)
            threshold: cutoff for end‐of‐chunk
        Returns:
            Tensor of shape (B, max_chunks, D)
        """
        B, T, D = frames.shape
        device = frames.device
        all_chunks = []

        for b in range(B):
            chunks = []
            start = 0

            # Slide until we've covered all T frames
            while start < T:
                # Grow the window until EoC or max length
                for length in range(1, min(self.max_chunk_len, T - start) + 1):
                    window = frames[b, start:start + length]                 # (length, D)
                    x = torch.cat([
                        self.cls_token.to(device),                           # (1,1,D)
                        window.unsqueeze(0)                                  # (1,length,D)
                    ], dim=1)                                                # (1, length+1, D)

                    # Build attention mask: allow full within‐window attention
                    L = length + 1
                    mask = torch.zeros(L, L, device=device).float()         # no masking

                    out = self.encoder(x, src_mask=mask)                   # (1, L, D)
                    cls_out = out[:, 0]                                     # (1, D)
                    p_eoc = self.eoc_classifier(cls_out).item()

                    # If end‐of‐chunk predicted or reached max length
                    if p_eoc >= threshold or length == min(self.max_chunk_len, T - start):
                        chunks.append(cls_out.squeeze(0))  # (D,)
                        start += length
                        break

            all_chunks.append(torch.stack(chunks, dim=0))  # (n_chunks_b, D)

        # Pad to the same number of chunks
        max_chunks = max(c.size(0) for c in all_chunks)
        padded = []
        for c in all_chunks:
            n = c.size(0)
            if n < max_chunks:
                pad = c.new_zeros(max_chunks - n, D)
                c = torch.cat([c, pad], dim=0)
            padded.append(c)
        return torch.stack(padded, dim=0)  # (B, max_chunks, D)
