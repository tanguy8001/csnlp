import torch, torch.nn as nn, math
# -----------------------------------------------------------
# 1. Encodages de position (sinusoïdal) et de span (optionnel)
# -----------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        self.register_buffer("pe", self._build(max_len, dim), persistent=False)

    @staticmethod
    def _build(L: int, D: int) -> torch.Tensor:
        pos = torch.arange(L, dtype=torch.float32).unsqueeze(1)          # (L,1)
        div = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))
    
        pe = torch.zeros(L, D)
        pe[:, 0::2] = torch.sin(pos * div)              # even positions
    
        if D % 2 == 1:                                  # odd dimension
            pe[:, 1::2] = torch.cos(pos * div[:-1])     # one div value fewer
        else:
            pe[:, 1::2] = torch.cos(pos * div)          # even dimension
    
        return pe

    def forward(self, x):                         # x (B,T,D)
        T = x.size(1)
        if T > self.pe.size(0):                   # agrandit si besoin
            self.pe = self._build(T, self.dim).to(x.device)
        return x + self.pe[:T]


class ChunkSpanEmbedding(nn.Module):
    """
    start / end frame IDs  → concat(start_emb ⊕ end_emb)
    spans : LongTensor (B, N, 2)  [start, end]
    """
    def __init__(self, dim, max_frames=4000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        half = dim // 2
        self.start_emb = nn.Embedding(max_frames + 1, half)
        self.end_emb   = nn.Embedding(max_frames + 1, half)

    def forward(self, spans):                     # (B,N,2)
        s, e = spans[..., 0], spans[..., 1]
        return torch.cat([self.start_emb(s), self.end_emb(e)], dim=-1)


# -----------------------------------------------------------
# 2. Bloc attention (self + 2×cross en option)
# -----------------------------------------------------------
class FusionBlock(nn.Module):
    def __init__(self, spa_dim, flow_dim=None, lmk_dim=None, nhead=8):
        super().__init__()
        self.self_attn = nn.TransformerEncoderLayer(spa_dim, nhead, batch_first=True)

        # flow cross-attention (only if provided)
        if flow_dim is not None:
            self.cross_flow = nn.MultiheadAttention(
                embed_dim=spa_dim, num_heads=nhead,
                kdim=flow_dim, vdim=flow_dim, batch_first=True
            )
        else:
            self.cross_flow = None

        # landmark cross-attention (only if provided)
        if lmk_dim is not None:
            self.cross_lmk = nn.MultiheadAttention(
                embed_dim=spa_dim, num_heads=nhead,
                kdim=lmk_dim, vdim=lmk_dim, batch_first=True
            )
        else:
            self.cross_lmk = None

    def forward(self, spa, flow=None, lmk=None):
        x = self.self_attn(spa)

        if self.cross_flow is not None and flow is not None:
            x, _ = self.cross_flow(query=x, key=flow, value=flow)

        if self.cross_lmk is not None and lmk is not None:
            x, _ = self.cross_lmk(query=x, key=lmk, value=lmk)

        return x


# -----------------------------------------------------------
# 3. Fusion complète (flow-centric)
# -----------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """
    spatial_feats  : (B, T, D_spa)   ← central tokens to refine
    flow_feats     : (B, N, D_flow)  ← optional context
    landmark_feats : (B, L, D_lmk)   ← optional context
    spans          : (B, N, 2)       ← optional [start,end] for each flow chunk
    returns        : (B, T, d_model)
    """
    def __init__(self,
                 spatial_dim   : int,                 # D_spa
                 flow_dim      : int | None = None,   # D_flow
                 landmark_dim  : int | None = None,   # D_lmk
                 d_model       : int = 2048,          # → LLM hidden
                 use_span      : bool = False,
                 nhead         : int = 8,
                 num_blocks    : int = 2,
                 max_len       : int = 1024,
                 max_frames    : int = 4000):
        super().__init__()

        self.use_flow = flow_dim is not None
        self.use_lmk  = landmark_dim is not None
        self.use_span = use_span

        self.pos_spa = SinusoidalPE(spatial_dim, max_len)
        self.pos_flow = SinusoidalPE(flow_dim, max_len) if self.use_flow else None
        self.pos_lmk  = SinusoidalPE(landmark_dim, max_len) if self.use_lmk else None
        if self.use_span and self.use_flow:
            self.span_enc = ChunkSpanEmbedding(flow_dim, max_frames)

        self.blocks = nn.ModuleList([
            FusionBlock(spatial_dim, flow_dim if self.use_flow else None,
                        landmark_dim if self.use_lmk else None, nhead)
            for _ in range(num_blocks)
        ])

        # final projection into LLM hidden size
        self.to_llm = nn.Linear(spatial_dim, d_model)


    def forward(self,
                spatial_feats : torch.Tensor,          # (B,T,D_spa)
                flow_feats    : torch.Tensor | None = None,
                landmark_feats: torch.Tensor | None = None,
                spans         : torch.Tensor | None = None,
               ) -> torch.Tensor:                       # (B,T,d_model)

        # 1. positional encodings
        s = self.pos_spa(spatial_feats)

        f = None
        if self.use_flow and flow_feats is not None:
            f = self.pos_flow(flow_feats)
            if self.use_span and spans is not None:
                f = f + self.span_enc(spans)

        l = None
        if self.use_lmk and landmark_feats is not None:
            l = self.pos_lmk(landmark_feats)

        # 2. stacked refinement (spatial-centric)
        for blk in self.blocks:
            s = blk(s, flow=f, lmk=l)

        # 3. map to LLM hidden size
        return self.to_llm(s)