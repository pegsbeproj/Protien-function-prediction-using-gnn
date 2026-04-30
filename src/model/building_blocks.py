

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GCNConv,
    LayerNorm,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import softmax


# ---------------------------------------------------------------------------
# Amino-acid vocabulary constants
# ---------------------------------------------------------------------------
AMINO_ACIDS: list[str] = list("ARNDCEQGHILKMFPSTWYV")
AA_TO_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AMINO_ACIDS: int = 20

NODE_DIM: int = 40       
EDGE_DIM: int = 5        
HIDDEN_DIM: int = 192
NUM_LAYERS: int = 4
NUM_HEADS: int = 6
DROPOUT: float = 0.2
LABEL_EMB_DIM: int = 192
CHAIN_EMB_DIM: int = 8
MAX_CHAINS: int = 64
ESM2_DIM: int = 1280     


# ===========================================================================
#  Residue-Level Pooling
# ===========================================================================

class GatedAttentionPool(nn.Module):
    

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        a = softmax(self.attn(x), batch)   

        g = self.gate(x)                   

        return global_add_pool(a * g * x, batch)  


# ===========================================================================
#  Message-Passing Block
# ===========================================================================

class DualConvBlock(nn.Module):
    def __init__(self, dim: int, heads: int, edge_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.gcn = GCNConv(dim, dim)
        self.bn  = BatchNorm(dim)

        self.gat = GATv2Conv(
            dim, dim // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            edge_dim=edge_dim,
        )
        self.ln   = LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # GCN branch
        h = self.gcn(x, edge_index)
        h = self.bn(h)
        h = F.gelu(h)

        # GATv2 branch
        h = self.gat(h, edge_index, edge_attr=edge_attr)
        h = self.ln(h)
        h = F.gelu(h)
        h = self.drop(h)

        # Residual connection (skip from block input)
        return h + x


# ===========================================================================
#  Classification Head
# ===========================================================================

class LabelEmbeddingHead(nn.Module):

    def __init__(
        self,
        in_dim: int,
        n_labels: int,
        emb_dim: int = 192,
        init_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        if init_embeddings is not None:
            assert init_embeddings.shape == (n_labels, emb_dim), (
                f"init_embeddings shape {init_embeddings.shape} != ({n_labels}, {emb_dim})"
            )
            self.label_emb = nn.Parameter(torch.from_numpy(init_embeddings))
        else:
            self.label_emb = nn.Parameter(torch.randn(n_labels, emb_dim) * 0.02)

        self.label_bias = nn.Parameter(torch.zeros(n_labels))
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = F.normalize(self.proj(x), dim=-1)           
        l = F.normalize(self.label_emb, dim=-1)         

        tau = self.temp.abs().clamp(min=0.01)

        return torch.matmul(p, l.T) / tau + self.label_bias   

    def get_label_embeddings(self) -> torch.Tensor:
        """Return L2-normalised label embeddings (detached)."""
        return F.normalize(self.label_emb, dim=-1).detach()


# ===========================================================================
#  ESM2 Gated Fusion
# ===========================================================================

class ESM2GatedFusion(nn.Module):


    def __init__(self, esm_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        # Project ESM2 high-dim → hidden dim
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Gate: concat([handcraft, esm_proj]) → hidden
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        # Learnable global ESM scale (scalar) — starts at 0.5
        self.esm_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        handcraft: torch.Tensor,
        esm_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if esm_emb is None:
            # No ESM2 available → v10 fallback (identity)
            return handcraft

        # Step 1: Project ESM2 → hidden space
        esm_h = self.esm_proj(esm_emb)                              # [N, hidden]

        # Step 2: Compute per-residue, per-feature gate
        gate  = self.gate_net(torch.cat([handcraft, esm_h], dim=-1))  # [N, hidden]

        # Step 3: Soft blend (clamp scale away from zero)
        scale = self.esm_scale.clamp(0.01, 1.0)
        return gate * (esm_h * scale) + (1.0 - gate) * handcraft    # [N, hidden]


# ===========================================================================
#  Chain-Level Pooling (v13 Novel Contributions)
# ===========================================================================

class ChainAttentionPool(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor, chain_batch: torch.Tensor) -> torch.Tensor:
        # Attention weights normalised within each chain
        a = softmax(self.attn(x), chain_batch)   # [N, 1]
        g = self.gate(x)                          # [N, dim]
        return global_add_pool(a * g * x, chain_batch)


class ProteinChainPool(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(
        self,
        chain_embs: torch.Tensor,
        chain_to_graph: torch.Tensor,
    ) -> torch.Tensor:
        a = softmax(self.attn(chain_embs), chain_to_graph)   # [num_chains, 1]
        g = self.gate(chain_embs)                             # [num_chains, dim]
        return global_add_pool(a * g * chain_embs, chain_to_graph)  # [B, dim]


class CCContextAttention(nn.Module):

    def __init__(self, protein_dim: int, chain_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(protein_dim, chain_dim)
        self.key_proj   = nn.Linear(chain_dim,   chain_dim)
        self.value_proj = nn.Linear(chain_dim,   chain_dim)
        self.out_proj   = nn.Sequential(
            nn.Linear(chain_dim, chain_dim),
            nn.LayerNorm(chain_dim),
            nn.GELU(),
        )

    def forward(
        self,
        protein_emb: torch.Tensor,
        chain_embs: torch.Tensor,
        chain_to_graph: torch.Tensor,
    ) -> torch.Tensor:
        
        # Broadcast protein query to every chain of the same protein
        q = self.query_proj(protein_emb[chain_to_graph])  # [num_chains, chain_dim]
        k = self.key_proj(chain_embs)                     # [num_chains, chain_dim]
        v = self.value_proj(chain_embs)                   # [num_chains, chain_dim]

        # Scaled dot-product score
        scale  = k.size(-1) ** 0.5
        score  = (q * k).sum(dim=-1, keepdim=True) / scale   # [num_chains, 1]
        attn   = softmax(score, chain_to_graph)               # [num_chains, 1]

        # Aggregate per protein
        context = global_add_pool(attn * v, chain_to_graph)  # [B, chain_dim]
        return self.out_proj(context)
