"""
building_blocks.py — Reusable GNN Components for Protein Function Prediction

Contains the neural-network primitives shared across model versions:
  - GatedAttentionPool   : single-head gated attention pooling over a graph
  - DualConvBlock        : parallel GCN + GATv2 message-passing block with residual
  - LabelEmbeddingHead   : cosine-similarity classification head with learnable label embeddings
  - ESM2GatedFusion      : gated soft-blend of handcrafted and ESM2 residue features
  - ChainAttentionPool   : gated attention pooling of residues within each protein chain
  - ProteinChainPool     : gated attention pooling of chain embeddings into a protein vector
  - CCContextAttention   : cross-attention from the protein vector to per-chain vectors
                           (used exclusively by the Cellular Component prediction head)

Design principles:
  - Each module is self-contained and independently testable.
  - All forward() signatures are documented with tensor shapes.
  - Modules degrade gracefully when optional inputs are None.
"""

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

# Default feature dimensions (shared across all model versions ≥ v11)
NODE_DIM: int = 40       # 20 one-hot + 8 physicochemical + 12 structural
EDGE_DIM: int = 5        # distance + seq_sep + backbone + local + decay
HIDDEN_DIM: int = 192
NUM_LAYERS: int = 4
NUM_HEADS: int = 6
DROPOUT: float = 0.2
LABEL_EMB_DIM: int = 192
CHAIN_EMB_DIM: int = 8
MAX_CHAINS: int = 64
ESM2_DIM: int = 1280     # esm2_t33_650M_UR50D default output size


# ===========================================================================
#  Residue-Level Pooling
# ===========================================================================

class GatedAttentionPool(nn.Module):
    """
    Single-head gated attention pooling for graph-level aggregation.

    Computes:
        a_i = softmax_batch( Linear(x_i) )       # attention weights
        g_i = sigmoid( Linear(x_i) )             # per-feature gate
        output = global_add_pool( a_i * g_i * x_i )

    Args:
        dim: Feature dimension of input node embeddings.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [N, dim]  node embeddings
            batch: [N]       graph index per node

        Returns:
            [B, dim]  pooled graph-level embeddings
        """
        # Step 1: Compute per-node attention scores, normalised within each graph
        a = softmax(self.attn(x), batch)   # [N, 1]

        # Step 2: Compute per-node, per-feature gating values
        g = self.gate(x)                   # [N, dim]

        # Step 3: Weighted aggregation
        return global_add_pool(a * g * x, batch)  # [B, dim]


# ===========================================================================
#  Message-Passing Block
# ===========================================================================

class DualConvBlock(nn.Module):
    """
    Parallel GCN + GATv2 message-passing block with residual connection.

    GCN branch:   cheap, symmetric neighbourhood averaging (global topology).
    GATv2 branch: multi-head attention weighted by edge features (local detail).

    Both outputs are added element-wise, then a skip connection from the input
    is added before LayerNorm — preventing vanishing gradients across 4 stacked
    blocks.

    Args:
        dim:      Hidden dimension (same for input and output).
        heads:    Number of GATv2 attention heads.
        edge_dim: Edge feature dimension fed to GATv2.
        dropout:  Dropout probability applied after LayerNorm.
    """

    def __init__(self, dim: int, heads: int, edge_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        # GCN — symmetric mean aggregation
        self.gcn = GCNConv(dim, dim)
        self.bn  = BatchNorm(dim)

        # GATv2 — per-edge attention scores
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
        """
        Args:
            x:          [N, dim]   node features
            edge_index: [2, E]     edge connectivity
            edge_attr:  [E, edge_dim] edge features

        Returns:
            [N, dim]  updated node features
        """
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
    """
    GO-term classification via cosine similarity in a shared embedding space.

    Projects the protein embedding and each GO-term embedding into a joint
    L2-normalised space, then computes dot-product similarity scores (scaled
    by a learnable temperature τ) and adds a per-class bias.

    This is equivalent to a linear head but with explicit label embeddings that
    can be pre-initialised from co-occurrence SVD.

    Args:
        in_dim:           Protein embedding dimension.
        n_labels:         Number of GO classes.
        emb_dim:          Shared embedding space dimension.
        init_embeddings:  Optional [n_labels, emb_dim] numpy array for warm-start.
    """

    def __init__(
        self,
        in_dim: int,
        n_labels: int,
        emb_dim: int = 192,
        init_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        # Two-layer protein projection into shared space
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Learnable label embeddings (warm-start from SVD if provided)
        if init_embeddings is not None:
            assert init_embeddings.shape == (n_labels, emb_dim), (
                f"init_embeddings shape {init_embeddings.shape} != ({n_labels}, {emb_dim})"
            )
            self.label_emb = nn.Parameter(torch.from_numpy(init_embeddings))
        else:
            self.label_emb = nn.Parameter(torch.randn(n_labels, emb_dim) * 0.02)

        self.label_bias = nn.Parameter(torch.zeros(n_labels))
        # Temperature τ = exp(log_temp) — keeps τ > 0
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim]  batch of protein embeddings

        Returns:
            [B, n_labels]  per-class logits
        """
        # L2-normalise both sides
        p = F.normalize(self.proj(x), dim=-1)           # [B, emb_dim]
        l = F.normalize(self.label_emb, dim=-1)         # [n_labels, emb_dim]

        # Clamp temperature away from zero for numerical stability
        tau = self.temp.abs().clamp(min=0.01)

        return torch.matmul(p, l.T) / tau + self.label_bias   # [B, n_labels]

    def get_label_embeddings(self) -> torch.Tensor:
        """Return L2-normalised label embeddings (detached)."""
        return F.normalize(self.label_emb, dim=-1).detach()


# ===========================================================================
#  ESM2 Gated Fusion
# ===========================================================================

class ESM2GatedFusion(nn.Module):
    """
    Learned per-residue gating between handcrafted and ESM2 node features.

    For each residue, a sigmoid gate g ∈ (0,1)^hidden learns how much to
    trust ESM2 versus the handcrafted structural features:

        fused = g ⊙ (esm_proj * scale) + (1 − g) ⊙ handcraft

    When esm_emb is None, returns handcraft unchanged (v10-compatible fallback).

    Args:
        esm_dim:    Dimension of raw ESM2 embeddings (default 1 280).
        hidden_dim: Hidden dimension to project ESM2 features into.
        dropout:    Dropout on the ESM2 projection.
    """

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
        """
        Args:
            handcraft: [N, hidden]   encoded handcrafted node features
            esm_emb:   [N, esm_dim] raw ESM2 per-residue embeddings, or None

        Returns:
            [N, hidden]  fused node representations
        """
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
    """
    Gated attention pooling of residue embeddings *within* each protein chain.

    Produces one embedding vector per chain:
        chain_emb_c = Σ_i  α_i · gate(x_i) · x_i
    where α_i = softmax_{i ∈ chain c}( Linear(x_i) ).

    This captures which residues are most important for characterising each
    chain, preserving chain-level identity that global pooling discards.

    Args:
        dim: Feature dimension of residue embeddings.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor, chain_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           [N, dim]  residue embeddings after GNN message passing
            chain_batch: [N]       consecutive chain index per residue
                                   (one unique integer per (graph_id, chain_id) pair)

        Returns:
            [num_chains, dim]  per-chain embeddings
        """
        # Attention weights normalised within each chain
        a = softmax(self.attn(x), chain_batch)   # [N, 1]
        g = self.gate(x)                          # [N, dim]
        return global_add_pool(a * g * x, chain_batch)


class ProteinChainPool(nn.Module):
    """
    Gated attention pooling of chain embeddings into a single protein vector.

    Produces one embedding vector per protein in the batch:
        protein_emb_p = Σ_c  β_c · gate(chain_emb_c) · chain_emb_c
    where β_c = softmax_{c ∈ protein p}( Linear(chain_emb_c) ).

    Learns which chains carry the most predictive signal for each protein.

    Args:
        dim: Feature dimension of chain embeddings.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(
        self,
        chain_embs: torch.Tensor,
        chain_to_graph: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            chain_embs:     [num_chains, dim]  per-chain embeddings
            chain_to_graph: [num_chains]       graph index per chain

        Returns:
            [B, dim]  protein-level chain-aware embedding
        """
        a = softmax(self.attn(chain_embs), chain_to_graph)   # [num_chains, 1]
        g = self.gate(chain_embs)                             # [num_chains, dim]
        return global_add_pool(a * g * chain_embs, chain_to_graph)  # [B, dim]


class CCContextAttention(nn.Module):
    """
    Cross-attention from the protein-level embedding to per-chain embeddings.

    Used exclusively by the Cellular Component (CC) prediction head because
    subcellular localisation depends on chain-specific signals (e.g. membrane-
    spanning chains, signal peptides) that the global protein vector may dilute.

    Mechanism (scaled dot-product attention, protein-as-query):
        q_c = W_q · protein_emb_p   (broadcast to each chain of protein p)
        k_c = W_k · chain_emb_c
        v_c = W_v · chain_emb_c
        score_c  = (q_c · k_c) / √d
        attn_c   = softmax_{c ∈ protein p}( score_c )
        context_p = Σ_c attn_c · v_c

    Args:
        protein_dim: Dimension of the merged protein embedding (2 × hidden).
        chain_dim:   Dimension of individual chain embeddings (hidden).
    """

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
        """
        Args:
            protein_emb:    [B, protein_dim]      merged protein representation
            chain_embs:     [num_chains, chain_dim]  per-chain embeddings
            chain_to_graph: [num_chains]           graph index per chain

        Returns:
            [B, chain_dim]  chain-aware context vector for CC prediction
        """
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
