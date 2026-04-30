
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.nn import global_max_pool, global_mean_pool

from .building_blocks import (
    CHAIN_EMB_DIM,
    DROPOUT,
    ESM2_DIM,
    HIDDEN_DIM,
    LABEL_EMB_DIM,
    MAX_CHAINS,
    NODE_DIM,
    EDGE_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    CCContextAttention,
    ChainAttentionPool,
    DualConvBlock,
    ESM2GatedFusion,
    GatedAttentionPool,
    LabelEmbeddingHead,
    ProteinChainPool,
)


class ProteinGNN(nn.Module):

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden: int = HIDDEN_DIM,
        n_mf: int = 489,
        n_bp: int = 1943,
        n_cc: int = 320,
        n_layers: int = NUM_LAYERS,
        heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
        use_label_embed: bool = True,
        label_emb_dim: int = LABEL_EMB_DIM,
        chain_emb_dim: int = CHAIN_EMB_DIM,
        max_chains: int = MAX_CHAINS,
        esm_dim: int = ESM2_DIM,
        use_chain_pool: bool = True,
        use_gradient_checkpointing: bool = True,
        init_embeddings_mf: Optional[np.ndarray] = None,
        init_embeddings_bp: Optional[np.ndarray] = None,
        init_embeddings_cc: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        # ── Flags ────────────────────────────────────────────────────────────
        self.use_label_embed             = use_label_embed
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_chain_pool              = use_chain_pool
        self.hidden                      = hidden
        self.chain_emb_dim               = chain_emb_dim
        self.esm_dim                     = esm_dim
        self.max_chains                  = max_chains

        # ── Serialisable configuration (for checkpoint round-trips) ──────────
        self.config: Dict = {
            "node_dim":        node_dim,
            "edge_dim":        edge_dim,
            "hidden":          hidden,
            "n_mf":            n_mf,
            "n_bp":            n_bp,
            "n_cc":            n_cc,
            "n_layers":        n_layers,
            "heads":           heads,
            "dropout":         dropout,
            "use_label_embed": use_label_embed,
            "label_emb_dim":   label_emb_dim,
            "chain_emb_dim":   chain_emb_dim,
            "max_chains":      max_chains,
            "esm_dim":         esm_dim,
            "use_chain_pool":  use_chain_pool,
            "version":         "v13",
        }

        # ── Stage 1: Encoding ────────────────────────────────────────────────

        # Chain identity embedding (learned discrete lookup)
        self.chain_embedding = nn.Embedding(max_chains, chain_emb_dim)

        # Node encoder: [node_dim + chain_emb_dim] → hidden
        enc_input_dim = node_dim + chain_emb_dim
        self.node_enc = nn.Sequential(
            nn.Linear(enc_input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ESM2 gated fusion (optional; falls back to structural features only)
        self.esm_fusion = ESM2GatedFusion(esm_dim, hidden, dropout)

        # Edge encoder: edge_dim → hidden  (used as edge_attr by GATv2)
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.GELU(),
        )

        # ── Stage 2: Message Passing ─────────────────────────────────────────

        self.blocks = nn.ModuleList([
            DualConvBlock(hidden, heads, hidden, dropout)
            for _ in range(n_layers)
        ])

        # ── Stage 3a: Residue-Level Triple Pooling (Branch 1) ───────────────

        self.gated_pool = GatedAttentionPool(hidden)
        self.pool_fuse  = nn.Sequential(
            nn.Linear(hidden * 3, hidden),   # gated + mean + max → hidden
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Stage 3b/c: Chain-Aware Hierarchical Pooling (Branch 2) ─────────

        if use_chain_pool:
            # Residues → chain embeddings
            self.chain_attn_pool   = ChainAttentionPool(hidden)
            # Chain embeddings → protein embedding
            self.protein_chain_pool = ProteinChainPool(hidden)
            # Linear projection for chain branch output
            self.chain_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # CC-specific cross-attention (protein ← chains)
            self.cc_context_attn = CCContextAttention(
                protein_dim=hidden * 2,   # merged: residue_emb ‖ chain_emb
                chain_dim=hidden,
            )
            # Dimension of merged embedding (MF/BP heads)
            mf_bp_in_dim = hidden * 2   # 384
            # Dimension of CC input (extra chain context)
            cc_in_dim    = hidden * 3   # 576
        else:
            # Chain pooling disabled → identical to v11 architecture
            mf_bp_in_dim = hidden
            cc_in_dim    = hidden

        # ── Stage 4: Prediction Heads ────────────────────────────────────────

        if use_label_embed:
            self.head_mf = LabelEmbeddingHead(mf_bp_in_dim, n_mf, label_emb_dim, init_embeddings_mf)
            self.head_bp = LabelEmbeddingHead(mf_bp_in_dim, n_bp, label_emb_dim, init_embeddings_bp)
            self.head_cc = LabelEmbeddingHead(cc_in_dim,    n_cc, label_emb_dim, init_embeddings_cc)
        else:
            self.head_mf = self._make_mlp_head(mf_bp_in_dim, n_mf, dropout)
            self.head_bp = self._make_mlp_head(mf_bp_in_dim, n_bp, dropout)
            self.head_cc = self._make_mlp_head(cc_in_dim,    n_cc, dropout)

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_mlp_head(in_dim: int, n_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, n_classes),
        )

    def _build_chain_mapping(
        self,
        batch: torch.Tensor,
        chain_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clamp chain indices to valid range
        chain_idx_clamped = chain_idx.clamp(0, self.max_chains - 1)
        # Encode (graph, chain) pair as a single integer
        pair = batch * self.max_chains + chain_idx_clamped
        unique_pairs, chain_batch = torch.unique(pair, return_inverse=True)
        # Recover graph index from pair encoding
        chain_to_graph = unique_pairs.div(self.max_chains, rounding_mode="floor")
        return chain_batch, chain_to_graph

    def _encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        chain_idx: Optional[torch.Tensor],
        esm_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Step 1: Attach chain identity embedding to node features
        if chain_idx is not None:
            ci = chain_idx.clamp(0, self.max_chains - 1)
        else:
            ci = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        chain_emb = self.chain_embedding(ci)              # [N, chain_emb_dim]
        x = torch.cat([x, chain_emb], dim=-1)            # [N, 48]

        # Step 2: Encode to hidden dimension
        h = self.node_enc(x)                             # [N, hidden]

        # Step 3: Gated fusion with ESM2 (no-op when esm_emb is None)
        h = self.esm_fusion(h, esm_emb)                 # [N, hidden]

        # Step 4: Encode edge features
        if edge_attr is not None:
            e = self.edge_enc(edge_attr)                  # [E, hidden]
        else:
            e = torch.zeros(edge_index.size(1), self.hidden, device=h.device)

        # Step 5: Stacked dual-conv message-passing blocks
        for block in self.blocks:
            if self.training and self.use_gradient_checkpointing and torch.is_grad_enabled():
                h = grad_checkpoint(block, h, edge_index, e, use_reentrant=False)
            else:
                h = block(h, edge_index, e)

        return h

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        chain_idx: Optional[torch.Tensor] = None,
        esm_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── GNN backbone ─────────────────────────────────────────────────────
        h = self._encode_nodes(x, edge_index, edge_attr, chain_idx, esm_emb)

        # ── Branch 1: Residue-level triple pooling ────────────────────────────
        gp = self.gated_pool(h, batch)          # gated attention pool
        mp = global_mean_pool(h, batch)          # mean pool
        xp = global_max_pool(h, batch)           # max pool
        protein_residue_emb = self.pool_fuse(
            torch.cat([gp, mp, xp], dim=-1)
        )                                        # [B, hidden]

        # ── Branch 2: Hierarchical chain-aware pooling ────────────────────────
        if self.use_chain_pool and chain_idx is not None:
            chain_batch, chain_to_graph = self._build_chain_mapping(batch, chain_idx)

            # Residues → chain embeddings
            chain_embs = self.chain_attn_pool(h, chain_batch)          # [C, hidden]
            # Chain embeddings → protein embedding
            protein_chain_emb = self.protein_chain_pool(chain_embs, chain_to_graph)
            protein_chain_emb = self.chain_proj(protein_chain_emb)     # [B, hidden]

            # Merge branches
            protein_emb = torch.cat(
                [protein_residue_emb, protein_chain_emb], dim=-1
            )                                                           # [B, 2·hidden]

            # CC-specific cross-attention context
            cc_context = self.cc_context_attn(protein_emb, chain_embs, chain_to_graph)
            cc_input   = torch.cat([protein_emb, cc_context], dim=-1)  # [B, 3·hidden]

        elif self.use_chain_pool:
            # chain_idx is None but model expects chain pooling → zero-pad
            zeros       = torch.zeros_like(protein_residue_emb)
            protein_emb = torch.cat([protein_residue_emb, zeros], dim=-1)
            cc_input    = torch.cat([protein_emb, zeros], dim=-1)

        else:
            # Chain pooling disabled → v11-equivalent single-branch
            protein_emb = protein_residue_emb   # [B, hidden]
            cc_input    = protein_residue_emb   # [B, hidden]

        # ── Ontology-specific prediction heads ────────────────────────────────
        mf_logits = self.head_mf(protein_emb)   # [B, n_mf]
        bp_logits = self.head_bp(protein_emb)   # [B, n_bp]
        cc_logits = self.head_cc(cc_input)      # [B, n_cc]

        return mf_logits, bp_logits, cc_logits

    # ── Utility / monitoring methods ─────────────────────────────────────────

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        chain_idx: Optional[torch.Tensor] = None,
        esm_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the merged protein-level embedding (before prediction heads)."""
        h = self._encode_nodes(x, edge_index, edge_attr, chain_idx, esm_emb)
        gp = self.gated_pool(h, batch)
        mp = global_mean_pool(h, batch)
        xp = global_max_pool(h, batch)
        protein_residue_emb = self.pool_fuse(torch.cat([gp, mp, xp], dim=-1))

        if self.use_chain_pool and chain_idx is not None:
            chain_batch, chain_to_graph = self._build_chain_mapping(batch, chain_idx)
            chain_embs            = self.chain_attn_pool(h, chain_batch)
            protein_chain_emb     = self.protein_chain_pool(chain_embs, chain_to_graph)
            protein_chain_emb     = self.chain_proj(protein_chain_emb)
            return torch.cat([protein_residue_emb, protein_chain_emb], dim=-1)
        elif self.use_chain_pool:
            zeros = torch.zeros_like(protein_residue_emb)
            return torch.cat([protein_residue_emb, zeros], dim=-1)
        else:
            return protein_residue_emb

    def get_embedding_regularization_loss(self) -> torch.Tensor:
        """L2 norm of label embeddings (used for co-occurrence regularisation)."""
        if self.use_label_embed:
            return (
                torch.norm(self.head_mf.label_emb, p=2)
                + torch.norm(self.head_bp.label_emb, p=2)
                + torch.norm(self.head_cc.label_emb, p=2)
            ) / 3.0
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_esm_gate_stats(self) -> Dict[str, float]:
        """Return ESM2 gate statistics for training-time monitoring."""
        linear = self.esm_fusion.gate_net[0]
        bias   = linear.bias.detach()
        return {
            "gate_bias_mean": bias.mean().item(),
            "gate_bias_std":  bias.std().item(),
            "esm_scale":      self.esm_fusion.esm_scale.item(),
        }

    def get_chain_pool_stats(self) -> Dict[str, float]:
        if not self.use_chain_pool:
            return {}
        chain_gate = self.chain_attn_pool.gate[0]
        prot_gate  = self.protein_chain_pool.gate[0]
        return {
            "chain_gate_bias_mean":   chain_gate.bias.detach().mean().item(),
            "chain_gate_bias_std":    chain_gate.bias.detach().std().item(),
            "protein_gate_bias_mean": prot_gate.bias.detach().mean().item(),
            "protein_gate_bias_std":  prot_gate.bias.detach().std().item(),
        }


# ===========================================================================
#  Factory Functions
# ===========================================================================

def create_model(
    n_mf: int = 489,
    n_bp: int = 1943,
    n_cc: int = 320,
    vram_gb: float = 8.0,
    esm_dim: int = ESM2_DIM,
    use_label_embed: bool = True,
    use_chain_pool: bool = True,
    init_embeddings_mf: Optional[np.ndarray] = None,
    init_embeddings_bp: Optional[np.ndarray] = None,
    init_embeddings_cc: Optional[np.ndarray] = None,
) -> ProteinGNN:
    if vram_gb >= 8.0:
        cfg = {"hidden": 192, "n_layers": 4, "heads": 6, "dropout": 0.2, "label_emb_dim": 192}
    elif vram_gb >= 6.0:
        cfg = {"hidden": 160, "n_layers": 4, "heads": 4, "dropout": 0.2, "label_emb_dim": 160}
    else:
        cfg = {"hidden": 128, "n_layers": 3, "heads": 4, "dropout": 0.25, "label_emb_dim": 128}

    return ProteinGNN(
        n_mf=n_mf,
        n_bp=n_bp,
        n_cc=n_cc,
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
        use_label_embed=use_label_embed,
        label_emb_dim=cfg["label_emb_dim"],
        chain_emb_dim=CHAIN_EMB_DIM,
        max_chains=MAX_CHAINS,
        esm_dim=esm_dim,
        use_chain_pool=use_chain_pool,
        use_gradient_checkpointing=True,
        init_embeddings_mf=init_embeddings_mf,
        init_embeddings_bp=init_embeddings_bp,
        init_embeddings_cc=init_embeddings_cc,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_layer_parameters(model: ProteinGNN) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    def _count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    counts["chain_embedding"]  = _count(model.chain_embedding)
    counts["node_encoder"]     = _count(model.node_enc)
    counts["esm_fusion"]       = _count(model.esm_fusion)
    counts["edge_encoder"]     = _count(model.edge_enc)
    counts["dual_conv_blocks"] = _count(model.blocks)
    counts["gated_pool"]       = _count(model.gated_pool)
    counts["pool_fuse"]        = _count(model.pool_fuse)

    if model.use_chain_pool:
        counts["chain_attn_pool"]    = _count(model.chain_attn_pool)
        counts["protein_chain_pool"] = _count(model.protein_chain_pool)
        counts["chain_proj"]         = _count(model.chain_proj)
        counts["cc_context_attn"]    = _count(model.cc_context_attn)

    counts["head_mf"] = _count(model.head_mf)
    counts["head_bp"] = _count(model.head_bp)
    counts["head_cc"] = _count(model.head_cc)

    return counts


# ===========================================================================
#  Self-Test
# ===========================================================================

if __name__ == "__main__":
    from building_blocks import NODE_DIM, EDGE_DIM

    print("Self-test: ProteinGNN (Hierarchical Chain-Aware Pooling)")
    print("=" * 65)

    model = create_model(vram_gb=8.0, use_chain_pool=True)
    model.eval()

    total = count_parameters(model)
    print(f"Total parameters: {total:,}")
    for name, cnt in count_layer_parameters(model).items():
        print(f"  {name}: {cnt:,} ({100*cnt/total:.1f}%)")

    # Synthetic mini-batch (4 proteins, 2 chains each, 25 residues each)
    B, R, H = 4, 25, 5
    N = B * R
    x          = torch.randn(N, NODE_DIM)
    edge_index = torch.randint(0, N, (2, N * H))
    edge_attr  = torch.randn(N * H, EDGE_DIM)
    batch      = torch.repeat_interleave(torch.arange(B), R)
    chain_idx  = torch.cat([
        torch.cat([torch.zeros(R // 2, dtype=torch.long),
                   torch.ones(R - R // 2, dtype=torch.long)])
        for _ in range(B)
    ])
    esm_emb    = torch.randn(N, 1280)

    with torch.no_grad():
        mf, bp, cc = model(x, edge_index, batch, edge_attr, chain_idx=chain_idx, esm_emb=esm_emb)

    print(f"\nForward pass OK:")
    print(f"  MF logits: {tuple(mf.shape)}")
    print(f"  BP logits: {tuple(bp.shape)}")
    print(f"  CC logits: {tuple(cc.shape)}")
    print("Self-test passed ✓")
