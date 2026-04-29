"""model sub-package — ProteinGNN architecture and building blocks."""
from .protein_gnn import ProteinGNN, create_model, count_parameters, count_layer_parameters
from .building_blocks import (
    GatedAttentionPool,
    DualConvBlock,
    LabelEmbeddingHead,
    ESM2GatedFusion,
    ChainAttentionPool,
    ProteinChainPool,
    CCContextAttention,
)

__all__ = [
    "ProteinGNN",
    "create_model",
    "count_parameters",
    "count_layer_parameters",
    "GatedAttentionPool",
    "DualConvBlock",
    "LabelEmbeddingHead",
    "ESM2GatedFusion",
    "ChainAttentionPool",
    "ProteinChainPool",
    "CCContextAttention",
]
