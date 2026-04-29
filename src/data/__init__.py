"""data sub-package — dataset loading and GO annotation utilities."""
from .annotations import GOAnnotationParser, GOTermAnalyzer
from .multi_chain_dataset import (
    MultiChainAnnotationParser,
    ESM2EmbeddingLoader,
    ProteinGraphDataset,
    collate_protein_graphs,
    get_dataloaders,
)
from .go_hierarchy import (
    GOHierarchy,
    HierarchicalConsistencyLoss,
    AncestorPropagator,
    download_obo,
)

__all__ = [
    "GOAnnotationParser",
    "GOTermAnalyzer",
    "MultiChainAnnotationParser",
    "ESM2EmbeddingLoader",
    "ProteinGraphDataset",
    "collate_protein_graphs",
    "get_dataloaders",
    "GOHierarchy",
    "HierarchicalConsistencyLoss",
    "AncestorPropagator",
    "download_obo",
]
