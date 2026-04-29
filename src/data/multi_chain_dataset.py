"""
Dataset for v11 (ESM2-Enhanced Inter-Chain Geometry Aware Graphs)

v11 builds on v10 by adding ESM2 per-residue embedding support.

Key Differences from v10:
  1. ESM2 embeddings loaded alongside v10 graphs (separate files)
  2. custom_collate_v11 pads ESM2 tensors per-batch
  3. get_dataloaders_v11 accepts esm2_dir parameter
  4. Falls back gracefully if ESM2 embeddings not found for a protein

Data Flow:
  - Graphs:  loaded from graphs_dir (same v10 batch files)
  - ESM2:    loaded from esm2_dir/{PDB_ID}.pt → {'esm_emb': Tensor[N, 1280], ...}
  - Labels:  from MultiChainAnnotationParser (same as v10)

Each graph in the DataLoader will have:
  - x:        [N, 40]     node features
  - chain_idx:[N]         chain assignment
  - edge_*:   standard PyG edge tensors
  - esm_emb:  [N, esm_dim] per-residue ESM2 embeddings (or None)
  - y_mf/y_bp/y_cc: [n_classes] label vectors

Re-uses from v10:
  - MultiChainAnnotationParser (PDB-level annotation aggregation)
  - ProteinGODataset base class from dataset.py
  - Same train/val/test splitting
"""

import json
import random
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from .annotations import GOAnnotationParser
from .base_dataset import ProteinGODataset, create_data_splits
from .multi_chain_parser import MultiChainAnnotationParser


# ════════════════════════════════════════════════════════════════
#  ESM2 Embedding Loader
# ════════════════════════════════════════════════════════════════
class ESM2EmbeddingLoader:
    """
    Loads pre-computed ESM2 embeddings for PDB structures.

    Looks up {esm2_dir}/{PDB_ID}.pt files containing per-residue embeddings.
    Caches recently loaded embeddings for efficiency.
    """

    def __init__(self, esm2_dir: str, cache_size: int = 500, expected_dim: int = 1280):
        self.esm2_dir = Path(esm2_dir)
        self.expected_dim = expected_dim
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_size = cache_size
        self._miss_count = 0
        self._hit_count = 0

        # Verify directory exists
        if not self.esm2_dir.exists():
            print(f"[WARN] ESM2 embedding directory not found: {self.esm2_dir}")
            print(f"       Model will run in v10 fallback mode (no ESM2)")
            self._available = False
        else:
            # Count available embeddings
            n_files = len(list(self.esm2_dir.glob("*.pt")))
            # Check index for dim info
            index_file = self.esm2_dir / "esm2_index.json"
            if index_file.exists():
                with open(index_file) as f:
                    idx = json.load(f)
                self.expected_dim = idx.get('embedding_dim', expected_dim)
                n_indexed = len(idx.get('processed_pdbs', []))
                print(f"[v11] ESM2 embeddings: {n_files} files, {n_indexed} indexed, dim={self.expected_dim}")
            else:
                print(f"[v11] ESM2 embeddings: {n_files} files (no index), dim={self.expected_dim}")
            self._available = True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def esm_dim(self) -> int:
        return self.expected_dim

    def get(self, pdb_id: str, num_nodes: int) -> Optional[torch.Tensor]:
        """
        Get ESM2 embeddings for a PDB structure.

        Args:
            pdb_id: PDB identifier (e.g. "1A2B")
            num_nodes: Expected number of nodes in the graph

        Returns:
            Tensor [num_nodes, esm_dim] in float32, or None if not available.
            If node count mismatch, pads/truncates to match.
        """
        if not self._available:
            return None

        # Check cache (OrderedDict: move to end on access for LRU-like behavior)
        if pdb_id in self._cache:
            self._hit_count += 1
            self._cache.move_to_end(pdb_id)
            emb = self._cache[pdb_id]
            # Cache stores fp16 to halve memory; convert to fp32 on retrieval
            return self._align_to_nodes(emb.float(), num_nodes)

        # Load from disk
        emb_file = self.esm2_dir / f"{pdb_id}.pt"
        if not emb_file.exists():
            # Try uppercase
            emb_file = self.esm2_dir / f"{pdb_id.upper()}.pt"
            if not emb_file.exists():
                self._miss_count += 1
                return None

        try:
            data = torch.load(emb_file, map_location='cpu', weights_only=False)
            emb = data['esm_emb'].half()  # Cache in fp16 to halve memory

            # Cache (FIFO via OrderedDict — O(1) eviction)
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)  # evict oldest
            self._cache[pdb_id] = emb
            self._hit_count += 1

            return self._align_to_nodes(emb.float(), num_nodes)

        except Exception as e:
            self._miss_count += 1
            return None

    def _align_to_nodes(self, emb: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Ensure embedding matches graph node count."""
        if emb.shape[0] == num_nodes:
            return emb
        elif emb.shape[0] > num_nodes:
            return emb[:num_nodes]
        else:
            pad = torch.zeros(num_nodes - emb.shape[0], emb.shape[1])
            return torch.cat([emb, pad], dim=0)

    def get_stats(self) -> Dict:
        return {
            'available': self._available,
            'esm_dim': self.expected_dim,
            'cache_size': len(self._cache),
            'hits': self._hit_count,
            'misses': self._miss_count,
            'hit_rate': self._hit_count / max(1, self._hit_count + self._miss_count),
        }


# ════════════════════════════════════════════════════════════════
#  v11 Dataset: ProteinGODataset + ESM2
# ════════════════════════════════════════════════════════════════
class ProteinGODatasetV11(ProteinGODataset):
    """
    Extends ProteinGODataset to attach ESM2 embeddings.

    On get(idx):
      1. Loads graph from v10 batch files (via parent class)
      2. Attaches GO labels (via parent class)
      3. Loads ESM2 embedding from esm2_dir/{graph_id}.pt
      4. Sets graph.esm_emb = Tensor[N, esm_dim] or None
    """

    def __init__(
        self,
        graphs_dir: str,
        annotation_parser,
        graph_ids: Optional[List[str]] = None,
        esm2_loader: Optional[ESM2EmbeddingLoader] = None,
        transform=None,
        pre_transform=None,
        max_nodes: int = None,
        max_edges: int = None,
    ):
        self.esm2_loader = esm2_loader
        super().__init__(
            graphs_dir, annotation_parser, graph_ids,
            transform, pre_transform, max_nodes, max_edges
        )
        # Override parent's cache size: ESM2 embeddings make each cached
        # graph ~1.3 MB (250 residues × 1280 × 4 bytes).  At 2000 graphs
        # that's 2.6 GB per dataset × 3 datasets = 7.8 GB system RAM.
        # With fp16 ESM2 caching this is halved. Increase cache for
        # fewer disk reads (RAM budget is generous).
        self._cache_size = 500
        self._batch_cache_size = 8  # Up from 3; reduces repeated disk I/O

    def get(self, idx: int) -> Data:
        """Get a graph with GO labels and ESM2 embeddings."""
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]

        # Load graph from parent (v10 batch files + labels)
        batch_file, batch_idx, graph_id = self._graph_locations[idx]

        batch_key = str(batch_file)
        if batch_key in self._batch_cache:
            graphs = self._batch_cache[batch_key]
        else:
            if batch_file == 'subset':
                graphs = torch.load(self.graphs_dir / 'subset_graphs.pt', weights_only=False)
            else:
                graphs = torch.load(batch_file, weights_only=False)
            if len(self._batch_cache) >= self._batch_cache_size:
                oldest_key = next(iter(self._batch_cache))
                del self._batch_cache[oldest_key]
            self._batch_cache[batch_key] = graphs

        graph = graphs[batch_idx]

        # Attach GO labels
        mf_labels, bp_labels, cc_labels = self.parser.get_labels(graph_id)
        graph.y_mf = mf_labels
        graph.y_bp = bp_labels
        graph.y_cc = cc_labels
        graph.graph_id = graph_id

        # Attach ESM2 embeddings (NEW in v11)
        if self.esm2_loader is not None:
            esm_emb = self.esm2_loader.get(graph_id, graph.x.shape[0])
            if esm_emb is not None:
                graph.esm_emb = esm_emb
            # If None, graph won't have esm_emb attr → model falls back to v10

        # Update cache (FIFO)
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[idx] = graph

        return graph


# ════════════════════════════════════════════════════════════════
#  Custom Collation for v11
# ════════════════════════════════════════════════════════════════
def custom_collate_v11(data_list):
    """
    Custom collate for v11: handles chain_idx + ESM2 embeddings.

    ESM2 embeddings are node-level tensors [N_i, esm_dim] with varying N_i.
    PyG's Batch.from_data_list concatenates them along dim=0 (like x),
    so esm_emb becomes [total_nodes, esm_dim] — same batching as x.

    If ANY graph in the batch lacks esm_emb, the entire batch gets esm_emb=None
    (to simplify model forward — either all have ESM2 or none do).
    """
    from copy import copy

    y_mf_list, y_bp_list, y_cc_list = [], [], []
    graph_ids = []
    clean_data_list = []
    has_esm = True

    for data in data_list:
        data_copy = copy(data)

        # Extract labels
        if hasattr(data, 'y_mf'):
            y_mf_list.append(data.y_mf)
            y_bp_list.append(data.y_bp)
            y_cc_list.append(data.y_cc)
            del data_copy.y_mf
            del data_copy.y_bp
            del data_copy.y_cc

        # Extract graph_id
        if hasattr(data, 'graph_id'):
            graph_ids.append(data.graph_id)
            del data_copy.graph_id

        # Check ESM2 availability
        if not hasattr(data, 'esm_emb'):
            has_esm = False

        # Remove non-batchable string/list attributes
        for attr in ['chain_ids', 'pdb_id', 'chain_id', 'num_chains']:
            if hasattr(data_copy, attr):
                delattr(data_copy, attr)

        # If not all graphs have ESM, remove it from all to avoid batching errors
        if not has_esm and hasattr(data_copy, 'esm_emb'):
            delattr(data_copy, 'esm_emb')

        clean_data_list.append(data_copy)

    # If mixed (some have, some don't), strip ESM from all
    if not has_esm:
        for d in clean_data_list:
            if hasattr(d, 'esm_emb'):
                delattr(d, 'esm_emb')

    # Batch: x, edge_index, edge_attr, chain_idx, esm_emb (if present) batched by PyG
    batch = Batch.from_data_list(clean_data_list)

    # Add back labels as stacked tensors
    if y_mf_list:
        batch.y_mf = torch.stack(y_mf_list)
        batch.y_bp = torch.stack(y_bp_list)
        batch.y_cc = torch.stack(y_cc_list)

    if graph_ids:
        batch.graph_id = graph_ids

    # Mark whether this batch has ESM2 embeddings
    batch._has_esm = has_esm

    return batch


# ════════════════════════════════════════════════════════════════
#  DataLoader Creation
# ════════════════════════════════════════════════════════════════
def get_dataloaders_v11(
    graphs_dir: str,
    annotation_file: str,
    esm2_dir: Optional[str] = None,
    esm2_dim: int = 1280,
    graph_ids: Optional[List[str]] = None,
    batch_size: int = 8,
    eval_batch_size: int = None,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    train_split_file: Optional[str] = None,
    valid_split_file: Optional[str] = None,
    test_split_file: Optional[str] = None,
    max_nodes: int = 2000,
    max_edges: int = 60000,
    pin_memory: bool = True,
) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, ProteinGODatasetV11, Optional[ESM2EmbeddingLoader]]:
    """
    Create DataLoaders for v11 training.

    Uses MultiChainAnnotationParser for PDB-level annotation aggregation.
    Uses ProteinGODatasetV11 for ESM2 embedding attachment.
    Uses custom_collate_v11 for proper batching.

    Args:
        graphs_dir: Directory containing v10 graph batch files
        annotation_file: GO annotation TSV file
        esm2_dir: Directory with ESM2 embedding .pt files (or None)
        esm2_dim: ESM2 embedding dimension
        graph_ids: Specific graph IDs to use (or None for auto-detect)
        batch_size: Training batch size
        eval_batch_size: Validation/test batch size (default: batch_size // 2)
                         Smaller eval batch prevents OOM on memory-constrained GPUs
                         since CUDA memory is fragmented after training.
        num_workers: DataLoader workers
        train_ratio/val_ratio/test_ratio: Split ratios
        seed: Random seed for reproducibility
        train_split_file/valid_split_file/test_split_file:
            Optional split files (one ID per line). If all are available,
            these are used instead of random splitting.
        max_nodes/max_edges: Graph size limits
        pin_memory: Pin memory for GPU transfer

    Returns:
        (train_loader, val_loader, test_loader, train_dataset, esm2_loader)
    """
    if eval_batch_size is None:
        eval_batch_size = max(batch_size // 2, 1)
    # Parse annotations with multi-chain support (same as v10)
    parser = MultiChainAnnotationParser(annotation_file)

    # Create ESM2 loader (if directory provided)
    esm2_loader = None
    if esm2_dir is not None:
        esm2_loader = ESM2EmbeddingLoader(esm2_dir, expected_dim=esm2_dim)
        if not esm2_loader.available:
            print("[WARN] ESM2 dir provided but not available. Running without ESM2.")
            esm2_loader = None

    # Get graph IDs (same logic as v10)
    if graph_ids is None:
        manifest_file = Path(graphs_dir) / 'subset_manifest.json'
        index_file = Path(graphs_dir) / 'graph_index.json'

        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            graph_ids = manifest.get('graph_ids', None)

        if graph_ids is None and index_file.exists():
            with open(index_file, 'r') as f:
                graph_index = json.load(f)
            available_ids = set(graph_index.keys())
            annotation_ids = set(parser.get_graph_ids())
            graph_ids = list(available_ids.intersection(annotation_ids))
            print(f"Using graph_index.json: {len(available_ids)} graphs, "
                  f"{len(annotation_ids)} annotations, {len(graph_ids)} overlap")

        if graph_ids is None:
            graph_ids = parser.get_graph_ids()

    print(f"Total graph IDs (v11 PDB-level): {len(graph_ids)}")

    # Resolve split files (explicit args first, then auto-detect common names)
    ann_path = Path(annotation_file)
    if train_split_file is None:
        auto_train = ann_path.parent / 'nrPDB-GO_2019.06.18_train.txt'
        if auto_train.exists():
            train_split_file = str(auto_train)
    if valid_split_file is None:
        auto_valid = ann_path.parent / 'nrPDB-GO_2019.06.18_valid.txt'
        if auto_valid.exists():
            valid_split_file = str(auto_valid)
    if test_split_file is None:
        auto_test = ann_path.parent / 'nrPDB-GO_2019.06.18_test.txt'
        if auto_test.exists():
            test_split_file = str(auto_test)

    split_files_ready = all([train_split_file, valid_split_file, test_split_file])

    if split_files_ready:
        available_ids = set(graph_ids)

        def _read_split_ids(path: str) -> List[str]:
            ids = []
            with open(path, 'r') as f:
                for line in f:
                    value = line.strip()
                    if not value:
                        continue
                    # Files can contain chain IDs (e.g., 4MID-A).
                    # v10/v11 graphs are PDB-level IDs, so use prefix.
                    pdb_id = value.split('-', 1)[0]
                    ids.append(pdb_id)
            return ids

        raw_train = _read_split_ids(train_split_file)
        raw_valid = _read_split_ids(valid_split_file)
        raw_test = _read_split_ids(test_split_file)

        # De-duplicate while preserving order, then keep only available graph IDs
        def _unique_and_filter(ids: List[str]) -> List[str]:
            seen = set()
            out = []
            for gid in ids:
                if gid in seen:
                    continue
                seen.add(gid)
                if gid in available_ids:
                    out.append(gid)
            return out

        train_ids = _unique_and_filter(raw_train)
        val_ids = _unique_and_filter(raw_valid)
        test_ids = _unique_and_filter(raw_test)

        print(f"Using provided split files:")
        print(f"  train: {train_split_file}")
        print(f"  valid: {valid_split_file}")
        print(f"  test:  {test_split_file}")
        print(
            f"  Resolved splits (after PDB mapping & graph filtering): "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

        if min(len(train_ids), len(val_ids), len(test_ids)) == 0:
            print("[WARN] At least one provided split became empty after filtering.")
            print("       Falling back to random split with fixed seed.")
            train_ids, val_ids, test_ids = create_data_splits(
                graph_ids, train_ratio, val_ratio, test_ratio, seed
            )
    else:
        # Create splits (same seed → same split as v10 for fair comparison)
        train_ids, val_ids, test_ids = create_data_splits(
            graph_ids, train_ratio, val_ratio, test_ratio, seed
        )

    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Create datasets with ESM2 support
    train_dataset = ProteinGODatasetV11(
        graphs_dir, parser, train_ids,
        esm2_loader=esm2_loader,
        max_nodes=max_nodes, max_edges=max_edges
    )
    val_dataset = ProteinGODatasetV11(
        graphs_dir, parser, val_ids,
        esm2_loader=esm2_loader,
        max_nodes=max_nodes, max_edges=max_edges
    )
    test_dataset = ProteinGODatasetV11(
        graphs_dir, parser, test_ids,
        esm2_loader=esm2_loader,
        max_nodes=max_nodes, max_edges=max_edges
    )

    # Print ESM2 coverage estimate
    if esm2_loader is not None and esm2_loader.available:
        sample_ids = train_ids[:min(100, len(train_ids))]
        found = sum(1 for gid in sample_ids if (Path(esm2_dir) / f"{gid}.pt").exists() or
                    (Path(esm2_dir) / f"{gid.upper()}.pt").exists())
        print(f"ESM2 coverage estimate: {found}/{len(sample_ids)} ({100*found/max(1,len(sample_ids)):.0f}%) of sampled train IDs")

    # Create dataloaders with v11 collate
    use_persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None

    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=custom_collate_v11,
        prefetch_factor=prefetch,
        persistent_workers=use_persistent
    )

    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_v11,
        prefetch_factor=prefetch,
        persistent_workers=use_persistent
    )

    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_v11,
        prefetch_factor=prefetch,
        persistent_workers=use_persistent
    )

    return train_loader, val_loader, test_loader, train_dataset, esm2_loader


if __name__ == '__main__':
    import sys

    annotation_file = 'annotations/nrPDB-GO_2019.06.18_annot.tsv'

    if not Path(annotation_file).exists():
        print(f"Annotation file not found: {annotation_file}")
        sys.exit(1)

    print("Testing v11 Dataset (ESM2-enhanced)...")

    # Test without ESM2 (should fall back to v10 behavior)
    parser = MultiChainAnnotationParser(annotation_file)
    stats = parser.get_statistics()
    print(f"\nv11 Annotation Statistics (same as v10):")
    print(f"  Total PDBs: {stats['total_pdbs']}")
    print(f"  GO terms: MF={stats['num_mf_terms']}, BP={stats['num_bp_terms']}, CC={stats['num_cc_terms']}")

    # Test ESM2 loader (may or may not have embeddings)
    esm_loader = ESM2EmbeddingLoader("output_v11/esm2_embeddings")
    print(f"\nESM2 loader stats: {esm_loader.get_stats()}")

    print("\n✓ Dataset v11 test passed!")
