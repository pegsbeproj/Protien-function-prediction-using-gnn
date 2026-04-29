"""
PyTorch Geometric Dataset for Protein GO Term Prediction

Features:
1. Memory-efficient lazy loading from batch files
2. Proper train/val/test splitting
3. GO term label attachment
4. DataLoader creation with proper batching
5. Class weight computation for imbalanced labels
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader as TorchDataLoader  # Use torch DataLoader for custom collate
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader  # Keep for reference
from tqdm import tqdm

from .annotations import GOAnnotationParser


def custom_collate(data_list):
    """
    Custom collate function to properly batch labels.
    
    PyG batches node/edge features correctly, but labels need special handling.
    Graph-level labels should be stacked [batch_size, num_classes], not concatenated.
    
    IMPORTANT: We work with copies to avoid modifying cached data objects.
    """
    from copy import copy
    
    # Save labels and create copies without labels for batching
    y_mf_list = []
    y_bp_list = []
    y_cc_list = []
    graph_ids = []
    clean_data_list = []
    
    for data in data_list:
        # Create a shallow copy so we don't modify the original/cached object
        data_copy = copy(data)
        
        if hasattr(data, 'y_mf'):
            y_mf_list.append(data.y_mf)
            y_bp_list.append(data.y_bp)
            y_cc_list.append(data.y_cc)
            # Remove from copy so Batch.from_data_list won't concatenate them
            del data_copy.y_mf
            del data_copy.y_bp
            del data_copy.y_cc
        
        if hasattr(data, 'graph_id'):
            graph_ids.append(data.graph_id)
            del data_copy.graph_id
        
        clean_data_list.append(data_copy)
    
    # Batch the rest (x, edge_index, etc.)
    batch = Batch.from_data_list(clean_data_list)
    
    # Add back labels as stacked tensors
    if y_mf_list:
        batch.y_mf = torch.stack(y_mf_list)  # [batch_size, num_mf]
        batch.y_bp = torch.stack(y_bp_list)  # [batch_size, num_bp]
        batch.y_cc = torch.stack(y_cc_list)  # [batch_size, num_cc]
    
    if graph_ids:
        batch.graph_id = graph_ids
    
    return batch


class ProteinGODataset(Dataset):
    """
    Dataset for protein graphs with GO term labels.
    
    Memory-efficient implementation that:
    - Loads graphs lazily from batch files
    - Attaches multi-label GO targets
    - Supports train/val/test splitting
    """
    
    def __init__(
        self,
        graphs_dir: str,
        annotation_parser: GOAnnotationParser,
        graph_ids: Optional[List[str]] = None,
        transform=None,
        pre_transform=None,
        max_nodes: int = None,
        max_edges: int = None
    ):
        """
        Initialize dataset.
        
        Args:
            graphs_dir: Directory containing graph batch files
            annotation_parser: Parsed GO annotations
            graph_ids: List of graph IDs to include (None = all)
            transform: PyG transform to apply
            pre_transform: PyG pre-transform
            max_nodes: Maximum nodes per graph (filter)
            max_edges: Maximum edges per graph (filter)
        """
        self.graphs_dir = Path(graphs_dir)
        self.parser = annotation_parser
        self.graph_ids = graph_ids
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        # Graph storage: list of (batch_file, batch_idx, graph_id)
        self._graph_locations: List[Tuple[str, int, str]] = []
        
        # Cache for loaded graphs (larger size for better throughput)
        self._cache: Dict[int, Data] = {}
        self._cache_size = 2000
        
        # Cache for batch files to avoid repeated disk reads
        self._batch_cache: Dict[str, list] = {}
        self._batch_cache_size = 10  # Keep last N batch files in memory
        
        # Load graph locations
        self._load_graph_locations()
        
        super().__init__(str(self.graphs_dir), transform, pre_transform)
    
    def _load_graph_locations(self):
        """Load graph locations from batch files."""
        allowed_ids = set(self.graph_ids) if self.graph_ids else None
        
        # Check for subset file first
        subset_file = self.graphs_dir / 'subset_graphs.pt'
        if subset_file.exists() and allowed_ids is not None:
            print(f"Loading from subset file: {subset_file}")
            try:
                graphs = torch.load(subset_file, weights_only=False)
                for idx, graph in enumerate(graphs):
                    graph_id = graph.graph_id if hasattr(graph, 'graph_id') else getattr(graph, 'pdb_id', None)
                    
                    if graph_id and self.parser.has_annotation(graph_id):
                        if allowed_ids is None or graph_id in allowed_ids:
                            if self._check_size(graph):
                                self._graph_locations.append(('subset', idx, graph_id))
                
                print(f"Loaded {len(self._graph_locations)} graphs from subset")
                return
            except Exception as e:
                print(f"Error loading subset file: {e}")
        
        # Otherwise load from batch files
        batch_files = sorted(self.graphs_dir.glob('graphs_batch_*.pt'))
        
        print(f"Scanning {len(batch_files)} batch files...")
        
        for batch_file in tqdm(batch_files, desc="Loading graph locations"):
            try:
                graphs = torch.load(batch_file, weights_only=False)
                
                for idx, graph in enumerate(graphs):
                    graph_id = graph.graph_id if hasattr(graph, 'graph_id') else getattr(graph, 'pdb_id', None)
                    
                    if not graph_id:
                        continue
                    
                    # Check if we should include this graph
                    if not self.parser.has_annotation(graph_id):
                        continue
                    
                    if allowed_ids is not None and graph_id not in allowed_ids:
                        continue
                    
                    if not self._check_size(graph):
                        continue
                    
                    self._graph_locations.append((str(batch_file), idx, graph_id))
                    
            except Exception as e:
                print(f"Error loading {batch_file}: {e}")
        
        print(f"Found {len(self._graph_locations)} valid graphs")
    
    def _check_size(self, graph: Data) -> bool:
        """Check if graph is within size limits."""
        if self.max_nodes and graph.num_nodes > self.max_nodes:
            return False
        if self.max_edges and graph.edge_index.shape[1] > self.max_edges:
            return False
        return True
    
    @property
    def num_mf(self) -> int:
        return self.parser.num_mf
    
    @property
    def num_bp(self) -> int:
        return self.parser.num_bp
    
    @property
    def num_cc(self) -> int:
        return self.parser.num_cc
    
    def len(self) -> int:
        return len(self._graph_locations)
    
    def get(self, idx: int) -> Data:
        """Get a graph with GO labels attached."""
        # Check cache
        if idx in self._cache:
            return self._cache[idx]
        
        # Load graph
        batch_file, batch_idx, graph_id = self._graph_locations[idx]
        
        # Use batch file cache to avoid repeated disk reads
        batch_key = str(batch_file)
        if batch_key in self._batch_cache:
            graphs = self._batch_cache[batch_key]
        else:
            if batch_file == 'subset':
                graphs = torch.load(self.graphs_dir / 'subset_graphs.pt', weights_only=False)
            else:
                graphs = torch.load(batch_file, weights_only=False)
            
            # Update batch cache (FIFO)
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
        
        # Update cache (FIFO)
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[idx] = graph
        
        return graph
    
    def get_graph_ids(self) -> List[str]:
        """Get all graph IDs in the dataset."""
        return [loc[2] for loc in self._graph_locations]
    
    def compute_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute class weights for handling label imbalance.
        
        Returns:
            Tuple of (mf_weights, bp_weights, cc_weights)
        """
        return self.parser.compute_class_weights()


class MemoryEfficientDataset(Dataset):
    """
    Ultra memory-efficient dataset that loads graphs one at a time.
    
    Use this when even the standard dataset causes memory issues.
    """
    
    def __init__(
        self,
        graphs_dir: str,
        annotation_parser: GOAnnotationParser,
        graph_ids: List[str],
        max_nodes: int = 1000,
        max_edges: int = 20000
    ):
        self.graphs_dir = Path(graphs_dir)
        self.parser = annotation_parser
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        # Build index: graph_id -> (batch_file, idx)
        self.graph_index: Dict[str, Tuple[str, int]] = {}
        self._build_index(graph_ids)
        
        # Ordered list of valid graph IDs
        self.graph_ids = [gid for gid in graph_ids if gid in self.graph_index]
        
        super().__init__(str(self.graphs_dir))
    
    def _build_index(self, target_ids: List[str]):
        """Build index of graph locations."""
        target_set = set(target_ids)
        
        # Check subset file
        subset_file = self.graphs_dir / 'subset_graphs.pt'
        if subset_file.exists():
            try:
                graphs = torch.load(subset_file, weights_only=False)
                for idx, graph in enumerate(graphs):
                    gid = graph.graph_id if hasattr(graph, 'graph_id') else graph.pdb_id
                    if gid in target_set:
                        if graph.num_nodes <= self.max_nodes and graph.edge_index.shape[1] <= self.max_edges:
                            self.graph_index[gid] = ('subset', idx)
                return
            except:
                pass
        
        # Scan batch files
        batch_files = sorted(self.graphs_dir.glob('graphs_batch_*.pt'))
        
        for bf in batch_files:
            try:
                graphs = torch.load(bf, weights_only=False)
                for idx, graph in enumerate(graphs):
                    gid = graph.graph_id if hasattr(graph, 'graph_id') else graph.pdb_id
                    if gid in target_set:
                        if graph.num_nodes <= self.max_nodes and graph.edge_index.shape[1] <= self.max_edges:
                            self.graph_index[gid] = (str(bf), idx)
            except:
                continue
    
    @property
    def num_mf(self) -> int:
        return self.parser.num_mf
    
    @property
    def num_bp(self) -> int:
        return self.parser.num_bp
    
    @property
    def num_cc(self) -> int:
        return self.parser.num_cc
    
    def len(self) -> int:
        return len(self.graph_ids)
    
    def get(self, idx: int) -> Data:
        graph_id = self.graph_ids[idx]
        batch_file, batch_idx = self.graph_index[graph_id]
        
        if batch_file == 'subset':
            graphs = torch.load(self.graphs_dir / 'subset_graphs.pt', weights_only=False)
        else:
            graphs = torch.load(batch_file, weights_only=False)
        
        graph = graphs[batch_idx]
        
        # Attach labels
        mf, bp, cc = self.parser.get_labels(graph_id)
        graph.y_mf = mf
        graph.y_bp = bp
        graph.y_cc = cc
        graph.graph_id = graph_id
        
        return graph
    
    def compute_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.parser.compute_class_weights()


def create_data_splits(
    graph_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits.
    
    Args:
        graph_ids: List of all graph IDs
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Shuffle with fixed seed
    ids = list(graph_ids)
    random.seed(seed)
    random.shuffle(ids)
    
    n = len(ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]
    
    return train_ids, val_ids, test_ids


def get_dataloaders(
    graphs_dir: str,
    annotation_file: str,
    graph_ids: Optional[List[str]] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_nodes: int = 1000,
    max_edges: int = 20000,
    pin_memory: bool = True
) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, ProteinGODataset]:
    """
    Create DataLoaders for training.
    
    Args:
        graphs_dir: Directory with graph files
        annotation_file: Path to GO annotation TSV
        graph_ids: Optional specific graph IDs to use
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        max_nodes: Maximum nodes per graph
        max_edges: Maximum edges per graph
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_dataset)
    """
    # Parse annotations
    parser = GOAnnotationParser(annotation_file)
    
    # Get graph IDs - must exist both in annotations AND as graphs
    if graph_ids is None:
        # First try manifest
        manifest_file = Path(graphs_dir) / 'subset_manifest.json'
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            graph_ids = manifest.get('graph_ids', None)
        
        # If no graph_ids in manifest, try graph_index.json
        if graph_ids is None:
            index_file = Path(graphs_dir) / 'graph_index.json'
            if index_file.exists():
                with open(index_file, 'r') as f:
                    graph_index = json.load(f)
                available_graph_ids = set(graph_index.keys())
                annotation_ids = set(parser.get_graph_ids())
                # Use only IDs that have both annotations AND graphs
                graph_ids = list(available_graph_ids.intersection(annotation_ids))
                print(f"Using graph_index.json: {len(available_graph_ids)} graphs, "
                      f"{len(annotation_ids)} annotations, {len(graph_ids)} overlap")
            else:
                graph_ids = parser.get_graph_ids()
    
    print(f"Total graph IDs: {len(graph_ids)}")
    
    # Create splits
    train_ids, val_ids, test_ids = create_data_splits(
        graph_ids, train_ratio, val_ratio, test_ratio, seed
    )
    
    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    # Create datasets
    train_dataset = ProteinGODataset(
        graphs_dir, parser, train_ids,
        max_nodes=max_nodes, max_edges=max_edges
    )
    
    val_dataset = ProteinGODataset(
        graphs_dir, parser, val_ids,
        max_nodes=max_nodes, max_edges=max_edges
    )
    
    test_dataset = ProteinGODataset(
        graphs_dir, parser, test_ids,
        max_nodes=max_nodes, max_edges=max_edges
    )
    
    # Create dataloaders - use torch DataLoader (not PyG) to properly use custom collate
    # Use prefetch_factor and persistent_workers for faster loading
    use_persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None
    
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Avoid batch norm issues with batch_size=1
        collate_fn=custom_collate,
        prefetch_factor=prefetch,
        persistent_workers=use_persistent
    )
    
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate,
        prefetch_factor=prefetch,
        persistent_workers=use_persistent
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate,
        prefetch_factor=prefetch,
        persistent_workers=use_persistent
    )
    
    return train_loader, val_loader, test_loader, train_dataset


if __name__ == '__main__':
    # Test the dataset
    import sys
    
    graphs_dir = 'graphs_v2'
    annotation_file = 'annotations/nrPDB-GO_2019.06.18_annot.tsv'
    
    if not Path(annotation_file).exists():
        print(f"Annotation file not found: {annotation_file}")
        sys.exit(1)
    
    if not Path(graphs_dir).exists():
        print(f"Graphs directory not found: {graphs_dir}")
        print("Run pdb_to_pyg_v2.py first to create graphs")
        sys.exit(1)
    
    print("Testing ProteinGODataset...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
        graphs_dir=graphs_dir,
        annotation_file=annotation_file,
        batch_size=4,
        num_workers=0
    )
    
    print(f"\nDataset info:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  MF classes: {train_dataset.num_mf}")
    print(f"  BP classes: {train_dataset.num_bp}")
    print(f"  CC classes: {train_dataset.num_cc}")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    for batch in train_loader:
        print(f"  Batch: {batch}")
        print(f"  Nodes: {batch.num_nodes}")
        print(f"  Edges: {batch.edge_index.shape}")
        print(f"  MF labels shape: {batch.y_mf.shape}")
        print(f"  BP labels shape: {batch.y_bp.shape}")
        print(f"  CC labels shape: {batch.y_cc.shape}")
        break
    
    print("\nDataset test passed!")
