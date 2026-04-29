"""
Dataset for v10 (Inter-Chain Geometry Aware Graphs)

v10 graphs have graph_id = PDB_ID (e.g., "1A2B") instead of PDB-chain ("1A2B-A").
Annotations are per-chain, so this module aggregates them to PDB-level:
  Union of GO terms across all chains → single label vector per PDB.

Key classes:
  MultiChainAnnotationParser: wraps GOAnnotationParser for PDB-level lookup
  custom_collate_v10: handles chain_idx/chain_ids in batching

Uses the same ProteinGODataset from dataset.py — only the parser changes.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from .annotations import GOAnnotationParser
from .base_dataset import ProteinGODataset, create_data_splits


# ════════════════════════════════════════════════════════════════
#  Multi-Chain Annotation Parser
# ════════════════════════════════════════════════════════════════
class MultiChainAnnotationParser:
    """
    Wraps GOAnnotationParser to provide PDB-level annotation lookup.
    
    Original annotations: "1A2B-A" → {mf: [...], bp: [...], cc: [...]}
    v10 lookup:           "1A2B"   → union of annotations from all chains
    
    Also supports chain-level lookup (backward compat with v9 graph_ids).
    """
    
    def __init__(self, annotation_file: str, min_count: int = 0):
        # Underlying chain-level parser
        self._parser = GOAnnotationParser(annotation_file, min_count=min_count)
        
        # Build PDB-level annotation index
        self._pdb_annotations: Dict[str, Dict[str, set]] = {}
        self._pdb_chains: Dict[str, List[str]] = defaultdict(list)
        
        for chain_id, annot in self._parser.annotations.items():
            # chain_id format: "PDB-chain" e.g. "1A2B-A"
            parts = chain_id.split('-')
            if len(parts) >= 2:
                pdb_id = parts[0]
                chain_letter = '-'.join(parts[1:])  # Handle multi-char chain IDs
            else:
                pdb_id = chain_id
                chain_letter = "?"
            
            if pdb_id not in self._pdb_annotations:
                self._pdb_annotations[pdb_id] = {'mf': set(), 'bp': set(), 'cc': set()}
            
            # Union of GO terms across chains
            self._pdb_annotations[pdb_id]['mf'].update(annot['mf'])
            self._pdb_annotations[pdb_id]['bp'].update(annot['bp'])
            self._pdb_annotations[pdb_id]['cc'].update(annot['cc'])
            
            self._pdb_chains[pdb_id].append(chain_letter)
        
        print(f"[v10] Built PDB-level annotations: {len(self._pdb_annotations)} PDBs "
              f"from {len(self._parser.annotations)} chain-level entries")
    
    @property
    def num_mf(self) -> int:
        return self._parser.num_mf
    
    @property
    def num_bp(self) -> int:
        return self._parser.num_bp
    
    @property
    def num_cc(self) -> int:
        return self._parser.num_cc
    
    @property
    def mf_terms(self) -> List[str]:
        return self._parser.mf_terms
    
    @property
    def bp_terms(self) -> List[str]:
        return self._parser.bp_terms
    
    @property
    def cc_terms(self) -> List[str]:
        return self._parser.cc_terms
    
    @property
    def mf_to_idx(self) -> Dict[str, int]:
        return self._parser.mf_to_idx
    
    @property
    def bp_to_idx(self) -> Dict[str, int]:
        return self._parser.bp_to_idx
    
    @property
    def cc_to_idx(self) -> Dict[str, int]:
        return self._parser.cc_to_idx
    
    @property
    def mf_counts(self):
        return self._parser.mf_counts
    
    @property
    def bp_counts(self):
        return self._parser.bp_counts
    
    @property
    def cc_counts(self):
        return self._parser.cc_counts
    
    @property
    def annotations(self):
        """Return PDB-level annotations (with lists instead of sets)."""
        return {
            pdb_id: {
                'mf': list(annot['mf']),
                'bp': list(annot['bp']),
                'cc': list(annot['cc']),
            }
            for pdb_id, annot in self._pdb_annotations.items()
        }
    
    def get_graph_ids(self) -> List[str]:
        """Get all PDB-level graph IDs that have annotations."""
        return list(self._pdb_annotations.keys())
    
    def has_annotation(self, graph_id: str) -> bool:
        """Check if a graph has annotations (supports both PDB and PDB-chain IDs)."""
        # Try PDB-level first
        if graph_id in self._pdb_annotations:
            return True
        # Fall back to chain-level (backward compat)
        return graph_id in self._parser.annotations
    
    def get_labels(self, graph_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get multi-label binary targets for a graph.
        
        Supports both:
          - PDB IDs ("1A2B") → returns union of all chain annotations
          - Chain IDs ("1A2B-A") → returns chain-specific annotations
        """
        # Try PDB-level
        if graph_id in self._pdb_annotations:
            annot = self._pdb_annotations[graph_id]
            
            mf_vec = torch.zeros(self.num_mf, dtype=torch.float32)
            bp_vec = torch.zeros(self.num_bp, dtype=torch.float32)
            cc_vec = torch.zeros(self.num_cc, dtype=torch.float32)
            
            for term in annot['mf']:
                if term in self._parser.mf_to_idx:
                    mf_vec[self._parser.mf_to_idx[term]] = 1.0
            
            for term in annot['bp']:
                if term in self._parser.bp_to_idx:
                    bp_vec[self._parser.bp_to_idx[term]] = 1.0
            
            for term in annot['cc']:
                if term in self._parser.cc_to_idx:
                    cc_vec[self._parser.cc_to_idx[term]] = 1.0
            
            return mf_vec, bp_vec, cc_vec
        
        # Fall back to chain-level
        return self._parser.get_labels(graph_id)
    
    def get_raw_annotations(self, graph_id: str) -> Optional[Dict[str, List[str]]]:
        """Get raw GO term lists."""
        if graph_id in self._pdb_annotations:
            annot = self._pdb_annotations[graph_id]
            return {
                'mf': list(annot['mf']),
                'bp': list(annot['bp']),
                'cc': list(annot['cc']),
            }
        return self._parser.get_raw_annotations(graph_id)
    
    def compute_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute class weights (delegates to underlying parser)."""
        return self._parser.compute_class_weights()
    
    def get_statistics(self) -> Dict:
        """Get annotation statistics at PDB level."""
        mf_per_pdb, bp_per_pdb, cc_per_pdb = [], [], []
        
        for annot in self._pdb_annotations.values():
            mf_per_pdb.append(len([t for t in annot['mf'] if t in self._parser.mf_to_idx]))
            bp_per_pdb.append(len([t for t in annot['bp'] if t in self._parser.bp_to_idx]))
            cc_per_pdb.append(len([t for t in annot['cc'] if t in self._parser.cc_to_idx]))
        
        return {
            'total_pdbs': len(self._pdb_annotations),
            'total_chain_entries': len(self._parser.annotations),
            'num_mf_terms': self.num_mf,
            'num_bp_terms': self.num_bp,
            'num_cc_terms': self.num_cc,
            'avg_mf_per_pdb': np.mean(mf_per_pdb) if mf_per_pdb else 0,
            'avg_bp_per_pdb': np.mean(bp_per_pdb) if bp_per_pdb else 0,
            'avg_cc_per_pdb': np.mean(cc_per_pdb) if cc_per_pdb else 0,
        }


# ════════════════════════════════════════════════════════════════
#  Custom Collation for v10
# ════════════════════════════════════════════════════════════════
def custom_collate_v10(data_list):
    """
    Custom collate for v10 multi-chain graphs.
    
    Handles:
    - y_mf/y_bp/y_cc: stacked as [batch_size, num_classes]
    - graph_id: collected as list
    - chain_ids: stripped (not batchable)
    - pdb_id / chain_id: stripped (string attributes)
    - chain_idx: kept (node-level tensor, batched by PyG automatically)
    - num_chains: stripped (graph-level int, not needed in training)
    """
    from copy import copy
    
    y_mf_list, y_bp_list, y_cc_list = [], [], []
    graph_ids = []
    clean_data_list = []
    
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
        
        # Extract non-batchable attributes
        if hasattr(data, 'graph_id'):
            graph_ids.append(data.graph_id)
            del data_copy.graph_id
        
        # Remove string/list attributes that can't be batched
        for attr in ['chain_ids', 'pdb_id', 'chain_id', 'num_chains']:
            if hasattr(data_copy, attr):
                delattr(data_copy, attr)
        
        clean_data_list.append(data_copy)
    
    # Batch the rest (x, edge_index, edge_attr, chain_idx, pos, etc.)
    batch = Batch.from_data_list(clean_data_list)
    
    # Add back labels as stacked tensors
    if y_mf_list:
        batch.y_mf = torch.stack(y_mf_list)
        batch.y_bp = torch.stack(y_bp_list)
        batch.y_cc = torch.stack(y_cc_list)
    
    if graph_ids:
        batch.graph_id = graph_ids
    
    return batch


# ════════════════════════════════════════════════════════════════
#  DataLoader Creation
# ════════════════════════════════════════════════════════════════
def get_dataloaders_v10(
    graphs_dir: str,
    annotation_file: str,
    graph_ids: Optional[List[str]] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_nodes: int = 2000,
    max_edges: int = 60000,
    pin_memory: bool = True
) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, ProteinGODataset]:
    """
    Create DataLoaders for v10 training.
    
    Uses MultiChainAnnotationParser for PDB-level annotation aggregation.
    Uses custom_collate_v10 for proper chain_idx batching.
    
    Default max_edges raised to 60000 to accommodate multi-chain graphs.
    """
    # Parse annotations with multi-chain support
    parser = MultiChainAnnotationParser(annotation_file)
    
    # Get graph IDs
    if graph_ids is None:
        # Try manifest / index files
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
    
    print(f"Total graph IDs (v10 PDB-level): {len(graph_ids)}")
    
    # Create splits (same seed → same split as v9 for fair comparison)
    train_ids, val_ids, test_ids = create_data_splits(
        graph_ids, train_ratio, val_ratio, test_ratio, seed
    )
    
    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    # Create datasets using existing ProteinGODataset + multi-chain parser
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
    
    # Create dataloaders with v10 collate
    use_persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None
    
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=custom_collate_v10,
        prefetch_factor=prefetch,
        persistent_workers=use_persistent
    )
    
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_v10,
        prefetch_factor=prefetch,
        persistent_workers=use_persistent
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_v10,
        prefetch_factor=prefetch,
        persistent_workers=use_persistent
    )
    
    return train_loader, val_loader, test_loader, train_dataset


if __name__ == '__main__':
    import sys
    
    annotation_file = 'annotations/nrPDB-GO_2019.06.18_annot.tsv'
    
    if not Path(annotation_file).exists():
        print(f"Annotation file not found: {annotation_file}")
        sys.exit(1)
    
    print("Testing MultiChainAnnotationParser...")
    parser = MultiChainAnnotationParser(annotation_file)
    
    stats = parser.get_statistics()
    print(f"\nv10 Annotation Statistics:")
    print(f"  Total PDBs: {stats['total_pdbs']}")
    print(f"  Total chain entries: {stats['total_chain_entries']}")
    print(f"  GO terms: MF={stats['num_mf_terms']}, BP={stats['num_bp_terms']}, CC={stats['num_cc_terms']}")
    print(f"  Avg per PDB: MF={stats['avg_mf_per_pdb']:.2f}, BP={stats['avg_bp_per_pdb']:.2f}, CC={stats['avg_cc_per_pdb']:.2f}")
    
    # Test PDB-level lookup
    pdb_ids = parser.get_graph_ids()[:3]
    for pdb_id in pdb_ids:
        mf, bp, cc = parser.get_labels(pdb_id)
        print(f"\n  {pdb_id}: MF={mf.sum():.0f}, BP={bp.sum():.0f}, CC={cc.sum():.0f}")
    
    print("\n✓ Dataset v10 test passed!")
