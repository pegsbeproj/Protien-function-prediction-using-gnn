"""
Configuration for Protein Function Prediction Pipeline

All hyperparameters and paths centralized for easy modification.
Designed for RTX 4060 (8GB VRAM) + 16GB RAM.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class PathConfig:
    """File and directory paths."""
    # Input data
    pdb_dir: str = "pdbs"
    annotation_file: str = "annotations/nrPDB-GO_2019.06.18_annot.tsv"
    
    # Output directories
    graphs_dir: str = "graphs_v2"
    subset_dir: str = "subset_1gb"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    
    def create_dirs(self):
        """Create all output directories."""
        for d in [self.graphs_dir, self.subset_dir, self.checkpoints_dir, self.logs_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


@dataclass
class GraphConfig:
    """Graph construction parameters."""
    # Distance threshold for edge creation (Angstroms)
    distance_threshold: float = 8.0
    
    # Maximum neighbors per node (O(N·k) instead of O(N²))
    max_neighbors: int = 20
    
    # Node features
    num_amino_acids: int = 20  # One-hot encoding size
    
    # Edge features
    include_edge_distances: bool = True
    
    # Chain handling
    per_chain: bool = True  # One graph per chain (matches GO annotations)
    
    # Graph size limits (skip graphs outside these bounds)
    min_nodes: int = 10
    max_nodes: int = 2000  # Prevent memory issues from giant structures
    max_edges: int = 40000  # max_nodes * max_neighbors
    
    # Processing
    num_workers: int = 4
    batch_size: int = 100  # Graphs per saved batch file


@dataclass
class SubsetConfig:
    """Subset creation parameters."""
    # Target size
    target_size_gb: float = 1.0
    
    # Sampling parameters
    random_seed: int = 42
    
    # Size distribution preservation
    preserve_size_distribution: bool = True
    size_bins: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000, 2000])
    
    # GO term preservation
    min_samples_per_go_term: int = 5  # Ensure rare terms appear
    
    # Graph filtering for subset
    max_nodes_subset: int = 1000  # More strict for 1GB subset
    max_edges_subset: int = 20000
    
    # Validation
    min_go_terms_per_graph: int = 1  # Must have at least one GO annotation


@dataclass 
class ModelConfig:
    """Model architecture parameters."""
    # Input
    input_dim: int = 20  # One-hot amino acids
    
    # Architecture (smaller for local training)
    hidden_dim: int = 128  # Reduced from 256
    num_gnn_layers: int = 2
    num_gat_heads: int = 4
    
    # Regularization
    dropout: float = 0.3
    
    # Output dimensions (will be set dynamically from annotations)
    num_mf_classes: int = 489
    num_bp_classes: int = 1943
    num_cc_classes: int = 320


@dataclass
class TrainingConfig:
    """Training parameters optimized for RTX 4060 (8GB VRAM)."""
    # Batch size (conservative for memory stability)
    batch_size: int = 8  # Small batches for memory efficiency
    
    # Gradient accumulation to simulate larger batch
    gradient_accumulation_steps: int = 4  # Effective batch = 32
    
    # Training epochs
    epochs: int = 100
    
    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Data loading
    num_workers: int = 0  # 0 for Windows compatibility and memory stability
    pin_memory: bool = True
    
    # Memory management
    clear_cache_every_n_batches: int = 50
    
    # Validation
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Random seed
    seed: int = 42
    
    # Loss weighting
    max_class_weight: float = 50.0  # Clip extreme weights
    
    # Gradient clipping
    gradient_clip: float = 1.0


@dataclass
class MemoryConfig:
    """Memory limits for safe operation."""
    # Maximum GPU memory usage (bytes) - leave 1GB headroom
    max_gpu_memory_gb: float = 7.0
    
    # Maximum single graph size that we'll attempt to process
    max_graph_memory_mb: float = 100.0
    
    # Batch memory limit
    max_batch_memory_mb: float = 500.0
    
    # Enable memory monitoring
    monitor_memory: bool = True
    
    # Clear GPU cache every N batches (0 to disable)
    clear_cache_every: int = 50


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    def save(self, path: str):
        """Save configuration to JSON."""
        config_dict = {
            'paths': self.paths.__dict__,
            'graph': self.graph.__dict__,
            'subset': self.subset.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'memory': self.memory.__dict__
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.paths = PathConfig(**config_dict.get('paths', {}))
        config.graph = GraphConfig(**config_dict.get('graph', {}))
        config.subset = SubsetConfig(**config_dict.get('subset', {}))
        config.model = ModelConfig(**config_dict.get('model', {}))
        config.training = TrainingConfig(**config_dict.get('training', {}))
        config.memory = MemoryConfig(**config_dict.get('memory', {}))
        return config


def get_default_config() -> Config:
    """Get default configuration for 1GB subset training."""
    return Config()


if __name__ == '__main__':
    # Print default configuration
    config = get_default_config()
    print("Default Configuration for Protein GNN Pipeline")
    print("=" * 60)
    print(f"\nGraph Construction:")
    print(f"  Distance threshold: {config.graph.distance_threshold} Å")
    print(f"  Max neighbors/node: {config.graph.max_neighbors}")
    print(f"  Max nodes/graph: {config.graph.max_nodes}")
    print(f"  Per-chain: {config.graph.per_chain}")
    
    print(f"\nSubset:")
    print(f"  Target size: {config.subset.target_size_gb} GB")
    print(f"  Max nodes: {config.subset.max_nodes_subset}")
    
    print(f"\nModel:")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  GNN layers: {config.model.num_gnn_layers}")
    print(f"  GAT heads: {config.model.num_gat_heads}")
    
    print(f"\nTraining:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch: {config.training.batch_size * config.training.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.training.learning_rate}")
    
    # Save default config
    config.save('config_default.json')
    print(f"\nSaved to config_default.json")
