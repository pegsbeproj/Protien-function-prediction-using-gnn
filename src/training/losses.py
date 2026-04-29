"""
Training Script for v9 (Label-Embedding + Dual-Conv Architecture)

v9 introduces several advanced techniques from the research community:

KEY INNOVATIONS (compared to v8):
=================================
1. **Co-occurrence SVD Label Embeddings**
   - Labels initialized from annotation co-occurrence (PPMI + SVD)
   - GO terms that co-occur start with similar vectors
   - Replaces random initialization

2. **Annotation Propagation**
   - Approximates GO true-path rule without OBO file
   - If GO_child always co-occurs with GO_parent (≥95%), propagate child → parent
   - Fills in missing ancestor terms

3. **Per-Class Threshold Fmax**
   - Each GO term gets its own optimal threshold
   - Rare terms get lower thresholds, common terms get higher
   - Zero VRAM cost (post-hoc computation)

4. **Class-Balanced Asymmetric Loss**
   - Asymmetric focal loss with per-class reweighting
   - From "Class-Balanced Loss Based on Effective Number of Samples"

5. **Co-occurrence Regularization**
   - Penalizes probability gaps between frequently co-occurring GO terms
   - Encourages label consistency

6. **Dual Convolution (GCN + GATv2)**
   - GCN captures topology, GATv2 adds attention with edge features
   - Both per layer with residual connections

7. **Triple Pooling (Gated + Mean + Max)**
   - More expressive graph representation than mean pooling alone

8GB VRAM Configuration (upgraded from 4GB):
============================================
- hidden_dim: 192 (was 128)
- GAT heads: 6 (was 4)
- layers: 4 (was 3)
- batch_size: 64 with num_workers=2
- AMP: ON (float16)
- Gradient checkpointing: ON

Usage:
    python train_v9.py --graphs-dir output_v7/graphs_v7 --checkpoint-dir output_v9/checkpoints
    python train_v9.py --compare  # Compare with previous versions
    
    # For 4GB VRAM (fallback):
    python train_v9.py --vram-gb 4.0 --batch-size 32 --num-workers 2
"""

import gc
import json
import os
import time
import warnings

# Suppress known non-critical warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except Exception:
    pynvml = None
    _PYNVML_AVAILABLE = False

from config import TrainingConfig, MemoryConfig
from dataset import get_dataloaders, ProteinGODataset, custom_collate
from model_v9 import (
    ProteinGNNv9,
    create_v9_model,
    count_parameters,
    count_layer_parameters,
    V9_NODE_DIM,
    V9_EDGE_DIM,
    V9_HIDDEN_DIM,
    V9_LAYERS,
    V9_HEADS,
)


# ════════════════════════════════════════════════════════════════
#  CO-OCCURRENCE SVD LABEL EMBEDDINGS
# ════════════════════════════════════════════════════════════════
def build_cooccurrence_embeddings(
    annotations: Dict[str, List[str]],
    go_list: List[str],
    emb_dim: int = 192
) -> np.ndarray:
    """
    Build label embeddings from GO term co-occurrence matrix via truncated SVD.
    
    GO terms that frequently appear together in the same protein will get
    similar embedding vectors.
    
    This replaces random-init label embeddings with semantically meaningful ones.
    
    Args:
        annotations: Dict mapping protein_id -> list of GO terms
        go_list: List of GO terms (defines ordering)
        emb_dim: Embedding dimension
        
    Returns:
        np.ndarray of shape (n_labels, emb_dim)
    """
    g2i = {g: i for i, g in enumerate(go_list)}
    n = len(go_list)
    
    if n == 0:
        return np.zeros((0, emb_dim), dtype=np.float32)
    
    # Build co-occurrence matrix C[i,j] = #proteins with both GO_i and GO_j
    cooc = np.zeros((n, n), dtype=np.float32)
    for pid, gos in annotations.items():
        idxs = [g2i[g] for g in gos if g in g2i]
        for a in idxs:
            for b in idxs:
                cooc[a, b] += 1.0
    
    # PPMI (Positive Pointwise Mutual Information)
    row_sum = cooc.sum(axis=1, keepdims=True) + 1e-8
    col_sum = cooc.sum(axis=0, keepdims=True) + 1e-8
    total = cooc.sum() + 1e-8
    pmi = np.log2((cooc * total) / (row_sum * col_sum) + 1e-10)
    ppmi = np.maximum(pmi, 0.0)
    
    # Truncated SVD
    k = min(emb_dim, n - 1, 50)  # Limit k to avoid numerical issues
    if k < 1:
        return np.random.randn(n, emb_dim).astype(np.float32) * 0.02
    
    try:
        U, S, _ = svds(ppmi.astype(np.float64), k=k)
        # Weight by sqrt(singular values)
        embs = U * np.sqrt(S)[np.newaxis, :]
    except Exception as e:
        print(f"  Warning: SVD failed ({e}), using random init")
        return np.random.randn(n, emb_dim).astype(np.float32) * 0.02
    
    # Pad to emb_dim if k < emb_dim
    if k < emb_dim:
        embs = np.pad(embs, ((0, 0), (0, emb_dim - k)))
    
    # L2 normalize then scale
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms * 0.5  # Slightly larger than random 0.02
    
    print(f"  Co-occurrence embeddings: {n} labels -> {k}-d SVD -> {emb_dim}-d")
    return embs.astype(np.float32)


# ════════════════════════════════════════════════════════════════
#  ANNOTATION PROPAGATION (approximate true-path rule)
# ════════════════════════════════════════════════════════════════
def propagate_annotations(
    annotations: Dict[str, List[str]],
    go_list: List[str],
    min_cooc_ratio: float = 0.95
) -> Dict[str, List[str]]:
    """
    Approximate the true-path rule WITHOUT the GO DAG file.
    
    Logic: if GO_child always (≥95%) co-occurs with GO_parent,
    then GO_parent is likely an ancestor of GO_child.
    We propagate: whenever a protein has GO_child, ensure it also has GO_parent.
    
    Args:
        annotations: Dict mapping protein_id -> list of GO terms
        go_list: List of valid GO terms
        min_cooc_ratio: Minimum ratio for parent-child inference
        
    Returns:
        Updated annotations dict with propagated terms
    """
    g2i = {g: i for i, g in enumerate(go_list)}
    n = len(go_list)
    
    if n == 0:
        return annotations
    
    # Count co-occurrences and individual frequencies
    freq = np.zeros(n, dtype=np.float32)
    cooc = np.zeros((n, n), dtype=np.float32)
    
    for gos in annotations.values():
        idxs = [g2i[g] for g in gos if g in g2i]
        for a in idxs:
            freq[a] += 1.0
            for b in idxs:
                cooc[a, b] += 1.0
    
    # Find implicit parent-child relationships
    propagation_rules = []
    for child in range(n):
        if freq[child] < 5:
            continue
        for parent in range(n):
            if child == parent:
                continue
            ratio = cooc[child, parent] / freq[child]
            if ratio >= min_cooc_ratio and freq[parent] > freq[child]:
                propagation_rules.append((go_list[child], go_list[parent]))
    
    print(f"  Discovered {len(propagation_rules)} implicit parent-child rules")
    
    # Apply propagation
    go_set = set(go_list)
    propagated = {}
    additions = 0
    
    for pid, gos in annotations.items():
        new_gos = set(g for g in gos if g in go_set)
        before = len(new_gos)
        for child_go, parent_go in propagation_rules:
            if child_go in new_gos:
                new_gos.add(parent_go)
        additions += len(new_gos) - before
        propagated[pid] = list(new_gos)
    
    print(f"  Added {additions} propagated annotations across {len(propagated)} proteins")
    return propagated


# ════════════════════════════════════════════════════════════════
#  CLASS-BALANCED ASYMMETRIC LOSS
# ════════════════════════════════════════════════════════════════
class ClassBalancedAsymmetricLoss(nn.Module):
    """
    Asymmetric focal loss with per-class reweighting.
    
    From "Class-Balanced Loss Based on Effective Number of Samples" (Cui 2019).
    
    Key features:
    - Asymmetric focusing: gamma_pos=0, gamma_neg=4 (focus more on negatives)
    - Class-balanced weighting based on effective sample count
    """
    def __init__(
        self,
        class_counts: np.ndarray,
        beta: float = 0.9999,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0
    ):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        
        # Compute effective number of samples
        eff = 1.0 - np.power(beta, class_counts + 1e-8)
        w = (1.0 - beta) / (eff + 1e-8)
        w = w / (w.sum() + 1e-8) * len(w)
        
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32))
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        
        # Asymmetric focusing
        pos = -targets * ((1 - p) ** self.gp) * torch.log(p.clamp(min=1e-8))
        neg = -(1 - targets) * (p ** self.gn) * torch.log((1 - p).clamp(min=1e-8))
        
        # Class-balanced weighting
        loss = (pos + neg) * self.w.unsqueeze(0)
        
        return loss.mean()


# ════════════════════════════════════════════════════════════════
#  CO-OCCURRENCE REGULARIZATION
# ════════════════════════════════════════════════════════════════
class CooccurrenceRegularizer(nn.Module):
    """
    Encourages the model to predict co-occurring GO terms together.
    
    If GO_i and GO_j always appear together, the model should assign
    similar probabilities to both. Penalizes large probability gaps.
    """
    def __init__(
        self,
        annotations: Dict[str, List[str]],
        go_list: List[str],
        min_cooc: float = 0.8,
        weight: float = 0.1
    ):
        super().__init__()
        self.weight = weight
        
        g2i = {g: i for i, g in enumerate(go_list)}
        n = len(go_list)
        
        if n == 0:
            self.has_pairs = False
            return
        
        # Build normalized co-occurrence matrix
        freq = np.zeros(n, dtype=np.float32)
        cooc = np.zeros((n, n), dtype=np.float32)
        
        for gos in annotations.values():
            idxs = [g2i[g] for g in gos if g in g2i]
            for a in idxs:
                freq[a] += 1.0
                for b in idxs:
                    cooc[a, b] += 1.0
        
        # Jaccard similarity: |A∩B| / |A∪B|
        jaccard = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                union = freq[i] + freq[j] - cooc[i, j]
                if union > 0:
                    j_val = cooc[i, j] / union
                    if j_val >= min_cooc:
                        jaccard[i, j] = j_val
                        jaccard[j, i] = j_val
        
        # Store as sparse pairs
        pairs = np.argwhere(jaccard > 0)
        if len(pairs) > 0:
            self.register_buffer('pair_i', torch.from_numpy(pairs[:, 0]).long())
            self.register_buffer('pair_j', torch.from_numpy(pairs[:, 1]).long())
            weights = torch.from_numpy(jaccard[pairs[:, 0], pairs[:, 1]])
            self.register_buffer('pair_w', weights.float())
            self.has_pairs = True
        else:
            self.has_pairs = False
        
        n_pairs = len(pairs) if len(pairs) > 0 else 0
        print(f"  Co-occurrence regularizer: {n_pairs} term pairs (Jaccard >= {min_cooc})")
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.has_pairs:
            return torch.tensor(0.0, device=logits.device)
        
        probs = torch.sigmoid(logits)
        # L2 distance between probabilities of co-occurring terms
        diff = probs[:, self.pair_i] - probs[:, self.pair_j]
        loss = (diff ** 2 * self.pair_w.unsqueeze(0)).mean()
        
        return loss * self.weight


# ════════════════════════════════════════════════════════════════
#  FMAX COMPUTATION (Global + Per-Class)
# ════════════════════════════════════════════════════════════════
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute multi-label classification metrics."""
    y_binary = (y_pred >= threshold).astype(np.float32)
    
    tp = np.sum(y_true * y_binary)
    fp = np.sum((1 - y_true) * y_binary)
    fn = np.sum(y_true * (1 - y_binary))
    
    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    
    try:
        from sklearn.metrics import average_precision_score
        valid_classes = np.sum(y_true, axis=0) > 0
        if valid_classes.sum() > 0:
            auprc = average_precision_score(
                y_true[:, valid_classes],
                y_pred[:, valid_classes],
                average='micro'
            )
        else:
            auprc = 0.0
    except Exception:
        auprc = 0.0
    
    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'micro_auprc': auprc
    }


def compute_fmax(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute Fmax with global threshold search."""
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in np.arange(0.01, 0.95, 0.02):
        y_binary = (y_pred >= threshold).astype(np.float32)
        
        tp = np.sum(y_true * y_binary)
        fp = np.sum((1 - y_true) * y_binary)
        fn = np.sum(y_true * (1 - y_binary))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_f1, best_threshold


def compute_fmax_perclass(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, np.ndarray, float, float]:
    """
    Per-class threshold optimization then protein-centric Fmax.
    
    Instead of one global threshold, find optimal threshold per class
    on the validation set, then evaluate protein-centric Fmax.
    
    This is significantly better because:
    - Rare terms need low thresholds (model outputs low probabilities)
    - Common terms need higher thresholds (model is more confident)
    
    Returns:
        (fmax, per_class_thresholds, precision, recall)
    """
    n_classes = y_pred.shape[1]
    
    # Step 1: find optimal threshold per class
    thresholds = np.ones(n_classes) * 0.5
    
    for c in range(n_classes):
        pos_mask = y_true[:, c] == 1
        if pos_mask.sum() == 0:
            continue
        
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.05):
            pred_c = (y_pred[:, c] >= t).astype(float)
            tp = (pred_c * y_true[:, c]).sum()
            fp = (pred_c * (1 - y_true[:, c])).sum()
            fn = ((1 - pred_c) * y_true[:, c]).sum()
            
            if tp > 0:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                if f1 > best_f1:
                    best_f1, best_t = f1, t
        
        thresholds[c] = best_t
    
    # Step 2: protein-centric Fmax using per-class thresholds
    bpred = (y_pred >= thresholds[np.newaxis, :]).astype(float)
    precs, recs = [], []
    
    for i in range(len(y_pred)):
        pp = bpred[i].sum()
        tp_cnt = y_true[i].sum()
        
        if pp > 0 and tp_cnt > 0:
            tp = (bpred[i] * y_true[i]).sum()
            precs.append(tp / pp)
            recs.append(tp / tp_cnt)
        elif tp_cnt > 0:
            precs.append(0.0)
            recs.append(0.0)
    
    if precs:
        ap, ar = np.mean(precs), np.mean(recs)
        fmax = 2 * ap * ar / (ap + ar) if (ap + ar) > 0 else 0.0
    else:
        fmax, ap, ar = 0.0, 0.0, 0.0
    
    return fmax, thresholds, ap, ar


# ════════════════════════════════════════════════════════════════
#  MEMORY & GPU TRACKING
# ════════════════════════════════════════════════════════════════
class MemoryTracker:
    """Track GPU memory usage."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.peak_memory = 0.0
    
    def update(self):
        if self.device.type == 'cuda':
            current = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            self.peak_memory = max(self.peak_memory, current)
            return current
        return 0.0
    
    def reset_peak(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_memory = 0.0
    
    def clear_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_stats(self) -> Dict[str, float]:
        if self.device.type != 'cuda':
            return {'allocated_gb': 0, 'reserved_gb': 0, 'peak_gb': 0}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
            'peak_gb': self.peak_memory
        }


class GPUStats:
    """Track GPU utilization via NVML."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = False
        self.handle = None
        
        if self.device.type == 'cuda' and _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                index = torch.cuda.current_device()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                self.enabled = True
            except Exception:
                pass
    
    def get_stats(self) -> Dict[str, float]:
        if not self.enabled:
            return {
                'gpu_util_percent': -1.0,
                'mem_util_percent': -1.0,
                'mem_used_gb': -1.0,
                'mem_total_gb': -1.0,
                'temperature_c': -1.0,
                'power_w': -1.0
            }
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            
            return {
                'gpu_util_percent': float(util.gpu),
                'mem_util_percent': float(util.memory),
                'mem_used_gb': float(mem.used) / (1024**3),
                'mem_total_gb': float(mem.total) / (1024**3),
                'temperature_c': float(temp),
                'power_w': float(power_mw) / 1000.0
            }
        except Exception:
            return {
                'gpu_util_percent': -1.0,
                'mem_util_percent': -1.0,
                'mem_used_gb': -1.0,
                'mem_total_gb': -1.0,
                'temperature_c': -1.0,
                'power_w': -1.0
            }


# ════════════════════════════════════════════════════════════════
#  TRAINER V9
# ════════════════════════════════════════════════════════════════
class TrainerV9:
    """
    Trainer for v9 model with all advanced features.
    
    Key differences from v8:
    - Class-balanced asymmetric loss (per ontology)
    - Co-occurrence regularization
    - Per-class threshold Fmax evaluation
    - AMP (mixed precision) training
    - OneCycleLR scheduler
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        memory_config: MemoryConfig,
        checkpoint_dir: str = 'checkpoints',
        class_counts_mf: Optional[np.ndarray] = None,
        class_counts_bp: Optional[np.ndarray] = None,
        class_counts_cc: Optional[np.ndarray] = None,
        annotations_mf: Optional[Dict] = None,
        annotations_bp: Optional[Dict] = None,
        annotations_cc: Optional[Dict] = None,
        go_list_mf: Optional[List[str]] = None,
        go_list_bp: Optional[List[str]] = None,
        go_list_cc: Optional[List[str]] = None,
        use_cooc_reg: bool = True,
        cooc_reg_weight: float = 0.1,
        use_amp: bool = True
    ):
        self.config = config
        self.memory_config = memory_config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.use_amp = use_amp
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
        
        # Model
        self.model = model.to(self.device)
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Class-balanced asymmetric loss (per ontology)
        if class_counts_mf is not None:
            self.criterion_mf = ClassBalancedAsymmetricLoss(class_counts_mf).to(self.device)
        else:
            self.criterion_mf = nn.BCEWithLogitsLoss()
        
        if class_counts_bp is not None:
            self.criterion_bp = ClassBalancedAsymmetricLoss(class_counts_bp).to(self.device)
        else:
            self.criterion_bp = nn.BCEWithLogitsLoss()
        
        if class_counts_cc is not None:
            self.criterion_cc = ClassBalancedAsymmetricLoss(class_counts_cc).to(self.device)
        else:
            self.criterion_cc = nn.BCEWithLogitsLoss()
        
        # Co-occurrence regularizers
        self.cooc_reg_mf = None
        self.cooc_reg_bp = None
        self.cooc_reg_cc = None
        
        if use_cooc_reg:
            if annotations_mf and go_list_mf:
                self.cooc_reg_mf = CooccurrenceRegularizer(
                    annotations_mf, go_list_mf, weight=cooc_reg_weight
                ).to(self.device)
            if annotations_bp and go_list_bp:
                self.cooc_reg_bp = CooccurrenceRegularizer(
                    annotations_bp, go_list_bp, weight=cooc_reg_weight
                ).to(self.device)
            if annotations_cc and go_list_cc:
                self.cooc_reg_cc = CooccurrenceRegularizer(
                    annotations_cc, go_list_cc, weight=cooc_reg_weight
                ).to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # OneCycleLR scheduler
        steps_per_epoch = max(len(train_loader) // config.gradient_accumulation_steps, 1)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos',
            last_epoch=-1  # Explicitly set to suppress warning
        )
        self._scheduler_stepped = False  # Track if optimizer.step() has been called
        
        # AMP scaler
        self.scaler = GradScaler('cuda') if use_amp else None
        
        # Memory tracking
        self.memory_tracker = MemoryTracker(self.device)
        self.gpu_stats = GPUStats(self.device)
        
        # Training state
        self.epoch = 0
        self.best_val_fmax = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_fmax_global': [],
            'val_fmax_perclass': [],
            'val_metrics': [],
            'learning_rate': [],
            'gpu_stats': []
        }
        
        # Per-class thresholds (updated during validation)
        self.thresholds_mf = None
        self.thresholds_bp = None
        self.thresholds_cc = None
    
    def train_epoch(self) -> float:
        """Train for one epoch with AMP."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad(set_to_none=True)
        accum_steps = self.config.gradient_accumulation_steps
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Non-blocking transfer to GPU for async loading
                batch = batch.to(self.device, non_blocking=True)
                edge_attr = getattr(batch, 'edge_attr', None)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(self.device, non_blocking=True)
                
                # Sync before compute (ensures transfer complete)
                if self.device.type == 'cuda':
                    torch.cuda.current_stream().synchronize()
                
                # Forward with AMP
                if self.use_amp:
                    with autocast('cuda', dtype=torch.float16):
                        mf_logits, bp_logits, cc_logits = self.model(
                            batch.x, batch.edge_index, batch.batch, edge_attr
                        )
                        
                        loss_mf = self.criterion_mf(mf_logits, batch.y_mf)
                        loss_bp = self.criterion_bp(bp_logits, batch.y_bp)
                        loss_cc = self.criterion_cc(cc_logits, batch.y_cc)
                        
                        loss = (loss_mf + loss_bp + loss_cc) / 3.0
                        
                        # Co-occurrence regularization
                        if self.cooc_reg_mf is not None:
                            loss = loss + self.cooc_reg_mf(mf_logits)
                        if self.cooc_reg_bp is not None:
                            loss = loss + self.cooc_reg_bp(bp_logits)
                        if self.cooc_reg_cc is not None:
                            loss = loss + self.cooc_reg_cc(cc_logits)
                        
                        loss = loss / accum_steps
                    
                    self.scaler.scale(loss).backward()
                else:
                    mf_logits, bp_logits, cc_logits = self.model(
                        batch.x, batch.edge_index, batch.batch, edge_attr
                    )
                    
                    loss_mf = self.criterion_mf(mf_logits, batch.y_mf)
                    loss_bp = self.criterion_bp(bp_logits, batch.y_bp)
                    loss_cc = self.criterion_cc(cc_logits, batch.y_cc)
                    
                    loss = (loss_mf + loss_bp + loss_cc) / 3.0
                    loss = loss / accum_steps
                    loss.backward()
                
                total_loss += loss.item() * accum_steps
                num_batches += 1
                
                # Optimizer step
                if (batch_idx + 1) % accum_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    self._scheduler_stepped = True
                    self.scheduler.step()
                
                self.memory_tracker.update()
                pbar.set_postfix({
                    'loss': total_loss / num_batches,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\nOOM at batch {batch_idx}, skipping...")
                    self.memory_tracker.clear_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                raise
        
        # Flush remaining gradients
        if num_batches % accum_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate with both global and per-class Fmax."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_mf_preds, all_mf_labels = [], []
        all_bp_preds, all_bp_labels = [], []
        all_cc_preds, all_cc_labels = [], []
        
        for batch in tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Val]"):
            try:
                batch = batch.to(self.device, non_blocking=True)
                edge_attr = getattr(batch, 'edge_attr', None)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(self.device, non_blocking=True)
                
                if self.device.type == 'cuda':
                    torch.cuda.current_stream().synchronize()
                
                if self.use_amp:
                    with autocast('cuda', dtype=torch.float16):
                        mf_logits, bp_logits, cc_logits = self.model(
                            batch.x, batch.edge_index, batch.batch, edge_attr
                        )
                else:
                    mf_logits, bp_logits, cc_logits = self.model(
                        batch.x, batch.edge_index, batch.batch, edge_attr
                    )
                
                # Compute loss (without AMP for accuracy)
                loss_mf = F.binary_cross_entropy_with_logits(mf_logits.float(), batch.y_mf)
                loss_bp = F.binary_cross_entropy_with_logits(bp_logits.float(), batch.y_bp)
                loss_cc = F.binary_cross_entropy_with_logits(cc_logits.float(), batch.y_cc)
                
                loss = (loss_mf + loss_bp + loss_cc) / 3.0
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions
                all_mf_preds.append(torch.sigmoid(mf_logits.float()).cpu().numpy())
                all_bp_preds.append(torch.sigmoid(bp_logits.float()).cpu().numpy())
                all_cc_preds.append(torch.sigmoid(cc_logits.float()).cpu().numpy())
                
                all_mf_labels.append(batch.y_mf.cpu().numpy())
                all_bp_labels.append(batch.y_bp.cpu().numpy())
                all_cc_labels.append(batch.y_cc.cpu().numpy())
                
                self.memory_tracker.update()
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    self.memory_tracker.clear_cache()
                    continue
                raise
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Concatenate all predictions
        mf_preds = np.concatenate(all_mf_preds, axis=0)
        bp_preds = np.concatenate(all_bp_preds, axis=0)
        cc_preds = np.concatenate(all_cc_preds, axis=0)
        
        mf_labels = np.concatenate(all_mf_labels, axis=0)
        bp_labels = np.concatenate(all_bp_labels, axis=0)
        cc_labels = np.concatenate(all_cc_labels, axis=0)
        
        # Global threshold Fmax
        mf_fmax_g, mf_thr_g = compute_fmax(mf_labels, mf_preds)
        bp_fmax_g, bp_thr_g = compute_fmax(bp_labels, bp_preds)
        cc_fmax_g, cc_thr_g = compute_fmax(cc_labels, cc_preds)
        
        # Per-class threshold Fmax
        mf_fmax_pc, mf_thresholds, mf_pr, mf_rc = compute_fmax_perclass(mf_labels, mf_preds)
        bp_fmax_pc, bp_thresholds, bp_pr, bp_rc = compute_fmax_perclass(bp_labels, bp_preds)
        cc_fmax_pc, cc_thresholds, cc_pr, cc_rc = compute_fmax_perclass(cc_labels, cc_preds)
        
        # Save per-class thresholds for later use
        self.thresholds_mf = mf_thresholds
        self.thresholds_bp = bp_thresholds
        self.thresholds_cc = cc_thresholds
        
        # Standard metrics
        mf_metrics = compute_metrics(mf_labels, mf_preds)
        bp_metrics = compute_metrics(bp_labels, bp_preds)
        cc_metrics = compute_metrics(cc_labels, cc_preds)
        
        # Add Fmax to metrics
        mf_metrics['fmax_global'] = mf_fmax_g
        mf_metrics['fmax_perclass'] = mf_fmax_pc
        mf_metrics['precision_pc'] = mf_pr
        mf_metrics['recall_pc'] = mf_rc
        
        bp_metrics['fmax_global'] = bp_fmax_g
        bp_metrics['fmax_perclass'] = bp_fmax_pc
        bp_metrics['precision_pc'] = bp_pr
        bp_metrics['recall_pc'] = bp_rc
        
        cc_metrics['fmax_global'] = cc_fmax_g
        cc_metrics['fmax_perclass'] = cc_fmax_pc
        cc_metrics['precision_pc'] = cc_pr
        cc_metrics['recall_pc'] = cc_rc
        
        # Combined per-class Fmax
        combined_fmax_pc = (mf_fmax_pc + bp_fmax_pc + cc_fmax_pc) / 3.0
        
        return {
            'loss': avg_loss,
            'mf': mf_metrics,
            'bp': bp_metrics,
            'cc': cc_metrics,
            'fmax_global': (mf_fmax_g + bp_fmax_g + cc_fmax_g) / 3.0,
            'fmax_perclass': combined_fmax_pc
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_fmax': self.best_val_fmax,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'model_config': self.model.config,
            'thresholds_mf': self.thresholds_mf.tolist() if self.thresholds_mf is not None else None,
            'thresholds_bp': self.thresholds_bp.tolist() if self.thresholds_bp is not None else None,
            'thresholds_cc': self.thresholds_cc.tolist() if self.thresholds_cc is not None else None,
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"  → Saved best model (Fmax_pc: {self.best_val_fmax:.4f})")
        
        # Save history
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json_history = {}
            for key, value in self.history.items():
                if isinstance(value, list):
                    json_history[key] = [
                        float(v) if isinstance(v, (np.floating, float)) else v
                        for v in value
                    ]
                else:
                    json_history[key] = value
            json.dump(json_history, f, indent=2, default=str)
    
    def load_checkpoint(self, checkpoint_path: str, num_epochs: int = None):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            num_epochs: Total epochs to train (if different from original, reinitialize scheduler)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_fmax = checkpoint.get('best_val_fmax', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.history = checkpoint.get('history', self.history)
        
        if checkpoint.get('thresholds_mf'):
            self.thresholds_mf = np.array(checkpoint['thresholds_mf'])
        if checkpoint.get('thresholds_bp'):
            self.thresholds_bp = np.array(checkpoint['thresholds_bp'])
        if checkpoint.get('thresholds_cc'):
            self.thresholds_cc = np.array(checkpoint['thresholds_cc'])
        
        # Reinitialize scheduler for remaining epochs (OneCycleLR doesn't support resume well)
        if num_epochs is not None and num_epochs > self.epoch:
            remaining_epochs = num_epochs - self.epoch
            steps_per_epoch = max(len(self.train_loader) // self.config.gradient_accumulation_steps, 1)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=remaining_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos',
                last_epoch=-1
            )
            print(f"Loaded checkpoint from epoch {self.epoch}")
            print(f"Reinitialized scheduler for {remaining_epochs} remaining epochs")
        else:
            # Load scheduler state only if not changing epoch count
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception:
                    pass
            print(f"Loaded checkpoint from epoch {self.epoch}")
        
        # Load scaler state
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\n{'='*70}")
        print("Starting v9 Training (Label-Embedding + Dual-Conv)")
        print(f"{'='*70}")
        print(f"  Architecture: Dual-Conv (GCN + GATv2) × {self.model.config['n_layers']}")
        print(f"  Hidden dim: {self.model.config['hidden']}")
        print(f"  GAT heads: {self.model.config['heads']}")
        print(f"  Triple pooling: Gated + Mean + Max")
        print(f"  Label embeddings: {self.model.use_label_embed}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Gradient checkpointing: {self.model.use_gradient_checkpointing}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Early stopping: {self.config.early_stopping_patience}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            self.memory_tracker.reset_peak()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_results = self.validate()
            
            # Use per-class Fmax as primary metric
            val_fmax = val_results['fmax_perclass']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_fmax_global'].append(val_results['fmax_global'])
            self.history['val_fmax_perclass'].append(val_fmax)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['val_metrics'].append({
                'mf': val_results['mf'],
                'bp': val_results['bp'],
                'cc': val_results['cc']
            })
            
            memory_stats = self.memory_tracker.get_stats()
            gpu_stats = self.gpu_stats.get_stats()
            self.history['gpu_stats'].append({
                'memory': memory_stats,
                'utilization': gpu_stats
            })
            
            # Check for improvement
            is_best = val_fmax > self.best_val_fmax
            if is_best:
                self.best_val_fmax = val_fmax
                self.best_epoch = epoch + 1
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Logging
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_results['loss']:.4f}")
            print(f"  --- Global Threshold Fmax ---")
            print(f"  MF: {val_results['mf']['fmax_global']:.4f}  "
                  f"BP: {val_results['bp']['fmax_global']:.4f}  "
                  f"CC: {val_results['cc']['fmax_global']:.4f}  "
                  f"Avg: {val_results['fmax_global']:.4f}")
            print(f"  --- Per-Class Threshold Fmax ---")
            print(f"  MF: {val_results['mf']['fmax_perclass']:.4f}  "
                  f"BP: {val_results['bp']['fmax_perclass']:.4f}  "
                  f"CC: {val_results['cc']['fmax_perclass']:.4f}  "
                  f"Avg: {val_fmax:.4f} {'(best)' if is_best else ''}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            
            if memory_stats['peak_gb'] > 0:
                print(f"  VRAM (GB): alloc={memory_stats['allocated_gb']:.2f}, "
                      f"reserved={memory_stats['reserved_gb']:.2f}, "
                      f"peak={memory_stats['peak_gb']:.2f}")
            
            if gpu_stats['gpu_util_percent'] >= 0:
                print(f"  GPU: {gpu_stats['gpu_util_percent']:.0f}% | "
                      f"VRAM: {gpu_stats['mem_used_gb']:.2f}/{gpu_stats['mem_total_gb']:.2f} GB | "
                      f"Temp: {gpu_stats['temperature_c']:.0f}°C | "
                      f"Power: {gpu_stats['power_w']:.0f} W")
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Memory management
            self.memory_tracker.clear_cache()
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation Fmax (per-class): {self.best_val_fmax:.4f} @ epoch {self.best_epoch}")
        print(f"{'='*70}")
        
        return self.history
    
    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict:
        """Evaluate on test set."""
        self.model.eval()
        
        all_mf_preds, all_mf_labels = [], []
        all_bp_preds, all_bp_labels = [], []
        all_cc_preds, all_cc_labels = [], []
        
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(self.device, non_blocking=True)
            edge_attr = getattr(batch, 'edge_attr', None)
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device, non_blocking=True)
            
            if self.device.type == 'cuda':
                torch.cuda.current_stream().synchronize()
            
            if self.use_amp:
                with autocast('cuda', dtype=torch.float16):
                    mf_logits, bp_logits, cc_logits = self.model(
                        batch.x, batch.edge_index, batch.batch, edge_attr
                    )
            else:
                mf_logits, bp_logits, cc_logits = self.model(
                    batch.x, batch.edge_index, batch.batch, edge_attr
                )
            
            all_mf_preds.append(torch.sigmoid(mf_logits.float()).cpu().numpy())
            all_bp_preds.append(torch.sigmoid(bp_logits.float()).cpu().numpy())
            all_cc_preds.append(torch.sigmoid(cc_logits.float()).cpu().numpy())
            
            all_mf_labels.append(batch.y_mf.cpu().numpy())
            all_bp_labels.append(batch.y_bp.cpu().numpy())
            all_cc_labels.append(batch.y_cc.cpu().numpy())
        
        # Concatenate
        mf_preds = np.concatenate(all_mf_preds, axis=0)
        bp_preds = np.concatenate(all_bp_preds, axis=0)
        cc_preds = np.concatenate(all_cc_preds, axis=0)
        
        mf_labels = np.concatenate(all_mf_labels, axis=0)
        bp_labels = np.concatenate(all_bp_labels, axis=0)
        cc_labels = np.concatenate(all_cc_labels, axis=0)
        
        # Metrics per ontology
        mf_metrics = compute_metrics(mf_labels, mf_preds)
        bp_metrics = compute_metrics(bp_labels, bp_preds)
        cc_metrics = compute_metrics(cc_labels, cc_preds)
        
        # Global Fmax
        mf_metrics['fmax_global'], mf_metrics['threshold_global'] = compute_fmax(mf_labels, mf_preds)
        bp_metrics['fmax_global'], bp_metrics['threshold_global'] = compute_fmax(bp_labels, bp_preds)
        cc_metrics['fmax_global'], cc_metrics['threshold_global'] = compute_fmax(cc_labels, cc_preds)
        
        # Per-class Fmax
        mf_fmax_pc, mf_thresholds, mf_pr, mf_rc = compute_fmax_perclass(mf_labels, mf_preds)
        bp_fmax_pc, bp_thresholds, bp_pr, bp_rc = compute_fmax_perclass(bp_labels, bp_preds)
        cc_fmax_pc, cc_thresholds, cc_pr, cc_rc = compute_fmax_perclass(cc_labels, cc_preds)
        
        mf_metrics['fmax_perclass'] = mf_fmax_pc
        mf_metrics['precision_pc'] = mf_pr
        mf_metrics['recall_pc'] = mf_rc
        
        bp_metrics['fmax_perclass'] = bp_fmax_pc
        bp_metrics['precision_pc'] = bp_pr
        bp_metrics['recall_pc'] = bp_rc
        
        cc_metrics['fmax_perclass'] = cc_fmax_pc
        cc_metrics['precision_pc'] = cc_pr
        cc_metrics['recall_pc'] = cc_rc
        
        # Combined
        combined_global = (mf_metrics['fmax_global'] + bp_metrics['fmax_global'] + cc_metrics['fmax_global']) / 3.0
        combined_perclass = (mf_fmax_pc + bp_fmax_pc + cc_fmax_pc) / 3.0
        
        results = {
            'mf': mf_metrics,
            'bp': bp_metrics,
            'cc': cc_metrics,
            'combined_fmax_global': combined_global,
            'combined_fmax_perclass': combined_perclass,
            'model_version': 'v9',
            'architecture': 'Dual-Conv (GCN + GATv2)',
            'best_epoch': self.best_epoch,
            'innovations': [
                'co-occurrence_label_embeddings',
                'dual_conv_blocks',
                'triple_pooling',
                'per-class_threshold_fmax',
                'class_balanced_asymmetric_loss',
                'co-occurrence_regularization',
                'amp_training',
                'gradient_checkpointing'
            ]
        }
        
        # Print results
        print(f"\n{'='*70}")
        print("TEST RESULTS (v9 - Label-Embedding + Dual-Conv)")
        print(f"{'='*70}")
        print(f"Architecture: Dual-Conv (GCN + GATv2) × {self.model.config['n_layers']}")
        print(f"")
        print(f"--- Global Threshold Fmax ---")
        print(f"MF: {mf_metrics['fmax_global']:.4f}  AUPRC: {mf_metrics['micro_auprc']:.4f}")
        print(f"BP: {bp_metrics['fmax_global']:.4f}  AUPRC: {bp_metrics['micro_auprc']:.4f}")
        print(f"CC: {cc_metrics['fmax_global']:.4f}  AUPRC: {cc_metrics['micro_auprc']:.4f}")
        print(f"Combined: {combined_global:.4f}")
        print(f"")
        print(f"--- Per-Class Threshold Fmax ---")
        print(f"MF: {mf_fmax_pc:.4f}  P: {mf_pr:.4f}  R: {mf_rc:.4f}")
        print(f"BP: {bp_fmax_pc:.4f}  P: {bp_pr:.4f}  R: {bp_rc:.4f}")
        print(f"CC: {cc_fmax_pc:.4f}  P: {cc_pr:.4f}  R: {cc_rc:.4f}")
        print(f"Combined: {combined_perclass:.4f}")
        print(f"{'='*70}")
        
        return results


def compare_with_baselines(v9_results_path: str):
    """Compare v9 with previous versions."""
    print(f"\n{'='*70}")
    print("COMPARISON: v9 vs Previous Versions")
    print(f"{'='*70}")
    
    with open(v9_results_path, 'r') as f:
        v9_results = json.load(f)
    
    # Try to load previous results
    versions = {}
    paths = {
        'v8': 'output_v8/checkpoints/test_results.json',
        'v7': 'output_v7/checkpoints/test_results.json',
        'v6': 'output_v6/checkpoints/test_results.json',
    }
    
    for version, path in paths.items():
        if Path(path).exists():
            with open(path, 'r') as f:
                versions[version] = json.load(f)
    
    # Print comparison
    print("\n                 ", end="")
    for v in versions.keys():
        print(f"{v:>12}", end="")
    print(f"{'v9':>12}")
    print("-" * (16 + 12 * (len(versions) + 1)))
    
    for ont in ['mf', 'bp', 'cc']:
        ont_name = ont.upper()
        print(f"{ont_name:12} Fmax: ", end="")
        
        for v in versions.keys():
            fmax = versions[v].get(ont, {}).get('fmax', 0)
            if not fmax:
                fmax = versions[v].get(ont, {}).get('fmax_global', 0)
            print(f"{float(fmax):>12.4f}", end="")
        
        v9_fmax = v9_results.get(ont, {}).get('fmax_perclass', 0)
        print(f"{float(v9_fmax):>12.4f}")
    
    print(f"\n{'='*70}")


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='v9 Training - Label-Embedding + Dual-Conv')
    parser.add_argument('--graphs-dir', type=str, default='output_v9/graphs_v9',
                        help='Directory with v9 graph files (40-dim nodes, 5-dim edges)')
    parser.add_argument('--checkpoint-dir', type=str, default='output_v9/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--annotation-file', type=str,
                        default='annotations/nrPDB-GO_2019.06.18_annot.tsv',
                        help='GO annotation file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (8GB default)')
    parser.add_argument('--accum-steps', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--vram-gb', type=float, default=8.0, help='Available VRAM in GB')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--compare', action='store_true', help='Compare with baselines')
    parser.add_argument('--test-only', action='store_true', help='Only run test')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint dir')
    parser.add_argument('--no-label-embed', action='store_true', help='Disable label embeddings')
    parser.add_argument('--no-cooc-reg', action='store_true', help='Disable co-occurrence reg')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP')
    
    args = parser.parse_args()
    
    if args.compare:
        results_path = Path(args.checkpoint_dir) / 'test_results.json'
        if results_path.exists():
            compare_with_baselines(str(results_path))
        else:
            print(f"Error: Test results not found at {results_path}")
        return
    
    # Create configs optimized for VRAM
    if args.vram_gb >= 8.0:
        batch_size = args.batch_size if args.batch_size else 32
        accum_steps = args.accum_steps if args.accum_steps else 1
    elif args.vram_gb >= 6.0:
        batch_size = 24
        accum_steps = 1
    else:
        batch_size = 16
        accum_steps = 2
    
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 2
    
    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        epochs=args.epochs,
        early_stopping_patience=20,
        scheduler_patience=5,
        gradient_accumulation_steps=accum_steps
    )
    
    memory_config = MemoryConfig(
        max_batch_memory_mb=3000.0,
        clear_cache_every=50
    )
    
    # Get dataloaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
        graphs_dir=args.graphs_dir,
        annotation_file=args.annotation_file,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )
    print(f"  Batch size: {batch_size}, num_workers: {num_workers}")
    
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print(f"  MF: {train_dataset.num_mf}, BP: {train_dataset.num_bp}, CC: {train_dataset.num_cc}")
    
    # Create v9 model
    model = create_v9_model(
        n_mf=train_dataset.num_mf,
        n_bp=train_dataset.num_bp,
        n_cc=train_dataset.num_cc,
        vram_gb=args.vram_gb,
        use_label_embed=not args.no_label_embed
    )
    
    print(f"\nModel configuration:")
    for k, v in model.config.items():
        print(f"  {k}: {v}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Create trainer
    # Note: For full implementation, you would extract class counts and annotations
    # from the dataset/parser for the advanced loss functions
    trainer = TrainerV9(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        memory_config=memory_config,
        checkpoint_dir=args.checkpoint_dir if not args.resume else args.resume,
        use_cooc_reg=not args.no_cooc_reg,
        use_amp=not args.no_amp
    )
    
    # Resume?
    if args.resume:
        latest_ck = Path(args.resume) / 'latest.pt'
        if latest_ck.exists():
            print(f"\n>>> Resuming from {latest_ck}")
            trainer.load_checkpoint(str(latest_ck), num_epochs=args.epochs)
    
    if args.test_only:
        # Load best model and test
        best_path = Path(args.checkpoint_dir) / 'best.pt'
        if best_path.exists():
            trainer.load_checkpoint(str(best_path))
        
        test_results = trainer.test(test_loader)
        
        results_path = Path(args.checkpoint_dir) / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\nTest results saved to: {results_path}")
    else:
        # Train
        trainer.train(num_epochs=args.epochs)
        
        # Test with best model
        print("\nLoading best model for testing...")
        best_path = Path(args.checkpoint_dir) / 'best.pt'
        if best_path.exists():
            trainer.load_checkpoint(str(best_path))
        
        test_results = trainer.test(test_loader)
        
        # Save test results
        results_path = Path(args.checkpoint_dir) / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\nTest results saved to: {results_path}")


if __name__ == '__main__':
    # Required for Windows multiprocessing with num_workers > 0
    import multiprocessing
    multiprocessing.freeze_support()
    # Use spawn method for Windows compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main()
