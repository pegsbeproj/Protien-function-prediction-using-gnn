"""
Training Script for v13 (Hierarchical Chain-Aware Pooling)

v13 = v12 + Hierarchical Pooling (Residue → Chain → Protein)

Changes from v12:
  1. Uses ProteinGNNv13 model with two-branch pooling
  2. CC head receives chain-level cross-attention context
  3. Toggle: --no-chain-pool for ablation (falls back to v12/v11 architecture)
  4. Tracks chain pooling statistics per epoch

Model Architecture: v11 backbone + v13 hierarchical pooling
  - Same ESM2 gated fusion, dual-conv backbone
  - NEW: Two-branch pooling (residue + chain)
  - NEW: CC-specific cross-attention on chain embeddings

Training Protocol: Identical to v12
  - ClassBalancedAsymmetricLoss per ontology (from v9)
  - HierarchicalConsistencyLoss per ontology (from v12)
  - Co-occurrence regularization (from v9)
  - AncestorPropagator at eval time (from v12)
  - OneCycleLR scheduler
  - AMP + gradient checkpointing
  - Per-class threshold Fmax
  - Early stopping (patience=20)

8GB VRAM:
  v13 adds ~450K params over v11 (~2.27M total). Fits 8 GB easily.
  Chain embeddings per batch: ~32 × 192 = 6 KB (negligible).

Usage:
    python train_v13.py --graphs-dir data/graphs --esm2-dir data/esm2_embeddings
    python train_v13.py --graphs-dir data/graphs --esm2-dir data/esm2_embeddings --resume output_v13/checkpoints
    python train_v13.py --test-only --checkpoint-dir output_v13/checkpoints
    python train_v13.py --no-chain-pool    # ablation: same as v12
    python train_v13.py --compare
"""

import gc
import json
import os
import time
import warnings

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
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except Exception:
    pynvml = None
    _PYNVML_AVAILABLE = False

from src.config import TrainingConfig, MemoryConfig

# Dataset (multi-chain + ESM2)
from src.data.multi_chain_dataset import get_dataloaders_v11 as get_dataloaders, ESM2EmbeddingLoader

# Model
from src.model.protein_gnn import (
    ProteinGNN,
    create_model,
    count_parameters,
    count_layer_parameters,
)

# Training utilities
from src.training.losses import (
    build_cooccurrence_embeddings,
    propagate_annotations,
    ClassBalancedAsymmetricLoss,
    CooccurrenceRegularizer,
    compute_metrics,
    compute_fmax,
    compute_fmax_perclass,
    MemoryTracker,
    GPUStats,
)

# GO hierarchy
from src.data.go_hierarchy import (
    GOHierarchy,
    HierarchicalConsistencyLoss,
    AncestorPropagator,
    download_obo,
)

# Base trainer
from src.training.trainer_base import TrainerV12 as TrainerBase


# ════════════════════════════════════════════════════════════════
#  TRAINER V13 (extends v12 with chain-pool tracking)
# ════════════════════════════════════════════════════════════════

class Trainer(TrainerBase):
    """
    Trainer for v13 model.

    Inherits ALL training logic from TrainerV12 (hierarchy loss,
    ancestor propagation, etc.) and adds:
      1. Chain pooling statistics tracking
      2. Version tag 'v13' in checkpoints
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add v13-specific history keys
        if 'chain_pool_stats' not in self.history:
            self.history['chain_pool_stats'] = []

    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint with v13 version tag."""
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
            'version': 'v13',
        }

        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"  → Saved best model (Fmax_best: {self.best_val_fmax:.4f})")

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

    def train(self, num_epochs: int):
        """Full training loop — same as v12, plus chain pool stats logging."""
        print(f"\n{'='*70}")
        print("Starting v13 Training (Hierarchical Chain-Aware Pooling)")
        print(f"{'='*70}")
        print(f"  Architecture: Dual-Conv (GCN + GATv2) × {self.model.config['n_layers']}")
        print(f"  Hidden dim: {self.model.config['hidden']}")
        print(f"  GAT heads: {self.model.config['heads']}")
        print(f"  Chain embedding: {self.model.config['chain_emb_dim']}-dim")
        print(f"  ESM2 dim: {self.model.config['esm_dim']}")
        print(f"  ESM2 fusion: Gated (learned per-node)")
        print(f"  Max chains: {self.model.config['max_chains']}")
        print(f"  Chain-aware pooling: {self.model.config.get('use_chain_pool', False)}")
        print(f"  Triple pooling: Gated + Mean + Max (residue-level)")
        print(f"  Label embeddings: {self.model.use_label_embed}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Gradient checkpointing: {self.model.use_gradient_checkpointing}")
        print(f"  Hierarchy loss: MF={self.hier_loss_mf is not None}, "
              f"BP={self.hier_loss_bp is not None}, CC={self.hier_loss_cc is not None}")
        print(f"  Ancestor propagation: MF={self.propagator_mf is not None}, "
              f"BP={self.propagator_bp is not None}, CC={self.propagator_cc is not None}")
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

            train_loss = self.train_epoch()

            # Clear caches before validation (prevents Windows process kill)
            self.train_loader.dataset._cache.clear()
            self.train_loader.dataset._batch_cache.clear()
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"  [Pre-val] GPU: {allocated:.2f} GB alloc, {reserved:.2f} GB reserved")

            val_results = self.validate()

            # Clear val caches
            self.val_loader.dataset._cache.clear()
            self.val_loader.dataset._batch_cache.clear()

            # Use the best of raw vs propagated Fmax for model selection
            val_fmax = val_results['fmax_best']
            val_fmax_raw = val_results['fmax_perclass']
            val_fmax_prop = val_results['fmax_propagated']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_fmax_global'].append(val_results['fmax_global'])
            self.history['val_fmax_perclass'].append(val_fmax_raw)
            self.history['val_fmax_propagated'].append(val_fmax_prop)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['val_metrics'].append({
                'mf': val_results['mf'],
                'bp': val_results['bp'],
                'cc': val_results['cc']
            })

            # ESM2 tracking
            esm_coverage = self._esm_batches / max(1, self._total_batches)
            self.history['esm2_coverage'].append(esm_coverage)

            gate_stats = self.model.get_esm_gate_stats() if hasattr(self.model, 'get_esm_gate_stats') else {}
            self.history['esm2_gate_stats'].append(gate_stats)

            # v12: Hierarchy violation tracking
            hier_stats = {}
            if self.hier_loss_mf and hasattr(self.hier_loss_mf, 'has_edges') and self.hier_loss_mf.has_edges:
                hier_stats['mf_edges'] = len(self.hier_loss_mf.child_idx)
            if self.hier_loss_bp and hasattr(self.hier_loss_bp, 'has_edges') and self.hier_loss_bp.has_edges:
                hier_stats['bp_edges'] = len(self.hier_loss_bp.child_idx)
            if self.hier_loss_cc and hasattr(self.hier_loss_cc, 'has_edges') and self.hier_loss_cc.has_edges:
                hier_stats['cc_edges'] = len(self.hier_loss_cc.child_idx)
            hier_stats['propagation_gain'] = val_fmax_prop - val_fmax_raw
            self.history['hier_violation_stats'].append(hier_stats)

            # v13 NEW: Chain pooling stats
            chain_stats = {}
            if hasattr(self.model, 'get_chain_pool_stats'):
                chain_stats = self.model.get_chain_pool_stats()
            self.history['chain_pool_stats'].append(chain_stats)

            memory_stats = self.memory_tracker.get_stats()
            gpu_stats = self.gpu_stats.get_stats()
            self.history['gpu_stats'].append({
                'memory': memory_stats,
                'utilization': gpu_stats
            })

            is_best = val_fmax > self.best_val_fmax
            if is_best:
                self.best_val_fmax = val_fmax
                self.best_epoch = epoch + 1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_results['loss']:.4f}")
            print(f"  --- Global Threshold Fmax ---")
            print(f"  MF: {val_results['mf']['fmax_global']:.4f}  "
                  f"BP: {val_results['bp']['fmax_global']:.4f}  "
                  f"CC: {val_results['cc']['fmax_global']:.4f}  "
                  f"Avg: {val_results['fmax_global']:.4f}")
            print(f"  --- Per-Class Threshold Fmax (raw) ---")
            print(f"  MF: {val_results['mf']['fmax_perclass']:.4f}  "
                  f"BP: {val_results['bp']['fmax_perclass']:.4f}  "
                  f"CC: {val_results['cc']['fmax_perclass']:.4f}  "
                  f"Avg: {val_fmax_raw:.4f}")
            print(f"  --- Per-Class Threshold Fmax (propagated) ---")
            print(f"  MF: {val_results['mf']['fmax_propagated']:.4f}  "
                  f"BP: {val_results['bp']['fmax_propagated']:.4f}  "
                  f"CC: {val_results['cc']['fmax_propagated']:.4f}  "
                  f"Avg: {val_fmax_prop:.4f} (Δ{val_fmax_prop - val_fmax_raw:+.4f})")
            print(f"  Best Fmax: {val_fmax:.4f} {'(best)' if is_best else ''}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            print(f"  ESM2 coverage: {100*esm_coverage:.1f}% of batches")
            if gate_stats:
                print(f"  ESM2 gate stats: scale={gate_stats.get('esm_scale', 0):.3f}, "
                      f"bias_mean={gate_stats.get('gate_bias_mean', 0):.3f}")
            if chain_stats:
                print(f"  Chain pool: chain_gate_bias={chain_stats.get('chain_attn_gate_bias_mean', 0):.3f}, "
                      f"protein_gate_bias={chain_stats.get('protein_chain_gate_bias_mean', 0):.3f}")

            if memory_stats['peak_gb'] > 0:
                print(f"  VRAM (GB): alloc={memory_stats['allocated_gb']:.2f}, "
                      f"reserved={memory_stats['reserved_gb']:.2f}, "
                      f"peak={memory_stats['peak_gb']:.2f}")

            if gpu_stats['gpu_util_percent'] >= 0:
                print(f"  GPU: {gpu_stats['gpu_util_percent']:.0f}% | "
                      f"VRAM: {gpu_stats['mem_used_gb']:.2f}/{gpu_stats['mem_total_gb']:.2f} GB | "
                      f"Temp: {gpu_stats['temperature_c']:.0f}°C | "
                      f"Power: {gpu_stats['power_w']:.0f} W")

            self.save_checkpoint(is_best=is_best)

            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            self.memory_tracker.clear_cache()

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation Fmax (best of raw/propagated): {self.best_val_fmax:.4f} @ epoch {self.best_epoch}")
        print(f"{'='*70}")

        return self.history

    @torch.no_grad()
    def test(self, test_loader) -> Dict:
        """Evaluate on test set with ancestor propagation (v13-tagged)."""
        self.model.eval()

        all_mf_preds, all_mf_labels = [], []
        all_bp_preds, all_bp_labels = [], []
        all_cc_preds, all_cc_labels = [], []
        esm_batches = 0
        total_batches = 0

        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            batch = batch.to(self.device, non_blocking=True)
            edge_attr = getattr(batch, 'edge_attr', None)
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device, non_blocking=True)

            has_esm = getattr(batch, 'esm_emb', None) is not None
            if has_esm:
                esm_batches += 1
            total_batches += 1

            if self.use_amp:
                with autocast('cuda', dtype=torch.float16):
                    mf_logits, bp_logits, cc_logits = self._model_forward(batch, edge_attr)
            else:
                mf_logits, bp_logits, cc_logits = self._model_forward(batch, edge_attr)

            all_mf_preds.append(torch.sigmoid(mf_logits.float()).cpu().numpy())
            all_bp_preds.append(torch.sigmoid(bp_logits.float()).cpu().numpy())
            all_cc_preds.append(torch.sigmoid(cc_logits.float()).cpu().numpy())

            all_mf_labels.append(batch.y_mf.cpu().numpy())
            all_bp_labels.append(batch.y_bp.cpu().numpy())
            all_cc_labels.append(batch.y_cc.cpu().numpy())

            del mf_logits, bp_logits, cc_logits, batch, edge_attr

        mf_preds = np.concatenate(all_mf_preds, axis=0)
        bp_preds = np.concatenate(all_bp_preds, axis=0)
        cc_preds = np.concatenate(all_cc_preds, axis=0)

        mf_labels = np.concatenate(all_mf_labels, axis=0)
        bp_labels = np.concatenate(all_bp_labels, axis=0)
        cc_labels = np.concatenate(all_cc_labels, axis=0)

        # ── Raw metrics ──
        mf_metrics = compute_metrics(mf_labels, mf_preds)
        bp_metrics = compute_metrics(bp_labels, bp_preds)
        cc_metrics = compute_metrics(cc_labels, cc_preds)

        mf_metrics['fmax_global'], mf_metrics['threshold_global'] = compute_fmax(mf_labels, mf_preds)
        bp_metrics['fmax_global'], bp_metrics['threshold_global'] = compute_fmax(bp_labels, bp_preds)
        cc_metrics['fmax_global'], cc_metrics['threshold_global'] = compute_fmax(cc_labels, cc_preds)

        mf_fmax_pc, mf_thresholds, mf_pr, mf_rc = compute_fmax_perclass(mf_labels, mf_preds)
        bp_fmax_pc, bp_thresholds, bp_pr, bp_rc = compute_fmax_perclass(bp_labels, bp_preds)
        cc_fmax_pc, cc_thresholds, cc_pr, cc_rc = compute_fmax_perclass(cc_labels, cc_preds)

        mf_metrics.update({'fmax_perclass': mf_fmax_pc, 'precision_pc': mf_pr, 'recall_pc': mf_rc})
        bp_metrics.update({'fmax_perclass': bp_fmax_pc, 'precision_pc': bp_pr, 'recall_pc': bp_rc})
        cc_metrics.update({'fmax_perclass': cc_fmax_pc, 'precision_pc': cc_pr, 'recall_pc': cc_rc})

        # ── Propagated metrics ──
        mf_preds_prop = self.propagator_mf.propagate(mf_preds) if self.propagator_mf else mf_preds
        bp_preds_prop = self.propagator_bp.propagate(bp_preds) if self.propagator_bp else bp_preds
        cc_preds_prop = self.propagator_cc.propagate(cc_preds) if self.propagator_cc else cc_preds

        mf_fmax_prop, _, mf_pr_prop, mf_rc_prop = compute_fmax_perclass(mf_labels, mf_preds_prop)
        bp_fmax_prop, _, bp_pr_prop, bp_rc_prop = compute_fmax_perclass(bp_labels, bp_preds_prop)
        cc_fmax_prop, _, cc_pr_prop, cc_rc_prop = compute_fmax_perclass(cc_labels, cc_preds_prop)

        mf_metrics.update({
            'fmax_propagated': mf_fmax_prop,
            'precision_prop': mf_pr_prop, 'recall_prop': mf_rc_prop,
        })
        bp_metrics.update({
            'fmax_propagated': bp_fmax_prop,
            'precision_prop': bp_pr_prop, 'recall_prop': bp_rc_prop,
        })
        cc_metrics.update({
            'fmax_propagated': cc_fmax_prop,
            'precision_prop': cc_pr_prop, 'recall_prop': cc_rc_prop,
        })

        combined_global = (mf_metrics['fmax_global'] + bp_metrics['fmax_global'] + cc_metrics['fmax_global']) / 3.0
        combined_raw = (mf_fmax_pc + bp_fmax_pc + cc_fmax_pc) / 3.0
        combined_prop = (mf_fmax_prop + bp_fmax_prop + cc_fmax_prop) / 3.0
        combined_best = max(combined_raw, combined_prop)

        esm_test_coverage = esm_batches / max(1, total_batches)

        results = {
            'mf': mf_metrics,
            'bp': bp_metrics,
            'cc': cc_metrics,
            'combined_fmax_global': combined_global,
            'combined_fmax_perclass': combined_raw,
            'combined_fmax_propagated': combined_prop,
            'combined_fmax_best': combined_best,
            'model_version': 'v13',
            'architecture': 'Hierarchical Chain-Aware Pooling (Residue→Chain→Protein) + ESM2 + GO-DAG',
            'best_epoch': self.best_epoch,
            'esm2_test_coverage': esm_test_coverage,
            'esm2_dim': self.model.config.get('esm_dim', 0),
            'use_chain_pool': self.model.config.get('use_chain_pool', False),
            'innovations': [
                'hierarchical_chain_aware_pooling',
                'chain_attention_pool',
                'protein_chain_pool',
                'cc_context_cross_attention',
                'two_branch_residue_chain_pooling',
                'go_dag_hierarchical_consistency_loss',
                'ancestor_propagation_postprocessing',
                'esm2_per_residue_embeddings',
                'gated_fusion_esm2_handcrafted',
                'multi_chain_graph_per_pdb',
                'inter_chain_edges',
                'learned_chain_embedding',
                'neighbor_hard_cap',
                'co-occurrence_label_embeddings',
                'dual_conv_blocks',
                'triple_pooling',
                'per-class_threshold_fmax',
                'class_balanced_asymmetric_loss',
                'co-occurrence_regularization',
                'amp_training',
                'gradient_checkpointing',
            ]
        }

        print(f"\n{'='*70}")
        print("TEST RESULTS (v13 - Hierarchical Chain-Aware Pooling)")
        print(f"{'='*70}")
        print(f"Architecture: Dual-Conv (GCN + GATv2) × {self.model.config['n_layers']} + "
              f"Chain Emb ({self.model.config['chain_emb_dim']}-d) + "
              f"ESM2 ({self.model.config['esm_dim']}-d) + "
              f"Chain Pool ({self.model.config.get('use_chain_pool', False)}) + GO-DAG")
        print(f"ESM2 test coverage: {100*esm_test_coverage:.1f}%")
        print(f"")
        print(f"--- Global Threshold Fmax ---")
        print(f"MF: {mf_metrics['fmax_global']:.4f}  AUPRC: {mf_metrics['micro_auprc']:.4f}")
        print(f"BP: {bp_metrics['fmax_global']:.4f}  AUPRC: {bp_metrics['micro_auprc']:.4f}")
        print(f"CC: {cc_metrics['fmax_global']:.4f}  AUPRC: {cc_metrics['micro_auprc']:.4f}")
        print(f"Combined: {combined_global:.4f}")
        print(f"")
        print(f"--- Per-Class Threshold Fmax (raw) ---")
        print(f"MF: {mf_fmax_pc:.4f}  P: {mf_pr:.4f}  R: {mf_rc:.4f}")
        print(f"BP: {bp_fmax_pc:.4f}  P: {bp_pr:.4f}  R: {bp_rc:.4f}")
        print(f"CC: {cc_fmax_pc:.4f}  P: {cc_pr:.4f}  R: {cc_rc:.4f}")
        print(f"Combined: {combined_raw:.4f}")
        print(f"")
        print(f"--- Per-Class Threshold Fmax (propagated) ---")
        print(f"MF: {mf_fmax_prop:.4f}  P: {mf_pr_prop:.4f}  R: {mf_rc_prop:.4f}")
        print(f"BP: {bp_fmax_prop:.4f}  P: {bp_pr_prop:.4f}  R: {bp_rc_prop:.4f}")
        print(f"CC: {cc_fmax_prop:.4f}  P: {cc_pr_prop:.4f}  R: {cc_rc_prop:.4f}")
        print(f"Combined: {combined_prop:.4f} (Δ{combined_prop - combined_raw:+.4f} from propagation)")
        print(f"")
        print(f"Best Combined Fmax: {combined_best:.4f}")
        print(f"{'='*70}")

        return results


def compare_with_baselines(v13_results_path: str):
    """Compare v13 with previous versions."""
    print(f"\n{'='*70}")
    print("COMPARISON: v13 vs Previous Versions")
    print(f"{'='*70}")

    with open(v13_results_path, 'r') as f:
        v13_results = json.load(f)

    versions = {}
    paths = {
        'v7': 'output_v7/checkpoints/test_results.json',
        'v9': 'output_v9/checkpoints/test_results.json',
        'v10': 'output_v10/checkpoints/test_results.json',
        'v11': 'output_v11/checkpoints/test_results.json',
        'v12': 'output_v12/checkpoints/test_results.json',
    }

    for version, path in paths.items():
        if Path(path).exists():
            with open(path, 'r') as f:
                versions[version] = json.load(f)

    print(f"\n{'':16}", end="")
    for v in versions.keys():
        print(f"{v:>12}", end="")
    print(f"{'v13 raw':>12}{'v13 prop':>12}")
    print("-" * (16 + 12 * (len(versions) + 2)))

    for ont in ['mf', 'bp', 'cc']:
        print(f"{ont.upper():12} Fmax: ", end="")

        for v in versions.keys():
            fmax = versions[v].get(ont, {}).get('fmax_perclass',
                   versions[v].get(ont, {}).get('fmax', 0))
            print(f"{float(fmax):>12.4f}", end="")

        v13_raw = v13_results.get(ont, {}).get('fmax_perclass', 0)
        v13_prop = v13_results.get(ont, {}).get('fmax_propagated', v13_raw)
        print(f"{float(v13_raw):>12.4f}{float(v13_prop):>12.4f}")

    # Combined
    print(f"\n{'Combined':12}      ", end="")
    for v in versions.keys():
        res = versions[v]
        vals = []
        for ont in ['mf', 'bp', 'cc']:
            f = res.get(ont, {}).get('fmax_perclass', res.get(ont, {}).get('fmax', 0))
            vals.append(float(f))
        print(f"{np.mean(vals):>12.4f}", end="")

    v13_raw_combined = float(v13_results.get('combined_fmax_perclass', 0))
    v13_prop_combined = float(v13_results.get('combined_fmax_propagated', v13_raw_combined))
    print(f"{v13_raw_combined:>12.4f}{v13_prop_combined:>12.4f}")

    # Deltas vs v12
    if "v12" in versions:
        print(f"\n  --- v12 → v13 Delta (Hierarchical Chain-Aware Pooling) ---")
        for ont in ['mf', 'bp', 'cc']:
            v12_fmax = float(versions['v12'].get(ont, {}).get('fmax_propagated',
                            versions['v12'].get(ont, {}).get('fmax_perclass',
                            versions['v12'].get(ont, {}).get('fmax', 0))) or 0)
            v13_fmax = float(v13_results.get(ont, {}).get('fmax_propagated',
                             v13_results.get(ont, {}).get('fmax_perclass', 0)) or 0)
            delta = v13_fmax - v12_fmax
            direction = "+" if delta > 0 else ""
            print(f"    {ont.upper()}: {v12_fmax:.4f} → {v13_fmax:.4f} ({direction}{delta:.4f})")

    print(f"{'='*70}")


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='v13 Training - Hierarchical Chain-Aware Pooling')
    parser.add_argument('--graphs-dir', type=str, default='data/graphs',
                        help='Directory with graph batch files')
    parser.add_argument('--esm2-dir', type=str, default='data/esm2_embeddings',
                        help='Directory with ESM2 embedding files')
    parser.add_argument('--esm2-dim', type=int, default=1280,
                        help=f'ESM2 embedding dimension (default: {V13_ESM2_DIM})')
    parser.add_argument('--obo-file', type=str, default='data/annotations/go-basic.obo',
                        help='Path to GO OBO file')
    parser.add_argument('--checkpoint-dir', type=str, default='output_v13/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--annotation-file', type=str,
                        default='data/annotations/nrPDB-GO_2019.06.18_annot.tsv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--vram-gb', type=float, default=8.0)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hier-weight', type=float, default=0.05,
                        help='Weight for hierarchical consistency loss (default: 0.05)')
    parser.add_argument('--hier-margin', type=float, default=0.0,
                        help='Margin for hierarchy violation (default: 0.0)')
    # v13 specific
    parser.add_argument('--no-chain-pool', action='store_true',
                        help='Disable chain-aware pooling (ablation: same as v12)')
    # General flags
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-label-embed', action='store_true')
    parser.add_argument('--no-cooc-reg', action='store_true')
    parser.add_argument('--no-hier-loss', action='store_true',
                        help='Disable hierarchy loss')
    parser.add_argument('--no-propagation', action='store_true',
                        help='Disable ancestor propagation at eval')
    parser.add_argument('--no-amp', action='store_true')

    args = parser.parse_args()

    if args.compare:
        results_path = Path(args.checkpoint_dir) / 'test_results.json'
        if results_path.exists():
            compare_with_baselines(str(results_path))
        else:
            print(f"Error: Test results not found at {results_path}")
        return

    # VRAM-based batch size
    if args.vram_gb >= 8.0:
        batch_size = args.batch_size if args.batch_size else 16
        accum_steps = args.accum_steps if args.accum_steps else 1
    elif args.vram_gb >= 6.0:
        batch_size = 12
        accum_steps = 1
    else:
        batch_size = 8
        accum_steps = 2

    num_workers = args.num_workers

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

    # ── Load GO hierarchy ──
    print("\n" + "=" * 60)
    print("Loading GO Hierarchy (DAG)")
    print("=" * 60)

    if not Path(args.obo_file).exists():
        print(f"  OBO file not found at: {args.obo_file}")
        print(f"  Attempting download...")
        download_obo(args.obo_file)

    go_hierarchy = GOHierarchy(args.obo_file)

    # ── Get dataloaders (same as v11/v12) ──
    print("\nCreating v13 data loaders (v10 graphs + ESM2 embeddings)...")
    eval_batch_size = max(batch_size // 2, 1)
    train_loader, val_loader, test_loader, train_dataset, esm2_loader = get_dataloaders_v11(
        graphs_dir=args.graphs_dir,
        annotation_file=args.annotation_file,
        esm2_dir=args.esm2_dir,
        esm2_dim=args.esm2_dim,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )
    print(f"  Batch size: {batch_size} (eval: {eval_batch_size}), num_workers: {num_workers}")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print(f"  MF: {train_dataset.num_mf}, BP: {train_dataset.num_bp}, CC: {train_dataset.num_cc}")
    if esm2_loader:
        print(f"  ESM2: dim={esm2_loader.esm_dim}, available={esm2_loader.available}")
    else:
        print(f"  ESM2: NOT AVAILABLE (running in v10 fallback mode)")

    esm_dim = esm2_loader.esm_dim if esm2_loader else args.esm2_dim

    # ── Verify ancestor closure ──
    print("\n" + "=" * 60)
    print("Verifying Ancestor Closure in Annotations")
    print("=" * 60)
    closure_results = go_hierarchy.verify_ancestor_closure(
        train_dataset.parser.annotations,
        train_dataset.parser.mf_terms,
        train_dataset.parser.bp_terms,
        train_dataset.parser.cc_terms,
    )

    # ── Build hierarchy losses and propagators ──
    print("\n" + "=" * 60)
    print("Building GO-DAG Components")
    print("=" * 60)

    hier_loss_mf = hier_loss_bp = hier_loss_cc = None
    propagator_mf = propagator_bp = propagator_cc = None

    if not args.no_hier_loss:
        print(f"\n  Building hierarchy losses (weight={args.hier_weight})...")

        mf_c, mf_p = go_hierarchy.get_edges(train_dataset.parser.mf_terms)
        bp_c, bp_p = go_hierarchy.get_edges(train_dataset.parser.bp_terms)
        cc_c, cc_p = go_hierarchy.get_edges(train_dataset.parser.cc_terms)

        hier_loss_mf = HierarchicalConsistencyLoss(mf_c, mf_p, weight=args.hier_weight, margin=args.hier_margin)
        hier_loss_bp = HierarchicalConsistencyLoss(bp_c, bp_p, weight=args.hier_weight, margin=args.hier_margin)
        hier_loss_cc = HierarchicalConsistencyLoss(cc_c, cc_p, weight=args.hier_weight, margin=args.hier_margin)
    else:
        print("\n  Hierarchy loss DISABLED (--no-hier-loss)")

    if not args.no_propagation:
        print(f"\n  Building ancestor propagators...")

        mf_anc = go_hierarchy.get_ancestor_matrix(train_dataset.parser.mf_terms)
        bp_anc = go_hierarchy.get_ancestor_matrix(train_dataset.parser.bp_terms)
        cc_anc = go_hierarchy.get_ancestor_matrix(train_dataset.parser.cc_terms)

        propagator_mf = AncestorPropagator(mf_anc)
        propagator_bp = AncestorPropagator(bp_anc)
        propagator_cc = AncestorPropagator(cc_anc)
    else:
        print("\n  Ancestor propagation DISABLED (--no-propagation)")

    # ── Create v13 model ──
    use_chain_pool = not args.no_chain_pool
    print(f"\n  Creating v13 model (chain-aware pooling: {use_chain_pool})...")

    model = create_v13_model(
        n_mf=train_dataset.num_mf,
        n_bp=train_dataset.num_bp,
        n_cc=train_dataset.num_cc,
        vram_gb=args.vram_gb,
        esm_dim=esm_dim,
        use_label_embed=not args.no_label_embed,
        use_chain_pool=use_chain_pool,
    )

    print(f"\nModel configuration (v13):")
    for k, v in model.config.items():
        print(f"  {k}: {v}")
    print(f"Total parameters: {count_parameters(model):,}")

    print(f"\nParameter breakdown:")
    for name, count in count_layer_parameters(model).items():
        pct = 100.0 * count / count_parameters(model)
        print(f"  {name}: {count:,} ({pct:.1f}%)")

    # ── Build co-occurrence annotations for regularization ──
    cooc_annotations = train_dataset.parser.annotations
    annotations_mf = {gid: annot.get('mf', []) for gid, annot in cooc_annotations.items()}
    annotations_bp = {gid: annot.get('bp', []) for gid, annot in cooc_annotations.items()}
    annotations_cc = {gid: annot.get('cc', []) for gid, annot in cooc_annotations.items()}

    # ── Create trainer ──
    trainer = TrainerV13(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        memory_config=memory_config,
        checkpoint_dir=args.checkpoint_dir if not args.resume else args.resume,
        annotations_mf=annotations_mf,
        annotations_bp=annotations_bp,
        annotations_cc=annotations_cc,
        go_list_mf=train_dataset.parser.mf_terms,
        go_list_bp=train_dataset.parser.bp_terms,
        go_list_cc=train_dataset.parser.cc_terms,
        use_cooc_reg=not args.no_cooc_reg,
        use_amp=not args.no_amp,
        # v12 hierarchy
        hier_loss_mf=hier_loss_mf,
        hier_loss_bp=hier_loss_bp,
        hier_loss_cc=hier_loss_cc,
        propagator_mf=propagator_mf,
        propagator_bp=propagator_bp,
        propagator_cc=propagator_cc,
    )

    if args.resume:
        latest_ck = Path(args.resume) / 'latest.pt'
        if latest_ck.exists():
            print(f"\n>>> Resuming from {latest_ck}")
            trainer.load_checkpoint(str(latest_ck), num_epochs=args.epochs)

    if args.test_only:
        best_path = Path(args.checkpoint_dir) / 'best.pt'
        if best_path.exists():
            trainer.load_checkpoint(str(best_path))

        test_results = trainer.test(test_loader)

        results_path = Path(args.checkpoint_dir) / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)

        print(f"\nTest results saved to: {results_path}")
    else:
        trainer.train(num_epochs=args.epochs)

        print("\nLoading best model for testing...")
        best_path = Path(args.checkpoint_dir) / 'best.pt'
        if best_path.exists():
            trainer.load_checkpoint(str(best_path))

        test_results = trainer.test(test_loader)

        results_path = Path(args.checkpoint_dir) / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)

        print(f"\nTest results saved to: {results_path}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()
