"""
v13 Pipeline Orchestrator - Hierarchical Chain-Aware Pooling

Runs the complete v13 experiment:
  Step 0: Download / verify GO OBO file
  Step 1: Verify ancestor closure in annotations
  Step 2: Train v13 model (hierarchical chain→protein pooling + GO-DAG)
  Step 3: Evaluate and compare with v12 and baselines

v13 reuses:
  - v10 graph structure      (from output_v10/graphs_v10)
  - v11 ESM2 embeddings      (from output_v11/esm2_embeddings)

v13 model adds over v12:
  - Two-branch pooling: residue-level (preserved) + chain-aware (new)
  - ChainAttentionPool:  residues → chain embeddings
  - ProteinChainPool:    chain embeddings → protein chain embedding
  - CCContextAttention:  cross-attention from protein to chains for CC head
  - Ablation toggle:     --no-chain-pool falls back to v12/v11 architecture

Usage:
    # Full pipeline:
    python run_pipeline_v13.py --output-dir output_v13 --graphs-dir output_v10/graphs_v10

    # Resume training:
    python run_pipeline_v13.py --output-dir output_v13 --graphs-dir output_v10/graphs_v10 --resume

    # Test only (compare):
    python run_pipeline_v13.py --test-only --output-dir output_v13

    # Compare only:
    python run_pipeline_v13.py --compare-only --output-dir output_v13

    # Ablation: no chain pooling (equivalent to v12):
    python run_pipeline_v13.py --no-chain-pool --output-dir output_v13_ablation

    # Ablation: no hierarchy loss:
    python run_pipeline_v13.py --no-hier-loss --output-dir output_v13

    # Ablation: no propagation:
    python run_pipeline_v13.py --no-propagation --output-dir output_v13
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# =========================================================================
# STEP 0: Download / Verify GO OBO File
# =========================================================================

def run_obo_setup(obo_file: str) -> dict:
    """Download GO OBO file if needed and verify it."""
    print("\n" + "=" * 60)
    print("STEP 0: Download / Verify GO OBO File")
    print("=" * 60)

    from src.data.go_hierarchy import download_obo, GOHierarchy

    obo_path = Path(obo_file)
    obo_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if not obo_path.exists():
        print(f"  OBO file not found: {obo_file}")
        print(f"  Downloading from Gene Ontology...")
        download_obo(str(obo_path))
    else:
        size_mb = obo_path.stat().st_size / (1024 * 1024)
        print(f"  OBO file found: {obo_file} ({size_mb:.1f} MB)")

    print(f"\n  Parsing OBO file...")
    hierarchy = GOHierarchy(str(obo_path))

    n_terms = len(hierarchy.child_to_parents)
    n_edges = sum(len(v) for v in hierarchy.child_to_parents.values())

    elapsed = time.time() - start_time
    print(f"\n  OBO setup complete in {format_time(elapsed)}")

    return {
        "obo_file": str(obo_path),
        "total_terms": n_terms,
        "total_edges": n_edges,
        "elapsed_time": elapsed,
        "hierarchy": hierarchy,
    }


# =========================================================================
# STEP 1: Verify Ancestor Closure
# =========================================================================

def run_ancestor_verification(
    hierarchy,
    annotation_file: str,
    graphs_dir: str,
    esm2_dir: str,
    esm2_dim: int,
) -> dict:
    """Verify that annotations have proper ancestor closure."""
    print("\n" + "=" * 60)
    print("STEP 1: Verify Ancestor Closure in Annotations")
    print("=" * 60)

    from dataset_v11 import get_dataloaders_v11

    start_time = time.time()

    print(f"  Loading dataset to get GO term vocabularies...")
    train_loader, _, _, train_dataset, _ = get_dataloaders_v11(
        graphs_dir=graphs_dir,
        annotation_file=annotation_file,
        esm2_dir=esm2_dir,
        esm2_dim=esm2_dim,
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )

    mf_terms = train_dataset.parser.mf_terms
    bp_terms = train_dataset.parser.bp_terms
    cc_terms = train_dataset.parser.cc_terms

    print(f"  Vocabulary sizes: MF={len(mf_terms)}, BP={len(bp_terms)}, CC={len(cc_terms)}")

    results = hierarchy.verify_ancestor_closure(
        train_dataset.parser.annotations,
        mf_terms, bp_terms, cc_terms,
    )

    del train_loader, train_dataset
    gc.collect()

    elapsed = time.time() - start_time
    print(f"\n  Ancestor verification complete in {format_time(elapsed)}")

    return {
        "mf_terms": len(mf_terms),
        "bp_terms": len(bp_terms),
        "cc_terms": len(cc_terms),
        "closure_results": results,
        "elapsed_time": elapsed,
    }


# =========================================================================
# STEP 2: Train v13 Model
# =========================================================================

def run_training(
    graphs_dir: str,
    output_dir: str,
    annotation_file: str,
    obo_file: str,
    hierarchy,
    esm2_dir: str = None,
    esm2_dim: int = 1280,
    epochs: int = 100,
    vram_gb: float = 8.0,
    batch_size: int = None,
    accum_steps: int = None,
    lr: float = 5e-4,
    hier_weight: float = 0.05,
    hier_margin: float = 0.0,
    resume: str = None,
    use_label_embed: bool = True,
    use_cooc_reg: bool = True,
    use_hier_loss: bool = True,
    use_propagation: bool = True,
    use_chain_pool: bool = True,
    use_amp: bool = True,
) -> dict:
    """Train v13 model with hierarchical chain-aware pooling + GO-DAG."""
    print("\n" + "=" * 60)
    print("STEP 2: Train v13 Model (Hierarchical Chain-Aware Pooling)")
    print("=" * 60)

    from src.model.protein_gnn import create_model, count_parameters, count_layer_parameters
    from src.training.trainer import Trainer
    from src.data.go_hierarchy import HierarchicalConsistencyLoss, AncestorPropagator
    from src.config import TrainingConfig, MemoryConfig

    # VRAM-based defaults
    if batch_size is None:
        batch_size = 16 if vram_gb >= 8.0 else (4 if vram_gb >= 6.0 else 2)
    if accum_steps is None:
        accum_steps = 1 if vram_gb >= 8.0 else (4 if vram_gb >= 6.0 else 8)

    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Graphs dir:        {graphs_dir}")
    print(f"  ESM2 dir:          {esm2_dir or 'NOT PROVIDED'}")
    print(f"  ESM2 dim:          {esm2_dim}")
    print(f"  OBO file:          {obo_file}")
    print(f"  Checkpoint dir:    {checkpoint_dir}")
    print(f"  VRAM:              {vram_gb} GB")
    print(f"  Batch size:        {batch_size}")
    print(f"  Accum steps:       {accum_steps}")
    print(f"  Effective batch:   {batch_size * accum_steps}")
    print(f"  Learning rate:     {lr}")
    print(f"  Chain-aware pool:  {use_chain_pool}")
    print(f"  Hier loss:         {use_hier_loss} (weight={hier_weight}, margin={hier_margin})")
    print(f"  Propagation:       {use_propagation}")
    print(f"  Label embed:       {use_label_embed}")
    print(f"  Cooc regularizer:  {use_cooc_reg}")
    print(f"  AMP:               {use_amp}")

    # Dataloaders (v11 = v10 graphs + ESM2)
    print("\n  Creating v13 data loaders (v10 graphs + ESM2 embeddings)...")
    eval_batch_size = max(batch_size // 2, 1)
    train_loader, val_loader, test_loader, train_dataset, esm2_loader = get_dataloaders_v11(
        graphs_dir=graphs_dir,
        annotation_file=annotation_file,
        esm2_dir=esm2_dir,
        esm2_dim=esm2_dim,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )

    num_mf = train_dataset.num_mf
    num_bp = train_dataset.num_bp
    num_cc = train_dataset.num_cc
    print(f"    Train: {len(train_loader.dataset)}")
    print(f"    Val:   {len(val_loader.dataset)}")
    print(f"    Test:  {len(test_loader.dataset)}")
    print(f"    Classes: MF={num_mf}, BP={num_bp}, CC={num_cc}")

    if esm2_loader:
        print(f"    ESM2: available={esm2_loader.available}, dim={esm2_loader.esm_dim}")
    else:
        print(f"    ESM2: NOT AVAILABLE (v10 fallback)")

    actual_esm_dim = esm2_loader.esm_dim if esm2_loader else esm2_dim

    # ── Build hierarchy components ──
    print("\n  Building GO-DAG components...")

    hier_loss_mf = hier_loss_bp = hier_loss_cc = None
    propagator_mf = propagator_bp = propagator_cc = None

    mf_terms = train_dataset.parser.mf_terms
    bp_terms = train_dataset.parser.bp_terms
    cc_terms = train_dataset.parser.cc_terms

    if use_hier_loss:
        mf_c, mf_p = hierarchy.get_edges(mf_terms)
        bp_c, bp_p = hierarchy.get_edges(bp_terms)
        cc_c, cc_p = hierarchy.get_edges(cc_terms)

        hier_loss_mf = HierarchicalConsistencyLoss(mf_c, mf_p, weight=hier_weight, margin=hier_margin)
        hier_loss_bp = HierarchicalConsistencyLoss(bp_c, bp_p, weight=hier_weight, margin=hier_margin)
        hier_loss_cc = HierarchicalConsistencyLoss(cc_c, cc_p, weight=hier_weight, margin=hier_margin)
        print(f"    Hierarchy losses built")

    if use_propagation:
        mf_anc = hierarchy.get_ancestor_matrix(mf_terms)
        bp_anc = hierarchy.get_ancestor_matrix(bp_terms)
        cc_anc = hierarchy.get_ancestor_matrix(cc_terms)

        propagator_mf = AncestorPropagator(mf_anc)
        propagator_bp = AncestorPropagator(bp_anc)
        propagator_cc = AncestorPropagator(cc_anc)
        print(f"    Ancestor propagators built")

    # ── Co-occurrence annotations ──
    annotations_mf = annotations_bp = annotations_cc = None
    go_list_mf = go_list_bp = go_list_cc = None

    if use_cooc_reg:
        print("\n  Preparing co-occurrence regularization...")
        cooc_annotations = train_dataset.parser.annotations
        annotations_mf = {gid: annot.get('mf', []) for gid, annot in cooc_annotations.items()}
        annotations_bp = {gid: annot.get('bp', []) for gid, annot in cooc_annotations.items()}
        annotations_cc = {gid: annot.get('cc', []) for gid, annot in cooc_annotations.items()}
        go_list_mf = mf_terms
        go_list_bp = bp_terms
        go_list_cc = cc_terms

    # ── Create v13 model ──
    print(f"\n  Creating v13 model (chain-aware pooling: {use_chain_pool})...")
    model = create_v13_model(
        n_mf=num_mf,
        n_bp=num_bp,
        n_cc=num_cc,
        vram_gb=vram_gb,
        esm_dim=actual_esm_dim,
        use_label_embed=use_label_embed,
        use_chain_pool=use_chain_pool,
    )
    print(f"    Parameters: {count_parameters(model):,}")

    print(f"\n    Parameter breakdown:")
    for name, count in count_layer_parameters(model).items():
        pct = 100.0 * count / count_parameters(model)
        print(f"      {name}: {count:,} ({pct:.1f}%)")

    # ── Training config ──
    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        epochs=epochs,
        early_stopping_patience=20,
        scheduler_patience=5,
        gradient_accumulation_steps=accum_steps,
    )

    memory_config = MemoryConfig(
        max_batch_memory_mb=3000.0 if vram_gb >= 8.0 else 2000.0,
        clear_cache_every=50,
    )

    # ── Create v13 trainer ──
    trainer = TrainerV13(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        memory_config=memory_config,
        checkpoint_dir=str(checkpoint_dir),
        annotations_mf=annotations_mf,
        annotations_bp=annotations_bp,
        annotations_cc=annotations_cc,
        go_list_mf=go_list_mf,
        go_list_bp=go_list_bp,
        go_list_cc=go_list_cc,
        use_cooc_reg=use_cooc_reg,
        use_amp=use_amp,
        # v12 hierarchy
        hier_loss_mf=hier_loss_mf,
        hier_loss_bp=hier_loss_bp,
        hier_loss_cc=hier_loss_cc,
        propagator_mf=propagator_mf,
        propagator_bp=propagator_bp,
        propagator_cc=propagator_cc,
    )

    if resume:
        print(f"\n  Resuming from: {resume}")
        trainer.load_checkpoint(resume, num_epochs=epochs)

    # Train
    start_time = time.time()
    history = trainer.train(num_epochs=epochs)
    elapsed = time.time() - start_time

    # Test with best model
    print("\n  Loading best model for testing...")
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        trainer.load_checkpoint(str(best_path))

    test_results = trainer.test(test_loader)

    # Save results
    results_path = checkpoint_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    print(f"\n  Training complete in {format_time(elapsed)}")
    print(f"  Test results saved to: {results_path}")

    if esm2_loader:
        esm_stats = esm2_loader.get_stats()
        print(f"  ESM2 loader stats: {esm_stats}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "elapsed_time": elapsed,
        "best_val_fmax": trainer.best_val_fmax,
        "test_results": test_results,
    }


# =========================================================================
# STEP 3: Compare v13 with baselines
# =========================================================================

def run_comparison(output_dir: str) -> dict:
    """Compare v13 with v12 and prior versions."""
    print("\n" + "=" * 60)
    print("STEP 3: Compare v13 with Baselines")
    print("=" * 60)

    import numpy as np

    v13_path = Path(output_dir) / "checkpoints" / "test_results.json"
    baselines = {
        "v7":  Path("output_v7")  / "checkpoints" / "test_results.json",
        "v9":  Path("output_v9")  / "checkpoints" / "test_results.json",
        "v10": Path("output_v10") / "checkpoints" / "test_results.json",
        "v11": Path("output_v11") / "checkpoints" / "test_results.json",
        "v12": Path("output_v12") / "checkpoints" / "test_results.json",
    }

    if not v13_path.exists():
        print("  v13 results not found, skipping comparison")
        return {}

    with open(v13_path) as f:
        v13 = json.load(f)

    comparison = {"v13": v13}

    for name, path in baselines.items():
        if path.exists():
            with open(path) as f:
                comparison[name] = json.load(f)
            print(f"  Loaded {name} results from {path}")
        else:
            print(f"  {name} results not found at {path}")

    # Print comparison table
    versions_available = [v for v in ["v7", "v9", "v10", "v11", "v12"] if v in comparison]

    print(f"\n  {'Version':<10}", end="")
    for ont in ['MF', 'BP', 'CC', 'Combined']:
        print(f"{ont:>12}", end="")
    print()
    print("  " + "-" * 58)

    for ver in versions_available:
        res = comparison[ver]
        vals = []
        print(f"  {ver:<10}", end="")
        for ont in ['mf', 'bp', 'cc']:
            fmax = res.get(ont, {}).get('fmax_perclass',
                   res.get(ont, {}).get('fmax', 0))
            fmax_val = float(fmax) if fmax else 0.0
            vals.append(fmax_val)
            print(f"{fmax_val:>12.4f}", end="")
        combined = sum(vals) / max(len(vals), 1)
        print(f"{combined:>12.4f}")

    # v13 raw
    print(f"  {'v13 raw':<10}", end="")
    v13_vals_raw = []
    for ont in ['mf', 'bp', 'cc']:
        fmax = float(v13.get(ont, {}).get('fmax_perclass', 0))
        v13_vals_raw.append(fmax)
        print(f"{fmax:>12.4f}", end="")
    combined_raw = sum(v13_vals_raw) / 3.0
    print(f"{combined_raw:>12.4f}")

    # v13 propagated
    print(f"  {'v13 prop':<10}", end="")
    v13_vals_prop = []
    for ont in ['mf', 'bp', 'cc']:
        fmax = float(v13.get(ont, {}).get('fmax_propagated', v13.get(ont, {}).get('fmax_perclass', 0)))
        v13_vals_prop.append(fmax)
        print(f"{fmax:>12.4f}", end="")
    combined_prop = sum(v13_vals_prop) / 3.0
    print(f"{combined_prop:>12.4f}")

    # Deltas vs v12
    if "v12" in comparison:
        print(f"\n  --- v12 → v13 Delta (Hierarchical Chain-Aware Pooling) ---")
        v12_vals = []
        for ont in ['mf', 'bp', 'cc']:
            v12_fmax = float(comparison['v12'].get(ont, {}).get('fmax_propagated',
                            comparison['v12'].get(ont, {}).get('fmax_perclass',
                            comparison['v12'].get(ont, {}).get('fmax', 0))) or 0)
            v13_fmax = v13_vals_prop[['mf', 'bp', 'cc'].index(ont)]
            delta = v13_fmax - v12_fmax
            direction = "+" if delta > 0 else ""
            v12_vals.append(v12_fmax)
            print(f"    {ont.upper()}: {v12_fmax:.4f} → {v13_fmax:.4f} ({direction}{delta:.4f})")

        v12_combined = sum(v12_vals) / 3.0
        delta = combined_prop - v12_combined
        direction = "+" if delta > 0 else ""
        print(f"    Combined: {v12_combined:.4f} → {combined_prop:.4f} ({direction}{delta:.4f})")

    # Propagation gain
    prop_gain = combined_prop - combined_raw
    print(f"\n  --- Propagation Gain ---")
    print(f"    Raw:        {combined_raw:.4f}")
    print(f"    Propagated: {combined_prop:.4f}")
    print(f"    Gain:       {prop_gain:+.4f}")

    # Model info
    use_chain_pool = v13.get('use_chain_pool', True)
    esm_coverage = v13.get('esm2_test_coverage', 0)
    esm_dim = v13.get('esm2_dim', 0)
    print(f"\n  v13 Model info:")
    print(f"    Chain-aware pooling: {use_chain_pool}")
    print(f"    ESM2 dim:           {esm_dim}")
    print(f"    ESM2 test coverage: {100*esm_coverage:.1f}%")

    # Save comparison
    report_path = Path(output_dir) / "v13_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n  Comparison saved to: {report_path}")

    return comparison


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="v13 Pipeline - Hierarchical Chain-Aware Pooling"
    )
    parser.add_argument("--output-dir", type=str, default="output_v13",
                        help="Output directory for v13 artifacts")
    parser.add_argument("--graphs-dir", type=str, default="output_v10/graphs_v10",
                        help="Pre-built v10 graphs (reused)")
    parser.add_argument("--esm2-dir", type=str, default=None,
                        help="ESM2 embeddings dir (default: output_v11/esm2_embeddings)")
    parser.add_argument("--esm2-dim", type=int, default=1280,
                        help="ESM2 embedding dimension (default: 1280)")
    parser.add_argument("--obo-file", type=str, default="annotations/go-basic.obo",
                        help="Path to GO OBO file (downloaded if absent)")
    parser.add_argument("--annotation-file", type=str,
                        default="annotations/nrPDB-GO_2019.06.18_annot.tsv")

    # Training options
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--vram-gb", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hier-weight", type=float, default=0.05,
                        help="Weight for hierarchical consistency loss")
    parser.add_argument("--hier-margin", type=float, default=0.0,
                        help="Margin for hierarchy violation")

    # v13 specific
    parser.add_argument("--no-chain-pool", action="store_true",
                        help="Disable chain-aware pooling (ablation: same as v12)")

    # Flags
    parser.add_argument("--test-only", action="store_true",
                        help="Only run test + comparison (requires trained model)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only run comparison")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--no-label-embed", action="store_true")
    parser.add_argument("--no-cooc-reg", action="store_true")
    parser.add_argument("--no-hier-loss", action="store_true",
                        help="Disable hierarchy loss (propagation-only ablation)")
    parser.add_argument("--no-propagation", action="store_true",
                        help="Disable ancestor propagation (hier-loss-only ablation)")
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    # Default ESM2 dir
    esm2_dir = args.esm2_dir or str(Path("output_v11") / "esm2_embeddings")
    use_chain_pool = not args.no_chain_pool

    print("=" * 60)
    print("  v13 Pipeline - Hierarchical Chain-Aware Pooling")
    print("  Building on v12 with chain→protein hierarchy")
    print("  Key additions:")
    print(f"    - Chain attention pool (residues → chains)")
    print(f"    - Protein chain pool   (chains → protein)")
    print(f"    - CC cross-attention   (protein × chains → CC context)")
    print(f"    - Chain-aware pooling:  {use_chain_pool}")
    print(f"    - GO hierarchy loss (weight={args.hier_weight})")
    print(f"    - Ancestor propagation at inference")
    print("  Reuses:")
    print(f"    - v10 graphs:       {args.graphs_dir}")
    print(f"    - v11 ESM2:         {esm2_dir}")
    print(f"  OBO file:             {args.obo_file}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pipeline_log = {
        "start_time": datetime.now().isoformat(),
        "version": "v13",
        "use_chain_pool": use_chain_pool,
        "hier_weight": args.hier_weight,
        "hier_margin": args.hier_margin,
        "use_hier_loss": not args.no_hier_loss,
        "use_propagation": not args.no_propagation,
    }

    # ── Compare only ──
    if args.compare_only:
        comparison = run_comparison(args.output_dir)
        pipeline_log["comparison"] = comparison
        return

    # ── Step 0: OBO setup ──
    obo_result = run_obo_setup(args.obo_file)
    hierarchy = obo_result.pop("hierarchy")
    pipeline_log["obo_setup"] = obo_result

    # ── Preflight checks ──
    if not Path(args.graphs_dir).exists():
        print(f"\n[ERROR] Graphs directory not found: {args.graphs_dir}")
        print("  Ensure v10 graphs are available (run run_pipeline_v10.py first)")
        sys.exit(1)

    if not Path(esm2_dir).exists():
        print(f"\n[WARNING] ESM2 directory not found: {esm2_dir}")
        print("  Model will run without ESM2 features (v10 fallback mode)")

    # ── Step 1: Verify ancestor closure ──
    if not args.test_only:
        verify_result = run_ancestor_verification(
            hierarchy=hierarchy,
            annotation_file=args.annotation_file,
            graphs_dir=args.graphs_dir,
            esm2_dir=esm2_dir,
            esm2_dim=args.esm2_dim,
        )
        pipeline_log["ancestor_verification"] = verify_result

    # ── Step 2: Training ──
    if not args.test_only:
        resume_ckpt = None
        if args.resume:
            latest = Path(args.output_dir) / "checkpoints" / "latest.pt"
            best = Path(args.output_dir) / "checkpoints" / "best.pt"
            if latest.exists():
                resume_ckpt = str(latest)
            elif best.exists():
                resume_ckpt = str(best)

        result = run_training(
            graphs_dir=args.graphs_dir,
            output_dir=args.output_dir,
            annotation_file=args.annotation_file,
            obo_file=args.obo_file,
            hierarchy=hierarchy,
            esm2_dir=esm2_dir,
            esm2_dim=args.esm2_dim,
            epochs=args.epochs,
            vram_gb=args.vram_gb,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            lr=args.lr,
            hier_weight=args.hier_weight,
            hier_margin=args.hier_margin,
            resume=resume_ckpt,
            use_label_embed=not args.no_label_embed,
            use_cooc_reg=not args.no_cooc_reg,
            use_hier_loss=not args.no_hier_loss,
            use_propagation=not args.no_propagation,
            use_chain_pool=use_chain_pool,
            use_amp=not args.no_amp,
        )
        pipeline_log["training"] = result
    elif args.test_only:
        # Test-only mode: load best model and test
        print("\n  Test-only mode: loading best model...")
        from train_v13 import TrainerV13
        from dataset_v11 import get_dataloaders_v11
        from model_v13 import create_v13_model
        from go_hierarchy import HierarchicalConsistencyLoss, AncestorPropagator
        from config import TrainingConfig, MemoryConfig

        checkpoint_path = Path(args.output_dir) / "checkpoints" / "best.pt"
        if not checkpoint_path.exists():
            print(f"  [ERROR] Best checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        model_config = checkpoint['model_config']

        batch_size = args.batch_size or 16
        eval_batch_size = max(batch_size // 2, 1)

        train_loader, val_loader, test_loader, train_dataset, esm2_loader = get_dataloaders_v11(
            graphs_dir=args.graphs_dir,
            annotation_file=args.annotation_file,
            esm2_dir=esm2_dir,
            esm2_dim=args.esm2_dim,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42,
        )

        actual_esm_dim = esm2_loader.esm_dim if esm2_loader else args.esm2_dim
        ckpt_chain_pool = model_config.get('use_chain_pool', use_chain_pool)

        model = create_v13_model(
            n_mf=train_dataset.num_mf,
            n_bp=train_dataset.num_bp,
            n_cc=train_dataset.num_cc,
            vram_gb=args.vram_gb,
            esm_dim=actual_esm_dim,
            use_chain_pool=ckpt_chain_pool,
        )

        # Build propagators for test
        propagator_mf = propagator_bp = propagator_cc = None
        if not args.no_propagation:
            mf_anc = hierarchy.get_ancestor_matrix(train_dataset.parser.mf_terms)
            bp_anc = hierarchy.get_ancestor_matrix(train_dataset.parser.bp_terms)
            cc_anc = hierarchy.get_ancestor_matrix(train_dataset.parser.cc_terms)
            propagator_mf = AncestorPropagator(mf_anc)
            propagator_bp = AncestorPropagator(bp_anc)
            propagator_cc = AncestorPropagator(cc_anc)

        training_config = TrainingConfig(batch_size=batch_size)
        memory_config = MemoryConfig()

        trainer = TrainerV13(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            memory_config=memory_config,
            checkpoint_dir=str(Path(args.output_dir) / "checkpoints"),
            use_amp=not args.no_amp,
            propagator_mf=propagator_mf,
            propagator_bp=propagator_bp,
            propagator_cc=propagator_cc,
        )

        trainer.load_checkpoint(str(checkpoint_path))

        test_results = trainer.test(test_loader)

        results_path = Path(args.output_dir) / "checkpoints" / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2, default=str)

        print(f"\n  Test results saved to: {results_path}")
        pipeline_log["test_results"] = test_results

    # ── Step 3: Compare ──
    comparison = run_comparison(args.output_dir)
    pipeline_log["comparison"] = comparison

    # Save pipeline log
    log_path = Path(args.output_dir) / "pipeline_log.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(pipeline_log, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"  v13 Pipeline Complete")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log saved: {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
