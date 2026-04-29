# Protein Function Prediction — Hierarchical Chain-Aware GNN (v13)

Multi-label prediction of Gene Ontology (GO) terms from protein 3D structure using a hierarchical chain-aware Graph Neural Network.

## Architecture Summary

| Stage | Component | Output shape |
|---|---|---|
| Input encoding | Chain embed + NodeEncoder + ESM2GatedFusion | [N, 192] |
| Message passing | DualConvBlock (GCN + GATv2) × 4 | [N, 192] |
| Residue pooling | GatedAttn + Mean + Max → PoolFuse | [B, 192] |
| Chain pooling *(novel)* | ChainAttentionPool → ProteinChainPool | [B, 192] |
| Merge | Concat branches | [B, 384] |
| CC cross-attention *(novel)* | CCContextAttention | [B, 192] |
| MF / BP heads | MLP(384 → 489 / 1943) | [B, n] |
| CC head | MLP(576 → 320) | [B, 320] |

Total parameters: **~2.28 M** (fits 8 GB VRAM comfortably).

## Project Structure

```
protein_gnn_v13/
├── README.md
├── requirements.txt
├── docs/
│   └── ARCHITECTURE.md        # Full deep-learning breakdown
├── scripts/
│   ├── build_graphs.py        # STEP 1 — PDB/CIF → PyG multi-chain graphs
│   ├── extract_esm2_embeddings.py  # STEP 2 — ESM2 per-residue embedding extraction
│   ├── run_pipeline.py        # STEP 3 — End-to-end training orchestrator (CLI)
│   └── esm2_finetune.py       # Optional — LoRA fine-tuning of ESM2
└── src/
    ├── config.py              # All hyperparameters and paths
    ├── model/
    │   ├── building_blocks.py # GNN primitives (GatedAttentionPool, DualConvBlock, …)
    │   └── protein_gnn.py     # ProteinGNN model + factory function
    ├── data/
    │   ├── annotations.py     # GO annotation parser
    │   ├── go_hierarchy.py    # GO DAG, hierarchy loss, ancestor propagation
    │   ├── base_dataset.py    # ProteinGODataset base class
    │   ├── multi_chain_parser.py  # MultiChainAnnotationParser (PDB-level labels)
    │   └── multi_chain_dataset.py # ESM2EmbeddingLoader + ProteinGraphDataset + DataLoaders
    └── training/
        ├── losses.py          # ClassBalancedAsymmetricLoss, CooccurrenceRegularizer, …
        ├── trainer_base.py    # TrainerV12 base class (hierarchy loss, AMP, early stopping)
        └── trainer.py         # Trainer (extends base with chain-pool tracking)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build protein graphs from PDB files (STEP 1 — ~2-4 hours for full dataset)
python scripts/build_graphs.py \
    --pdb-dir data/pdbs \
    --output-dir data/graphs \
    --threshold 10.0 \
    --max-neighbors 32 \
    --workers 4

# 3. Extract ESM2 per-residue embeddings (STEP 2 — requires GPU, ~4-8 hours)
python scripts/extract_esm2_embeddings.py --from-graphs \
    --graphs-dir data/graphs \
    --output-dir data/esm2_embeddings \
    --model esm2_t33_650M_UR50D \
    --resume

# 4. Run full training pipeline (STEP 3 — requires GPU, ~12-24 hours)
python scripts/run_pipeline.py \
    --graphs-dir data/graphs \
    --esm2-dir   data/esm2_embeddings \
    --output-dir outputs/

# Resume interrupted training
python scripts/run_pipeline.py \
    --graphs-dir data/graphs \
    --output-dir outputs/ \
    --resume

# Ablation: disable hierarchical chain pooling (reproduces v11/v12 behaviour)
python scripts/run_pipeline.py \
    --graphs-dir data/graphs \
    --output-dir outputs/ablation_no_chain_pool/ \
    --no-chain-pool
```

## Key Results (best epoch)

| Ontology | Fmax (raw) | Fmax (propagated) |
|---|---|---|
| MF | see `output_v13/checkpoints/test_results.json` | |
| BP | | |
| CC | | |

## Novel Contributions

1. **PDB-level multi-chain graphs** — full protein complex with inter-chain edges
2. **Hierarchical chain-aware pooling** — Residue → Chain → Protein (mirrors biological organisation)
3. **CC cross-attention** — chain-level context for subcellular localisation prediction
4. **ESM2 gated fusion** — learned per-residue blending of evolutionary and structural features

## Data Dependencies

| Artefact | Source |
|---|---|
| v10 graphs | `run_pipeline_v10.py` in parent project |
| ESM2 embeddings | `esm2_finetune.py` + `extract_embeddings_v5.py` in parent project |
| GO annotations | `annotations/nrPDB-GO_2019.06.18_annot.tsv` |
| GO OBO file | Auto-downloaded from geneontology.org if absent |
