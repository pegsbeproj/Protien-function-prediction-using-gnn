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

# 2. Prepare data folders
mkdir -p data/pdbs data/annotations data/graphs data/esm2_embeddings

# Place annotation file at:
#   data/annotations/nrPDB-GO_2019.06.18_annot.tsv
# (optional official split files can be placed in the same folder)

# 3. Build protein graphs from PDB/CIF files (STEP 1)
python scripts/build_graphs.py \
    --pdb-dir data/pdbs \
    --output-dir data/graphs \
    --threshold 10.0 \
    --max-neighbors 32 \
    --workers 4

# 4. Extract ESM2 per-residue embeddings (STEP 2)
python scripts/extract_esm2_embeddings.py --from-graphs \
    --graphs-dir data/graphs \
    --output-dir data/esm2_embeddings \
    --model esm2_t33_650M_UR50D \
    --resume

# 5. Run full v13 pipeline (STEP 3)
python scripts/run_pipeline.py \
    --graphs-dir data/graphs \
    --esm2-dir   data/esm2_embeddings \
    --annotation-file data/annotations/nrPDB-GO_2019.06.18_annot.tsv \
    --obo-file data/annotations/go-basic.obo \
    --output-dir output_v13

# Resume interrupted training
python scripts/run_pipeline.py \
    --graphs-dir data/graphs \
    --esm2-dir   data/esm2_embeddings \
    --annotation-file data/annotations/nrPDB-GO_2019.06.18_annot.tsv \
    --obo-file data/annotations/go-basic.obo \
    --output-dir output_v13 \
    --resume

# Use random splits (disable DeepFRI split auto-detection)
python scripts/run_pipeline.py \
    --graphs-dir data/graphs \
    --esm2-dir data/esm2_embeddings \
    --annotation-file data/annotations/nrPDB-GO_2019.06.18_annot.tsv \
    --obo-file data/annotations/go-basic.obo \
    --output-dir output_v13_random \
    --no-deepfri-splits

# Ablation: disable hierarchical chain pooling (reproduces v11/v12 behavior)
python scripts/run_pipeline.py \
    --graphs-dir data/graphs \
    --esm2-dir data/esm2_embeddings \
    --annotation-file data/annotations/nrPDB-GO_2019.06.18_annot.tsv \
    --obo-file data/annotations/go-basic.obo \
    --output-dir output_v13_ablation_no_chain_pool \
    --no-chain-pool
```

## Key Results (best epoch)

From `output_v13/checkpoints/test_results.json` and `output_v13/smin_results.json`:

| Ontology | Fmax (raw) | Fmax (propagated) | AUPR (micro) | Smin |
|---|---:|---:|---:|---:|
| MF | 0.7749 | 0.7696 | 0.7916 | 11.1329 |
| BP | 0.6743 | 0.6707 | 0.5454 | 73.9094 |
| CC | 0.5639 | 0.5653 | 0.4445 | 16.8913 |
| **Combined** | **0.6710** | **0.6685** | — | **33.9778*** |

\* Combined Smin is the mean of ontology Smin values.

## Novel Contributions

1. **PDB-level multi-chain graphs** — full protein complex with inter-chain edges
2. **Hierarchical chain-aware pooling** — Residue → Chain → Protein (mirrors biological organisation)
3. **CC cross-attention** — chain-level context for subcellular localisation prediction
4. **ESM2 gated fusion** — learned per-residue blending of evolutionary and structural features

## Data Dependencies

| Artefact | Required path |
|---|---|
| Graph batches | `data/graphs/graphs_batch_*.pt` (from `scripts/build_graphs.py`) |
| ESM2 embeddings | `data/esm2_embeddings/*.pt` (from `scripts/extract_esm2_embeddings.py`) |
| GO annotations | `data/annotations/nrPDB-GO_2019.06.18_annot.tsv` |
| Optional official splits | `data/annotations/nrPDB-GO_2019.06.18_{train,valid,test}.txt` |
| GO OBO file | `data/annotations/go-basic.obo` (auto-downloaded if absent) |

## Reproducibility Notes

1. Run commands from the repository root (`protein_gnn_v13/`) so imports like `src.*` resolve.
2. `--test-only` expects `output_v13/checkpoints/best.pt`.
3. Output artifacts for paper reporting:
   - `output_v13/checkpoints/test_results.json`
   - `output_v13/smin_results.json`
   - `output_v13/pipeline_log.json`
