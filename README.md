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
│   └── run_pipeline.py        # End-to-end orchestrator (CLI)
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
# Install dependencies
pip install -r requirements.txt

# Full training pipeline (needs v10 graphs + v11 ESM2 embeddings)
python scripts/run_pipeline.py \
    --graphs-dir  ../output_v10/graphs_v10 \
    --esm2-dir    ../output_v11/esm2_embeddings \
    --output-dir  ../output_v13

# Resume interrupted training
python scripts/run_pipeline.py \
    --graphs-dir ../output_v10/graphs_v10 \
    --output-dir ../output_v13 \
    --resume

# Ablation: disable hierarchical chain pooling (reproduces v11/v12 behaviour)
python scripts/run_pipeline.py \
    --graphs-dir ../output_v10/graphs_v10 \
    --output-dir ../output_v13_ablation \
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
