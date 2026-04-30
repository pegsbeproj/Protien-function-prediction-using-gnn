"""
extract_esm2_embeddings.py — ESM2 Per-Residue Embedding Extractor

Extracts per-residue ESM2 embeddings from protein structures and stores them
for use during GNN training. This is STEP 2 of the data pipeline.

This script works with multi-chain PDB-level graphs (built by build_graphs.py).
It does NOT regenerate graphs; it reads the existing graph batch files to
recover sequences, then runs ESM2 inference.

ESM2 Model Options:
  esm2_t6_8M_UR50D     →  320-dim,   8M params  (~30 MB, fastest)
  esm2_t12_35M_UR50D   →  480-dim,  35M params  (~140 MB)
  esm2_t30_150M_UR50D  →  640-dim, 150M params  (~600 MB, recommended)
  esm2_t33_650M_UR50D  → 1280-dim, 650M params  (~2.5 GB, best quality)

Default: esm2_t33_650M_UR50D (1280-dim) — matches training configuration.

Storage Format:
  data/esm2_embeddings/
    ├── {PDB_ID}.pt          # {'esm_emb': Tensor[N, 1280], 'num_nodes': N, ...}
    ├── ...
    └── esm2_index.json      # Index of all extracted embeddings

Two Operating Modes:
  1. --from-pdb:    Extract directly from raw PDB/CIF files
  2. --from-graphs: Extract from existing graph batch files (no PDB files needed)

Usage (STEP 2):
    # Mode 1: from PDB files
    python scripts/extract_esm2_embeddings.py --from-pdb \\
        --pdb-dir data/pdbs --output-dir data/esm2_embeddings

    # Mode 2: from graphs (recommended if graphs already built)
    python scripts/extract_esm2_embeddings.py --from-graphs \\
        --graphs-dir data/graphs --output-dir data/esm2_embeddings

    # Resume interrupted extraction
    python scripts/extract_esm2_embeddings.py --from-graphs \\
        --graphs-dir data/graphs --output-dir data/esm2_embeddings --resume
"""

import gc
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# One-hot index → single-letter AA (same ordering as v10)
IDX_TO_AA_1LETTER = list("ARNDCEQGHILKMFPSTWYV")

ESM2_MODELS = {
    'esm2_t6_8M_UR50D':   {'layers': 6,  'dim': 320,  'params': '8M'},
    'esm2_t12_35M_UR50D':  {'layers': 12, 'dim': 480,  'params': '35M'},
    'esm2_t30_150M_UR50D': {'layers': 30, 'dim': 640,  'params': '150M'},
    'esm2_t33_650M_UR50D': {'layers': 33, 'dim': 1280, 'params': '650M'},
}


# ═══════════════════════════════════════════════════════════════════
#  ESM2 Extractor (PDB-Level, Multi-Chain)
# ═══════════════════════════════════════════════════════════════════

class ESM2ExtractorV11:
    """
    ESM2 embedding extractor for v11 PDB-level graphs.

    Processes each chain separately, then concatenates in the same order
    as the v10 merged graph (chain_idx order).

    Embeddings are stored as float16 to reduce disk and memory usage.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cuda",
        max_length: int = 1022,
        half_precision: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.half_precision = half_precision and self.device.type == 'cuda'
        self.model_name = model_name

        model_info = ESM2_MODELS.get(model_name)
        if model_info is None:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(ESM2_MODELS.keys())}")

        self.repr_layer = model_info['layers']
        self.embedding_dim = model_info['dim']

        print(f"Loading ESM2 model: {model_name}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Repr layer:    {self.repr_layer}")
        print(f"  Device:        {self.device}")

        import esm
        loader = getattr(esm.pretrained, model_name)
        self.model, self.alphabet = loader()
        self.batch_converter = self.alphabet.get_batch_converter()

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.half_precision:
            self.model = self.model.half()

        for param in self.model.parameters():
            param.requires_grad = False

        if self.device.type == 'cuda':
            alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            print(f"  GPU memory after loading: {alloc:.2f} GB")

    @torch.no_grad()
    def extract_single(self, sequence: str, label: str = "") -> torch.Tensor:
        """
        Extract per-residue embeddings for a single sequence.

        Args:
            sequence: Single-letter AA sequence
            label: Identifier for logging

        Returns:
            Tensor [seq_len, embedding_dim] in float32
        """
        orig_len = len(sequence)
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]

        data = [(label, sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        if self.half_precision:
            with torch.amp.autocast('cuda'):
                results = self.model(
                    batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
                )
        else:
            results = self.model(
                batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
            )

        # [1, seq_len+2, dim] → [seq_len, dim] (strip BOS/EOS)
        emb = results["representations"][self.repr_layer][0, 1:-1, :].float().cpu()

        # If truncated, pad with zeros for remaining residues
        if orig_len > self.max_length:
            pad = torch.zeros(orig_len - self.max_length, self.embedding_dim)
            emb = torch.cat([emb, pad], dim=0)

        return emb

    @torch.no_grad()
    def extract_multichain(
        self,
        chain_sequences: List[Tuple[str, str]],
        pdb_id: str = ""
    ) -> torch.Tensor:
        """
        Extract embeddings for a multi-chain PDB structure.

        Processes each chain independently, then concatenates in order.
        This matches v10 merged graph node ordering.

        Args:
            chain_sequences: List of (chain_id, sequence) tuples IN ORDER
            pdb_id: PDB ID for logging

        Returns:
            Tensor [total_residues, embedding_dim] in float32
        """
        all_embs = []

        for chain_id, seq in chain_sequences:
            if len(seq) == 0:
                continue
            emb = self.extract_single(seq, label=f"{pdb_id}-{chain_id}")
            all_embs.append(emb)

        if not all_embs:
            return torch.zeros(0, self.embedding_dim)

        return torch.cat(all_embs, dim=0)

    def clear_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


# ═══════════════════════════════════════════════════════════════════
#  Mode 1: Extract from PDB files
# ═══════════════════════════════════════════════════════════════════

def extract_from_pdb_files(
    pdb_dir: str,
    output_dir: str,
    extractor: ESM2ExtractorV11,
    min_residues: int = 10,
    max_residues: int = 5000,
    resume: bool = True,
):
    """
    Extract ESM2 embeddings from raw PDB/CIF files.

    Produces one embedding file per PDB (multi-chain merged, matching v10).
    """
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa

    pdb_path = Path(pdb_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(
        list(pdb_path.glob("*.pdb")) + list(pdb_path.glob("*.cif"))
    )
    print(f"Found {len(pdb_files)} PDB/CIF files")

    # Resume support
    index_file = out_path / "esm2_index.json"
    if resume and index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        processed = set(index.get('processed_pdbs', []))
        print(f"Resuming: {len(processed)} PDBs already processed")
    else:
        index = {
            'processed_pdbs': [],
            'embedding_dim': extractor.embedding_dim,
            'model': extractor.model_name,
            'pdb_entries': {},
            'stats': {'total_pdbs': 0, 'total_residues': 0, 'skipped': 0},
        }
        processed = set()

    remaining = [p for p in pdb_files if p.stem.upper() not in processed]
    print(f"Remaining to process: {len(remaining)}")

    save_interval = 50

    for i, pdb_file in enumerate(tqdm(remaining, desc="Extracting ESM2")):
        pdb_id = pdb_file.stem.upper()

        try:
            # Parse structure
            if pdb_file.suffix.lower() == '.cif':
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_id, str(pdb_file))

            chain_sequences = []
            total_residues = 0

            for model in structure:
                for chain in model:
                    chain_id = chain.id or "A"
                    seq_parts = []

                    for res in chain:
                        if not is_aa(res, standard=True):
                            continue
                        if 'CA' not in res:
                            continue
                        resname = res.get_resname()
                        aa = THREE_TO_ONE.get(resname, 'X')
                        if aa != 'X':
                            seq_parts.append(aa)

                    seq = ''.join(seq_parts)
                    if len(seq) >= min_residues:
                        chain_sequences.append((chain_id, seq))
                        total_residues += len(seq)
                break  # first model only

            if not chain_sequences or total_residues < min_residues:
                index['stats']['skipped'] += 1
                continue

            if total_residues > max_residues:
                index['stats']['skipped'] += 1
                continue

            # Extract ESM2 embeddings
            esm_emb = extractor.extract_multichain(chain_sequences, pdb_id)

            # Save as float16 to save disk space
            save_data = {
                'esm_emb': esm_emb.half(),
                'num_nodes': esm_emb.shape[0],
                'esm_dim': extractor.embedding_dim,
                'pdb_id': pdb_id,
                'chains': [c[0] for c in chain_sequences],
                'chain_lengths': [len(c[1]) for c in chain_sequences],
            }
            torch.save(save_data, out_path / f"{pdb_id}.pt")

            # Update index
            index['processed_pdbs'].append(pdb_id)
            processed.add(pdb_id)
            index['pdb_entries'][pdb_id] = {
                'num_nodes': esm_emb.shape[0],
                'chains': [c[0] for c in chain_sequences],
            }
            index['stats']['total_pdbs'] += 1
            index['stats']['total_residues'] += total_residues

        except Exception as e:
            index['stats']['skipped'] += 1
            if index['stats']['skipped'] <= 10:
                print(f"\n  [ERROR] {pdb_id}: {e}")
            extractor.clear_cache()
            continue

        # Periodic save
        if (i + 1) % save_interval == 0:
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
            extractor.clear_cache()

    # Final save
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n{'=' * 60}")
    print("ESM2 Embedding Extraction Complete (from PDB)")
    print(f"{'=' * 60}")
    print(f"  Model:          {extractor.model_name}")
    print(f"  Embedding dim:  {extractor.embedding_dim}")
    print(f"  PDBs processed: {index['stats']['total_pdbs']}")
    print(f"  Total residues: {index['stats']['total_residues']:,}")
    print(f"  Skipped:        {index['stats']['skipped']}")
    print(f"  Output dir:     {out_path}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════
#  Mode 2: Extract from existing v10 graph files
# ═══════════════════════════════════════════════════════════════════

def recover_sequence_from_onehot(x: torch.Tensor, chain_idx: torch.Tensor) -> List[Tuple[str, str]]:
    """
    Recover per-chain amino acid sequences from v10 graph one-hot features.

    Args:
        x: [N, 40] node features (first 20 dims = one-hot AA)
        chain_idx: [N] chain assignment

    Returns:
        List of (chain_id_str, sequence) in chain order
    """
    onehot = x[:, :20]
    aa_indices = onehot.argmax(dim=1)

    chains = {}
    for node_i in range(len(aa_indices)):
        c = int(chain_idx[node_i].item())
        if c not in chains:
            chains[c] = []
        chains[c].append(IDX_TO_AA_1LETTER[aa_indices[node_i].item()])

    result = []
    for c in sorted(chains.keys()):
        seq = ''.join(chains[c])
        result.append((str(c), seq))

    return result


def extract_from_graph_files(
    graphs_dir: str,
    output_dir: str,
    extractor: ESM2ExtractorV11,
    resume: bool = True,
):
    """
    Extract ESM2 embeddings by reading sequences from existing v10 graph batch files.

    No PDB files needed — sequences are recovered from one-hot node features.
    """
    graphs_path = Path(graphs_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(graphs_path.glob("graphs_batch_*.pt"))
    print(f"Found {len(batch_files)} graph batch files")

    # Resume support
    index_file = out_path / "esm2_index.json"
    if resume and index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        processed = set(index.get('processed_pdbs', []))
        print(f"Resuming: {len(processed)} PDBs already processed")
    else:
        index = {
            'processed_pdbs': [],
            'embedding_dim': extractor.embedding_dim,
            'model': extractor.model_name,
            'pdb_entries': {},
            'stats': {'total_pdbs': 0, 'total_residues': 0, 'skipped': 0},
        }
        processed = set()

    save_interval = 20  # Save index every N batch files

    for bf_idx, batch_file in enumerate(tqdm(batch_files, desc="Processing graph batches")):
        try:
            graphs = torch.load(batch_file, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"\n  [ERROR] Loading {batch_file.name}: {e}")
            continue

        for graph in graphs:
            pdb_id = graph.graph_id if hasattr(graph, 'graph_id') else getattr(graph, 'pdb_id', None)
            if not pdb_id or pdb_id in processed:
                continue

            try:
                chain_idx = getattr(graph, 'chain_idx', None)
                if chain_idx is None:
                    chain_idx = torch.zeros(graph.x.shape[0], dtype=torch.long)

                chain_sequences = recover_sequence_from_onehot(graph.x, chain_idx)

                if not chain_sequences:
                    index['stats']['skipped'] += 1
                    continue

                total_res = sum(len(s) for _, s in chain_sequences)

                # Extract ESM2 embeddings
                esm_emb = extractor.extract_multichain(chain_sequences, pdb_id)

                # Verify node count matches
                if esm_emb.shape[0] != graph.x.shape[0]:
                    print(f"\n  [WARN] {pdb_id}: ESM nodes {esm_emb.shape[0]} != graph nodes {graph.x.shape[0]}")
                    # Pad or truncate to match
                    if esm_emb.shape[0] < graph.x.shape[0]:
                        pad = torch.zeros(graph.x.shape[0] - esm_emb.shape[0], extractor.embedding_dim)
                        esm_emb = torch.cat([esm_emb, pad], dim=0)
                    else:
                        esm_emb = esm_emb[:graph.x.shape[0]]

                # Save
                save_data = {
                    'esm_emb': esm_emb.half(),
                    'num_nodes': esm_emb.shape[0],
                    'esm_dim': extractor.embedding_dim,
                    'pdb_id': pdb_id,
                    'chains': [c[0] for c in chain_sequences],
                    'chain_lengths': [len(c[1]) for c in chain_sequences],
                }
                torch.save(save_data, out_path / f"{pdb_id}.pt")

                index['processed_pdbs'].append(pdb_id)
                processed.add(pdb_id)
                index['pdb_entries'][pdb_id] = {'num_nodes': esm_emb.shape[0]}
                index['stats']['total_pdbs'] += 1
                index['stats']['total_residues'] += total_res

            except Exception as e:
                index['stats']['skipped'] += 1
                if index['stats']['skipped'] <= 10:
                    print(f"\n  [ERROR] {pdb_id}: {e}")
                extractor.clear_cache()

        # Periodic save
        if (bf_idx + 1) % save_interval == 0:
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
            extractor.clear_cache()

    # Final save
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n{'=' * 60}")
    print("ESM2 Embedding Extraction Complete (from graphs)")
    print(f"{'=' * 60}")
    print(f"  Model:          {extractor.model_name}")
    print(f"  Embedding dim:  {extractor.embedding_dim}")
    print(f"  PDBs processed: {index['stats']['total_pdbs']}")
    print(f"  Total residues: {index['stats']['total_residues']:,}")
    print(f"  Skipped:        {index['stats']['skipped']}")
    print(f"  Output dir:     {out_path}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="v11 ESM2 Embedding Extraction for PDB-Level Graphs"
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-pdb", action="store_true",
                      help="Extract from raw PDB/CIF files")
    mode.add_argument("--from-graphs", action="store_true",
                      help="Extract from existing v10 graph batch files")

    # Shared arguments
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for ESM2 embeddings")
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D",
                        choices=list(ESM2_MODELS.keys()),
                        help="ESM2 model (default: esm2_t33_650M_UR50D)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max-length", type=int, default=1022,
                        help="Max sequence length per chain (default: 1022)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous progress")
    parser.add_argument("--no-half", action="store_true",
                        help="Disable FP16 for ESM2 inference")

    # Mode-specific arguments
    parser.add_argument("--pdb-dir", type=str, default="data/pdbs",
                        help="PDB/CIF file directory (for --from-pdb)")
    parser.add_argument("--graphs-dir", type=str, default="data/graphs",
                        help="Graph directory (for --from-graphs)")
    parser.add_argument("--min-residues", type=int, default=10)
    parser.add_argument("--max-residues", type=int, default=5000)

    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"  v11 ESM2 Embedding Extraction")
    print(f"  Model:  {args.model} ({ESM2_MODELS[args.model]['dim']}-dim)")
    print(f"  Mode:   {'from-pdb' if args.from_pdb else 'from-graphs'}")
    print(f"  Device: {args.device}")
    print(f"{'=' * 60}")

    # Create extractor
    extractor = ESM2ExtractorV11(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        half_precision=not args.no_half,
    )

    if args.from_pdb:
        extract_from_pdb_files(
            pdb_dir=args.pdb_dir,
            output_dir=args.output_dir,
            extractor=extractor,
            min_residues=args.min_residues,
            max_residues=args.max_residues,
            resume=args.resume,
        )
    else:
        extract_from_graph_files(
            graphs_dir=args.graphs_dir,
            output_dir=args.output_dir,
            extractor=extractor,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
