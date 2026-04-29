"""
build_graphs.py — PDB to PyG Graph Converter (Multi-Chain, Inter-Chain Geometry Aware)

Converts raw PDB/CIF structures into PyTorch Geometric (PyG) graph objects.
This is STEP 1 of the data pipeline for the protein_gnn_v13 project.

One graph is created per PDB entry (full complex), with all chains merged.
Inter-chain edges are added for any two residues within the distance threshold.
A chain_idx tensor is stored per-node so the GNN can learn chain embeddings.

Node features (40-dim):
  [0:20]  - One-hot amino acid encoding
  [20:28] - Physicochemical properties (hydropathy, volume, charge, polarity,
             aromatic, H-bond donors, H-bond acceptors, flexibility)
  [28:40] - Structural features (relative_position, sin/cos position,
             distance_to_center, local_density, centered_coords×3,
             curvature, normalized_positions×2, chain_length_normalized)
  ** Structural features are computed PER-CHAIN to preserve intra-chain semantics **

Edge features (5-dim):
  [0] normalized_distance   = dist / threshold
  [1] sequence_separation   = |i - j| / n  (1.0 for inter-chain)
  [2] backbone_indicator    = 1.0 if same chain and |i - j| == 1
  [3] local_contact         = 1.0 if same chain and |i - j| <= 4
  [4] distance_decay        = exp(-dist / 5.0)

Additional per-node attribute:
  chain_idx: int tensor [N] — chain assignment index per residue (0, 1, 2, ...)

Graph-level attributes:
  graph_id:   PDB ID (e.g., "1A2B")
  pdb_id:     PDB ID
  chain_ids:  list of chain ID strings that were merged (e.g., ["A", "B"])

Usage (STEP 1):
    python scripts/build_graphs.py --pdb-dir data/pdbs --output-dir data/graphs
    python scripts/build_graphs.py --pdb-dir data/pdbs --output-dir data/graphs --resume
    python scripts/build_graphs.py --pdb-dir data/pdbs --output-dir data/graphs --workers 8

Output:
    data/graphs/graphs_batch_0000.pt, graphs_batch_0001.pt, ...
"""

import os
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# Constants
# =============================================================================

V10_NODE_DIM = 40   # Same raw node features as v9 (chain embedding added in model)
V10_EDGE_DIM = 5    # Same edge features as v9
V10_MAX_CHAINS = 64 # Maximum number of chains per PDB (for embedding table)

AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

ONE_TO_THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

# ---------------------------------------------------------------------------
# Physicochemical properties for each amino acid (8 features)
# ---------------------------------------------------------------------------
AA_PROPERTIES: Dict[str, List[float]] = {
    'ALA': [ 1.8,   88.6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.357],
    'ARG': [-4.5,  173.4,  1.0,  1.0,  0.0,  5.0,  1.0,  0.529],
    'ASN': [-3.5,  114.1,  0.0,  1.0,  0.0,  2.0,  2.0,  0.463],
    'ASP': [-3.5,  111.1, -1.0,  1.0,  0.0,  0.0,  3.0,  0.511],
    'CYS': [ 2.5,  108.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.346],
    'GLN': [-3.5,  143.8,  0.0,  1.0,  0.0,  2.0,  2.0,  0.493],
    'GLU': [-3.5,  138.4, -1.0,  1.0,  0.0,  0.0,  3.0,  0.497],
    'GLY': [-0.4,   60.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.544],
    'HIS': [-3.2,  153.2,  0.5,  1.0,  1.0,  1.0,  1.0,  0.323],
    'ILE': [ 4.5,  166.7,  0.0,  0.0,  0.0,  0.0,  0.0,  0.462],
    'LEU': [ 3.8,  166.7,  0.0,  0.0,  0.0,  0.0,  0.0,  0.365],
    'LYS': [-3.9,  168.6,  1.0,  1.0,  0.0,  3.0,  0.0,  0.466],
    'MET': [ 1.9,  162.9,  0.0,  0.0,  0.0,  0.0,  1.0,  0.295],
    'PHE': [ 2.8,  189.9,  0.0,  0.0,  1.0,  0.0,  0.0,  0.314],
    'PRO': [-1.6,  112.7,  0.0,  0.0,  0.0,  0.0,  0.0,  0.509],
    'SER': [-0.8,   89.0,  0.0,  1.0,  0.0,  1.0,  1.0,  0.507],
    'THR': [-0.7,  116.1,  0.0,  1.0,  0.0,  1.0,  1.0,  0.444],
    'TRP': [-0.9,  227.8,  0.0,  0.0,  1.0,  1.0,  0.0,  0.305],
    'TYR': [-1.3,  193.6,  0.0,  1.0,  1.0,  1.0,  1.0,  0.420],
    'VAL': [ 4.2,  140.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.386],
}

# Precompute normalization ranges
_all_props = np.array(list(AA_PROPERTIES.values()), dtype=np.float32)
_PROP_MIN = _all_props.min(axis=0)
_PROP_MAX = _all_props.max(axis=0)
_PROP_RANGE = np.where((_PROP_MAX - _PROP_MIN) > 1e-8, _PROP_MAX - _PROP_MIN, 1.0)


# =============================================================================
# Resume utilities
# =============================================================================

def recover_processed_from_batches(output_dir: Path):
    """Scan existing batch files to recover which graphs were already processed."""
    processed_graph_ids = set()
    processed_pdb_ids = set()

    batch_files = sorted(output_dir.glob("graphs_batch_*.pt"))
    print(f"[RESUME] Found {len(batch_files)} existing batch files")

    for bf in batch_files:
        try:
            graphs = torch.load(bf, map_location="cpu", weights_only=False)
            for g in graphs:
                gid = g.graph_id
                processed_graph_ids.add(gid)
                processed_pdb_ids.add(gid)  # In v10, graph_id IS the PDB ID
        except Exception as e:
            print(f"[WARN] Could not read {bf.name}: {e}")

    print(f"[RESUME] Recovered {len(processed_graph_ids)} graphs "
          f"from {len(processed_pdb_ids)} PDBs")
    return processed_graph_ids, processed_pdb_ids, len(batch_files)


# =============================================================================
# Node Feature Construction (40-dim, per-chain computation)
# =============================================================================

def get_physicochemical_features(resname: str) -> np.ndarray:
    """Return normalized 8-dim physicochemical feature vector."""
    if len(resname) == 1:
        resname = ONE_TO_THREE.get(resname, 'ALA')
    resname = resname.upper()
    raw = np.array(AA_PROPERTIES.get(resname, AA_PROPERTIES['ALA']), dtype=np.float32)
    return (raw - _PROP_MIN) / _PROP_RANGE


def get_onehot_features(resname: str) -> np.ndarray:
    """Return 20-dim one-hot amino acid encoding."""
    onehot = np.zeros(20, dtype=np.float32)
    if len(resname) == 1:
        resname = ONE_TO_THREE.get(resname, 'ALA')
    idx = AA_TO_IDX.get(resname.upper(), 0)
    onehot[idx] = 1.0
    return onehot


def compute_structural_features(coords: np.ndarray) -> np.ndarray:
    """
    Compute 12-dim structural features for each residue from CA coordinates.
    Computed PER-CHAIN to preserve intra-chain structural semantics.

    Features (same as v9):
      [0]  relative_position
      [1]  sin_position
      [2]  cos_position
      [3]  distance_to_center (normalized)
      [4]  local_density (normalized)
      [5]  centered_x
      [6]  centered_y
      [7]  centered_z
      [8]  curvature (normalized)
      [9]  norm_x
      [10] norm_y
      [11] chain_length_normalized = n / 5000
    """
    n = len(coords)
    structural = np.zeros((n, 12), dtype=np.float32)

    if n == 0:
        return structural

    for i in range(n):
        structural[i, 0] = i / max(n - 1, 1)
        structural[i, 1] = np.sin(2.0 * np.pi * i / max(n, 1))
        structural[i, 2] = np.cos(2.0 * np.pi * i / max(n, 1))

    centroid = coords.mean(axis=0)
    dists_to_center = np.linalg.norm(coords - centroid, axis=1)
    max_dist = dists_to_center.max() + 1e-8
    structural[:, 3] = dists_to_center / max_dist

    tree = cKDTree(coords)
    for i in range(n):
        count = len(tree.query_ball_point(coords[i], r=10.0)) - 1
        structural[i, 4] = count / max(n - 1, 1)

    centered = (coords - centroid) / max_dist
    structural[:, 5] = centered[:, 0]
    structural[:, 6] = centered[:, 1]
    structural[:, 7] = centered[:, 2]

    for i in range(1, n - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-8 and n2 > 1e-8:
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            structural[i, 8] = np.arccos(cos_angle) / np.pi
        else:
            structural[i, 8] = 0.5
    if n > 2:
        structural[0, 8] = structural[1, 8]
        structural[n - 1, 8] = structural[n - 2, 8]

    max_abs = np.abs(coords).max() + 1e-8
    norm_coords = coords / max_abs
    structural[:, 9] = norm_coords[:, 0]
    structural[:, 10] = norm_coords[:, 1]

    structural[:, 11] = n / 5000.0

    return structural


def build_node_features(residue_names: List[str], coords: np.ndarray) -> np.ndarray:
    """
    Build complete 40-dim node feature matrix for ONE chain.

    Args:
        residue_names: list of 3-letter amino acid codes
        coords: (N, 3) CA coordinates

    Returns:
        features: (N, 40) float32 array
    """
    n = len(residue_names)
    features = np.zeros((n, V10_NODE_DIM), dtype=np.float32)

    for i, name in enumerate(residue_names):
        features[i, :20] = get_onehot_features(name)

    for i, name in enumerate(residue_names):
        features[i, 20:28] = get_physicochemical_features(name)

    structural = compute_structural_features(coords)
    features[:, 28:40] = structural

    return features


# =============================================================================
# Edge Feature Construction (5-dim, multi-chain aware)
# =============================================================================

def build_edges_multichain(
    coords: np.ndarray,
    chain_assignments: np.ndarray,
    seq_indices_within_chain: np.ndarray,
    threshold: float,
    max_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build edges with 5-dim features for multi-chain merged graphs.

    Inter-chain edges get:
      seq_sep = 1.0 (maximum), backbone = 0, local = 0
    Intra-chain edges get standard v9 features.

    Args:
        coords: (N, 3) CA coordinates for ALL residues across all chains
        chain_assignments: (N,) int, chain index per node (0, 1, 2, ...)
        seq_indices_within_chain: (N,) int, sequence position within own chain
        threshold: distance cutoff in Angstroms
        max_neighbors: max edges per node (HARD CAP — critical for memory)

    Returns:
        edge_index: (2, E) int64
        edge_attr: (E, 5) float32
    """
    n = len(coords)
    tree = cKDTree(coords)

    src, dst = [], []
    for i in range(n):
        neighbors = [j for j in tree.query_ball_point(coords[i], r=threshold) if j != i]
        if len(neighbors) > max_neighbors:
            # Keep K nearest neighbors — memory safety valve
            neighbors = sorted(
                neighbors,
                key=lambda j: np.linalg.norm(coords[i] - coords[j])
            )[:max_neighbors]
        for j in neighbors:
            src.append(i)
            dst.append(j)

    num_edges = len(src)
    if num_edges == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, V10_EDGE_DIM), dtype=np.float32)

    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    edge_index = np.stack([src, dst], axis=0)

    # Distances
    dists = np.linalg.norm(coords[src] - coords[dst], axis=1)

    # Same-chain mask
    same_chain = (chain_assignments[src] == chain_assignments[dst])

    # Within-chain sequence separation
    seq_sep = np.abs(
        seq_indices_within_chain[src] - seq_indices_within_chain[dst]
    ).astype(np.float32)

    edge_attr = np.zeros((num_edges, V10_EDGE_DIM), dtype=np.float32)

    # [0] Normalized distance (universal)
    edge_attr[:, 0] = dists / threshold

    # [1] Sequence separation: intra-chain uses actual value, inter-chain = 1.0
    edge_attr[:, 1] = np.where(same_chain, seq_sep / max(n, 1), 1.0)

    # [2] Backbone indicator: only same-chain sequential neighbors
    edge_attr[:, 2] = np.where(same_chain & (seq_sep == 1), 1.0, 0.0)

    # [3] Local contact: only same-chain nearby residues
    edge_attr[:, 3] = np.where(same_chain & (seq_sep <= 4), 1.0, 0.0)

    # [4] Distance decay (universal)
    edge_attr[:, 4] = np.exp(-dists / 5.0)

    return edge_index, edge_attr


# =============================================================================
# PDB Processing — ONE GRAPH PER PDB ENTRY
# =============================================================================

def extract_ca_coord(residue):
    """Extract CA atom coordinate from a residue."""
    if 'CA' in residue:
        return residue['CA'].get_coord()
    return None


def extract_chain_data(chain) -> Optional[Tuple[np.ndarray, List[str], np.ndarray]]:
    """
    Extract residue data from a single chain.

    Returns:
        (coords, resnames, seq_indices) or None if too few residues.
    """
    coords_list = []
    resnames = []
    seq_indices = []
    seq_idx = 0

    for res in chain:
        if not is_aa(res, standard=True):
            continue
        ca = extract_ca_coord(res)
        if ca is None:
            continue
        coords_list.append(ca)
        resnames.append(res.get_resname())
        seq_indices.append(seq_idx)
        seq_idx += 1

    if len(coords_list) < 5:
        return None

    return (
        np.asarray(coords_list, dtype=np.float32),
        resnames,
        np.array(seq_indices, dtype=np.int64),
    )


def process_pdb(pdb_path: str, args) -> List[Data]:
    """
    Process a PDB/CIF file into a SINGLE merged graph (one per PDB entry).

    All chains are merged. Inter-chain edges are created by distance threshold.
    A chain_idx tensor records which chain each node belongs to.

    Returns:
        List with 0 or 1 Data objects.
    """
    pdb_id = Path(pdb_path).stem.upper()

    try:
        if pdb_path.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception:
        return []

    # ---- Collect per-chain data ----
    chains_data = []  # List of (chain_id, coords, resnames, seq_indices)

    for model in structure:
        for chain in model:
            chain_id = chain.id or "A"
            result = extract_chain_data(chain)
            if result is None:
                continue
            coords, resnames, seq_indices = result
            # Per-chain size filter
            if len(coords) < args.min_nodes:
                continue
            chains_data.append((chain_id, coords, resnames, seq_indices))
        break  # first model only

    if not chains_data:
        return []

    # ---- Merge all chains into one graph ----
    all_coords = []
    all_features = []
    all_chain_idx = []
    all_seq_within_chain = []
    chain_ids_list = []

    for chain_num, (chain_id, coords, resnames, seq_indices) in enumerate(chains_data):
        n_chain = len(coords)

        # Build 40-dim features per-chain (preserves intra-chain structural semantics)
        feats = build_node_features(resnames, coords)

        all_coords.append(coords)
        all_features.append(feats)
        all_chain_idx.extend([chain_num] * n_chain)
        all_seq_within_chain.append(seq_indices)
        chain_ids_list.append(chain_id)

    merged_coords = np.concatenate(all_coords, axis=0)
    merged_features = np.concatenate(all_features, axis=0)
    merged_chain_idx = np.array(all_chain_idx, dtype=np.int64)
    merged_seq_within_chain = np.concatenate(all_seq_within_chain, axis=0)

    total_nodes = len(merged_coords)

    # ---- Size filter for merged graph ----
    if total_nodes < args.min_nodes or total_nodes > args.max_nodes:
        return []

    # ---- Build edges with inter-chain awareness ----
    edge_index, edge_attr = build_edges_multichain(
        merged_coords,
        merged_chain_idx,
        merged_seq_within_chain,
        threshold=args.threshold,
        max_neighbors=args.max_neighbors,
    )

    if edge_index.shape[1] == 0:
        return []

    # ---- Assemble Data object ----
    data = Data(
        x=torch.from_numpy(merged_features),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        pos=torch.from_numpy(merged_coords),
        chain_idx=torch.from_numpy(merged_chain_idx),
        graph_id=pdb_id,
        pdb_id=pdb_id,
        chain_ids=chain_ids_list,
        num_chains=len(chains_data),
        num_nodes=total_nodes,
    )

    return [data]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build v10 protein graphs (multi-chain merged, inter-chain edges)")
    parser.add_argument("--pdb-dir", required=True,
                        help="Directory containing PDB/CIF files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save graph batch files")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Distance cutoff for edges in Angstroms (default: 10.0)")
    parser.add_argument("--max-neighbors", type=int, default=32,
                        help="Max neighbors per node — hard cap (default: 32)")
    parser.add_argument("--min-nodes", type=int, default=10,
                        help="Minimum residues per graph (default: 10)")
    parser.add_argument("--max-nodes", type=int, default=5000,
                        help="Maximum residues per graph (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Graphs per batch file (default: 100)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing batch files")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(
        list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.cif"))
    )
    print(f"[INFO] Found {len(pdb_files)} PDB/CIF files in {pdb_dir}")
    print(f"[INFO] v10 mode: one graph per PDB (multi-chain merged)")
    print(f"[INFO] Threshold: {args.threshold} A, Max neighbors: {args.max_neighbors}")

    # Resume support
    processed_graphs, processed_pdbs, batch_num = set(), set(), 0
    if args.resume:
        processed_graphs, processed_pdbs, batch_num = recover_processed_from_batches(output_dir)

    remaining = [p for p in pdb_files if p.stem.upper() not in processed_pdbs]
    print(f"[INFO] Processing {len(remaining)} remaining PDBs "
          f"(skipped {len(pdb_files) - len(remaining)} already done)")

    all_graphs = []
    total_graphs = len(processed_graphs)
    failed = 0
    multi_chain_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_pdb, str(p), args): p for p in remaining}

        with tqdm(total=len(futures), desc="Processing PDBs (v10 multi-chain)") as pbar:
            for fut in as_completed(futures):
                try:
                    graphs = fut.result()
                    for g in graphs:
                        if g.graph_id in processed_graphs:
                            continue
                        all_graphs.append(g)
                        processed_graphs.add(g.graph_id)
                        total_graphs += 1
                        if g.num_chains > 1:
                            multi_chain_count += 1

                        if len(all_graphs) >= args.batch_size:
                            out_path = output_dir / f"graphs_batch_{batch_num:04d}.pt"
                            torch.save(all_graphs, out_path)
                            batch_num += 1
                            all_graphs = []
                except Exception:
                    failed += 1
                pbar.update(1)

    if all_graphs:
        out_path = output_dir / f"graphs_batch_{batch_num:04d}.pt"
        torch.save(all_graphs, out_path)
        batch_num += 1

    print(f"\n{'='*60}")
    print(f"  v10 Graph Construction Complete (Inter-Chain Aware)")
    print(f"{'='*60}")
    print(f"  Total graphs:        {total_graphs}")
    print(f"  Multi-chain graphs:  {multi_chain_count}")
    print(f"  Single-chain graphs: {total_graphs - multi_chain_count}")
    print(f"  Batch files:         {batch_num}")
    print(f"  Failed PDBs:         {failed}")
    print(f"  Output dir:          {output_dir}")
    print(f"  Node dim:            {V10_NODE_DIM}")
    print(f"  Edge dim:            {V10_EDGE_DIM}")
    print(f"  Max neighbors:       {args.max_neighbors}")
    print(f"  Threshold:           {args.threshold} A")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
