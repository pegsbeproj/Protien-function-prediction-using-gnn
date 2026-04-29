"""
GO Hierarchy (DAG) Utilities for v12

Parses Gene Ontology OBO files and builds:
  1. Parent-child edge lists (restricted to vocabulary terms)
  2. Ancestor matrices for post-processing propagation
  3. Ancestor closure verification for annotation files

The GO DAG is a directed acyclic graph where edges represent:
  - is_a: "X is_a Y" means X is a subtype of Y
  - part_of: "X part_of Y" means X is part of Y

True-path rule: if a protein is annotated with term X,
it should also be annotated with all ancestors of X.

Usage:
    from go_hierarchy import GOHierarchy

    hier = GOHierarchy('annotations/go-basic.obo')
    child_idx, parent_idx = hier.get_edges(go_terms)
    ancestor_matrix = hier.get_ancestor_matrix(go_terms)
"""

import os
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


# ════════════════════════════════════════════════════════════════
#  OBO Parser
# ════════════════════════════════════════════════════════════════

def parse_obo(obo_path: str) -> Dict[str, List[str]]:
    """
    Parse GO OBO file → dict mapping child_id → [parent_ids].

    Extracts is_a and part_of relationships.

    Args:
        obo_path: Path to go-basic.obo file

    Returns:
        Dict mapping GO term ID → list of direct parent IDs
    """
    child_to_parents: Dict[str, List[str]] = defaultdict(list)
    term_namespace: Dict[str, str] = {}

    current_id = None
    current_namespace = None
    is_obsolete = False

    with open(obo_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '[Term]':
                current_id = None
                current_namespace = None
                is_obsolete = False
            elif line == '[Typedef]':
                current_id = None  # skip typedefs
            elif line.startswith('id: GO:'):
                current_id = line.split('id: ')[1].strip()
            elif line.startswith('namespace:'):
                current_namespace = line.split('namespace: ')[1].strip()
                if current_id:
                    term_namespace[current_id] = current_namespace
            elif line.startswith('is_obsolete: true'):
                is_obsolete = True
            elif line.startswith('is_a:') and current_id and not is_obsolete:
                parent = line.split('is_a: ')[1].split('!')[0].strip()
                if parent.startswith('GO:'):
                    child_to_parents[current_id].append(parent)
            elif line.startswith('relationship: part_of') and current_id and not is_obsolete:
                parts = line.split('part_of ')
                if len(parts) > 1:
                    parent = parts[1].split('!')[0].strip()
                    if parent.startswith('GO:'):
                        child_to_parents[current_id].append(parent)

    return dict(child_to_parents), term_namespace


def _get_transitive_ancestors(
    term: str,
    child_to_parents: Dict[str, List[str]],
    cache: Dict[str, Set[str]]
) -> Set[str]:
    """Get all ancestors of a term (transitive closure), with memoization."""
    if term in cache:
        return cache[term]

    ancestors = set()
    for parent in child_to_parents.get(term, []):
        ancestors.add(parent)
        ancestors |= _get_transitive_ancestors(parent, child_to_parents, cache)

    cache[term] = ancestors
    return ancestors


# ════════════════════════════════════════════════════════════════
#  GO Hierarchy Class
# ════════════════════════════════════════════════════════════════

class GOHierarchy:
    """
    GO DAG wrapper for hierarchical consistency loss and ancestor propagation.

    Loads the GO OBO file once and provides:
      - get_edges(): child→parent index pairs for vocabulary terms
      - get_ancestor_matrix(): binary [n, n] ancestor matrix
      - verify_ancestor_closure(): check if annotations are ancestor-expanded
      - propagate_labels(): apply true-path rule to label vectors
    """

    def __init__(self, obo_path: str):
        """
        Args:
            obo_path: Path to go-basic.obo file
        """
        self.obo_path = obo_path

        if not Path(obo_path).exists():
            raise FileNotFoundError(
                f"GO OBO file not found: {obo_path}\n"
                f"Download from: http://release.geneontology.org/2019-07-01/ontology/go-basic.obo\n"
                f"Place in: annotations/go-basic.obo"
            )

        print(f"[GO-DAG] Parsing OBO file: {obo_path}")
        self.child_to_parents, self.term_namespace = parse_obo(obo_path)
        self._ancestor_cache: Dict[str, Set[str]] = {}

        n_terms = len(self.child_to_parents)
        n_edges = sum(len(v) for v in self.child_to_parents.values())
        print(f"[GO-DAG] Parsed {n_terms} terms with {n_edges} direct edges")

    def get_ancestors(self, term: str) -> Set[str]:
        """Get all transitive ancestors of a GO term."""
        return _get_transitive_ancestors(term, self.child_to_parents, self._ancestor_cache)

    def get_edges(self, go_terms: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build child→parent edge index restricted to vocabulary terms.

        If a direct parent is not in the vocabulary, we BFS upward to find
        the nearest ancestor that IS in the vocabulary. This ensures no
        hierarchy information is lost due to vocabulary filtering.

        Args:
            go_terms: Ordered list of GO term IDs in the vocabulary

        Returns:
            (child_indices, parent_indices): np.int64 arrays
            Each pair (child_indices[i], parent_indices[i]) means
            go_terms[child_idx] is_a go_terms[parent_idx]
        """
        term_set = set(go_terms)
        term_to_idx = {t: i for i, t in enumerate(go_terms)}

        child_idxs = []
        parent_idxs = []
        seen_edges = set()

        for term in go_terms:
            # Direct parents in vocabulary
            direct_parents_in_vocab = set()
            for parent in self.child_to_parents.get(term, []):
                if parent in term_set:
                    direct_parents_in_vocab.add(parent)

            # If no direct parents in vocab, BFS to find nearest vocab ancestors
            if not direct_parents_in_vocab:
                queue = list(self.child_to_parents.get(term, []))
                visited = set()
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)
                    if node in term_set:
                        direct_parents_in_vocab.add(node)
                        # Don't continue past a vocab term (nearest ancestor)
                    else:
                        queue.extend(self.child_to_parents.get(node, []))

            for parent in direct_parents_in_vocab:
                if parent != term:
                    edge = (term_to_idx[term], term_to_idx[parent])
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        child_idxs.append(edge[0])
                        parent_idxs.append(edge[1])

        child_arr = np.array(child_idxs, dtype=np.int64) if child_idxs else np.array([], dtype=np.int64)
        parent_arr = np.array(parent_idxs, dtype=np.int64) if parent_idxs else np.array([], dtype=np.int64)

        print(f"[GO-DAG] Vocabulary {len(go_terms)} terms → {len(child_idxs)} DAG edges")
        return child_arr, parent_arr

    def get_ancestor_matrix(self, go_terms: List[str]) -> np.ndarray:
        """
        Build binary ancestor matrix for post-processing propagation.

        Returns:
            ancestor_matrix: [n, n] float32 where [i, j]=1 means
            go_terms[j] is a (transitive) ancestor of go_terms[i]
        """
        term_set = set(go_terms)
        term_to_idx = {t: i for i, t in enumerate(go_terms)}
        n = len(go_terms)

        matrix = np.zeros((n, n), dtype=np.float32)

        for term in go_terms:
            ancestors = self.get_ancestors(term)
            term_idx = term_to_idx[term]
            for anc in ancestors:
                if anc in term_to_idx:
                    matrix[term_idx, term_to_idx[anc]] = 1.0

        n_edges = int(matrix.sum())
        print(f"[GO-DAG] Ancestor matrix: {n}×{n}, {n_edges} ancestor relationships")
        return matrix

    def verify_ancestor_closure(
        self,
        annotations: Dict[str, Dict[str, List[str]]],
        go_terms_mf: List[str],
        go_terms_bp: List[str],
        go_terms_cc: List[str],
        sample_size: int = 500,
    ) -> Dict[str, float]:
        """
        Verify whether annotations are ancestor-expanded.

        For a sample of proteins, check what fraction of expected ancestor
        terms are present in the annotations. Returns closure fraction
        per ontology (1.0 = fully closed, <0.5 = likely leaf-only).

        Args:
            annotations: {pdb_chain: {'mf': [...], 'bp': [...], 'cc': [...]}}
            go_terms_mf: MF vocabulary
            go_terms_bp: BP vocabulary
            go_terms_cc: CC vocabulary
            sample_size: Number of proteins to check

        Returns:
            Dict with 'mf', 'bp', 'cc' closure fractions and 'avg_terms_bp'
        """
        import random
        results = {}

        pids = list(annotations.keys())
        sample = random.sample(pids, min(sample_size, len(pids)))

        for ont_name, go_terms, key in [
            ('mf', go_terms_mf, 'mf'),
            ('bp', go_terms_bp, 'bp'),
            ('cc', go_terms_cc, 'cc'),
        ]:
            term_set = set(go_terms)
            expected = 0
            present = 0
            term_counts = []

            for pid in sample:
                annot_terms = set(annotations[pid].get(key, []))
                term_counts.append(len(annot_terms))

                for term in annot_terms:
                    if term not in term_set:
                        continue
                    # Check if all vocab ancestors are present
                    ancestors = self.get_ancestors(term)
                    vocab_ancestors = ancestors & term_set
                    expected += len(vocab_ancestors)
                    present += len(vocab_ancestors & annot_terms)

            closure = present / max(expected, 1)
            avg_terms = np.mean(term_counts) if term_counts else 0

            results[ont_name] = closure
            results[f'avg_terms_{ont_name}'] = avg_terms
            print(f"[GO-DAG] {ont_name.upper()} ancestor closure: {closure:.3f} "
                  f"(avg {avg_terms:.1f} terms/protein, "
                  f"{present}/{expected} ancestor terms present)")

        return results

    def propagate_annotation_labels(
        self,
        label_vector: np.ndarray,
        ancestor_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Apply true-path rule to binary label vectors.

        If label_vector[i] = 1 and ancestor_matrix[i, j] = 1,
        then set label_vector[j] = 1.

        Args:
            label_vector: [batch, n_terms] or [n_terms] binary labels
            ancestor_matrix: [n_terms, n_terms] from get_ancestor_matrix()

        Returns:
            Propagated label vector (same shape)
        """
        was_1d = label_vector.ndim == 1
        if was_1d:
            label_vector = label_vector[np.newaxis, :]

        # For each positive label, set all its ancestors to positive
        propagated = label_vector.copy()
        # Matrix multiply: if label[i]=1, add all ancestor_matrix[i, :] entries
        ancestor_additions = (label_vector @ ancestor_matrix) > 0
        propagated = np.maximum(propagated, ancestor_additions.astype(propagated.dtype))

        if was_1d:
            propagated = propagated[0]
        return propagated


# ════════════════════════════════════════════════════════════════
#  Hierarchical Consistency Loss
# ════════════════════════════════════════════════════════════════

class HierarchicalConsistencyLoss(torch.nn.Module):
    """
    Penalizes predictions where P(child) > P(parent) in the GO DAG.

    For each directed edge (child → parent) in the DAG:
        loss += max(0, sigmoid(z_child) - sigmoid(z_parent))^2

    This is a soft constraint — it adds gradient pressure toward
    hierarchy-consistent predictions without hard-clamping.

    Memory overhead: ~40KB per ontology (two int64 index tensors).
    """

    def __init__(
        self,
        child_indices: np.ndarray,
        parent_indices: np.ndarray,
        weight: float = 0.05,
        margin: float = 0.0,
    ):
        """
        Args:
            child_indices: [E] indices of child terms
            parent_indices: [E] indices of parent terms
            weight: Loss weight multiplier
            margin: Optional margin (default 0 = strict consistency)
        """
        super().__init__()
        self.weight = weight
        self.margin = margin

        if len(child_indices) > 0:
            self.register_buffer('child_idx', torch.from_numpy(child_indices).long())
            self.register_buffer('parent_idx', torch.from_numpy(parent_indices).long())
            self.has_edges = True
        else:
            self.has_edges = False

        n_edges = len(child_indices)
        print(f"  HierarchicalConsistencyLoss: {n_edges} edges, weight={weight}, margin={margin}")

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, n_classes] raw logits

        Returns:
            Scalar loss (already weighted)
        """
        if not self.has_edges:
            return torch.tensor(0.0, device=logits.device)

        probs = torch.sigmoid(logits)

        child_probs = probs[:, self.child_idx]    # [B, E]
        parent_probs = probs[:, self.parent_idx]  # [B, E]

        # Violation: child prob exceeds parent prob (+ margin)
        violation = torch.clamp(
            child_probs - parent_probs + self.margin,
            min=0.0
        )

        return self.weight * (violation ** 2).mean()


# ════════════════════════════════════════════════════════════════
#  Ancestor Propagator (post-processing at inference)
# ════════════════════════════════════════════════════════════════

class AncestorPropagator:
    """
    Post-processing: propagate prediction scores up the GO DAG.

    For each ancestor term, set its score to the max of its own score
    and all descendant scores. Guarantees hierarchy-consistent outputs.

    Used at eval/test time only — does not affect training.
    """

    def __init__(self, ancestor_matrix: np.ndarray):
        """
        Args:
            ancestor_matrix: [n_terms, n_terms] where [i,j]=1 means
                            term j is ancestor of term i
        """
        self.ancestor_matrix = ancestor_matrix
        self.n_terms = ancestor_matrix.shape[0]

        # Pre-compute descendant lists for each term (for efficient propagation)
        # descendant_map[j] = list of term indices i where j is ancestor of i
        self.descendant_map: Dict[int, List[int]] = {}
        for j in range(self.n_terms):
            descendants = np.where(ancestor_matrix[:, j] > 0)[0]
            if len(descendants) > 0:
                self.descendant_map[j] = descendants.tolist()

        n_with_desc = len(self.descendant_map)
        print(f"  AncestorPropagator: {n_with_desc}/{self.n_terms} terms have descendants")

    def propagate(self, scores: np.ndarray) -> np.ndarray:
        """
        Propagate prediction scores upward through the DAG.

        Args:
            scores: [batch_size, n_terms] prediction scores (probabilities)

        Returns:
            propagated: [batch_size, n_terms] with hierarchy-consistent scores
        """
        propagated = scores.copy()

        for parent_idx, desc_indices in self.descendant_map.items():
            desc_max = scores[:, desc_indices].max(axis=1)
            propagated[:, parent_idx] = np.maximum(
                propagated[:, parent_idx],
                desc_max
            )

        return propagated


# ════════════════════════════════════════════════════════════════
#  Convenience: Download OBO
# ════════════════════════════════════════════════════════════════

def download_obo(output_path: str = "annotations/go-basic.obo", version: str = "2019-07-01"):
    """
    Download GO OBO file matching the nrPDB-GO 2019.06.18 annotation release.

    Args:
        output_path: Where to save the OBO file
        version: GO release date (default matches nrPDB-GO 2019 annotations)
    """
    url = f"http://release.geneontology.org/{version}/ontology/go-basic.obo"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"[GO-DAG] OBO file already exists: {output} ({size_mb:.1f} MB)")
        return str(output)

    print(f"[GO-DAG] Downloading GO OBO from: {url}")
    print(f"[GO-DAG] Saving to: {output}")

    try:
        urllib.request.urlretrieve(url, str(output))
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"[GO-DAG] Downloaded successfully ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"[GO-DAG] Download failed: {e}")
        print(f"[GO-DAG] Please download manually from:")
        print(f"  {url}")
        print(f"  and place at: {output}")
        raise

    return str(output)


# ════════════════════════════════════════════════════════════════
#  Main: standalone testing
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    obo_path = "annotations/go-basic.obo"

    if not Path(obo_path).exists():
        print("Downloading GO OBO file...")
        try:
            download_obo(obo_path)
        except Exception:
            print("Download failed. Please download manually.")
            sys.exit(1)

    # Load hierarchy
    hier = GOHierarchy(obo_path)

    # Load annotation parser to get vocabularies
    from .annotations import GOAnnotationParser
    parser = GOAnnotationParser("annotations/nrPDB-GO_2019.06.18_annot.tsv")

    print(f"\nVocabulary sizes: MF={parser.num_mf}, BP={parser.num_bp}, CC={parser.num_cc}")

    # Build edges for each ontology
    for ont_name, terms in [('MF', parser.mf_terms), ('BP', parser.bp_terms), ('CC', parser.cc_terms)]:
        c_idx, p_idx = hier.get_edges(terms)
        anc_mat = hier.get_ancestor_matrix(terms)
        print(f"\n{ont_name}:")
        print(f"  Terms: {len(terms)}")
        print(f"  Direct DAG edges: {len(c_idx)}")
        print(f"  Ancestor matrix density: {anc_mat.sum() / max(anc_mat.size, 1):.4f}")
        print(f"  Avg ancestors per term: {anc_mat.sum(axis=1).mean():.1f}")
        print(f"  Max ancestors: {anc_mat.sum(axis=1).max():.0f}")

    # Verify ancestor closure
    print("\n--- Ancestor Closure Verification ---")
    closure = hier.verify_ancestor_closure(
        parser.annotations, parser.mf_terms, parser.bp_terms, parser.cc_terms
    )
    for k, v in closure.items():
        print(f"  {k}: {v:.4f}")
