"""
GO Annotation Handler

Parses GO annotation files and creates:
1. GO term vocabularies for MF, BP, CC ontologies
2. Multi-label binary target vectors
3. GO term frequency statistics

Annotation file format (TSV):
    PDB-chain    GO-terms (MF)    GO-terms (BP)    GO-terms (CC)
    1A2B-A       GO:0001,GO:0002  GO:0003          GO:0004,GO:0005
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class GOAnnotationParser:
    """
    Parser for GO annotation files.
    
    Creates vocabularies and multi-label targets for protein function prediction.
    """
    
    def __init__(self, annotation_file: str, min_count: int = 0):
        """
        Initialize parser.
        
        Args:
            annotation_file: Path to TSV annotation file
            min_count: Minimum number of proteins for a GO term to be included
        """
        self.annotation_file = Path(annotation_file)
        self.min_count = min_count
        
        # Annotations: {graph_id: {'mf': [...], 'bp': [...], 'cc': [...]}}
        self.annotations: Dict[str, Dict[str, List[str]]] = {}
        
        # Vocabularies: ordered list of GO terms
        self.mf_terms: List[str] = []
        self.bp_terms: List[str] = []
        self.cc_terms: List[str] = []
        
        # GO term to index mappings
        self.mf_to_idx: Dict[str, int] = {}
        self.bp_to_idx: Dict[str, int] = {}
        self.cc_to_idx: Dict[str, int] = {}
        
        # Statistics
        self.mf_counts: Dict[str, int] = defaultdict(int)
        self.bp_counts: Dict[str, int] = defaultdict(int)
        self.cc_counts: Dict[str, int] = defaultdict(int)
        
        # Parse the file
        self._parse_file()
        self._build_vocabularies()
    
    def _parse_file(self):
        """Parse the annotation TSV file."""
        print(f"Parsing annotation file: {self.annotation_file}")
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the data section (after GO-names headers)
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('### PDB-chain'):
                data_start = i + 1
                break
        
        # Parse each annotation line
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            pdb_chain = parts[0].strip()
            
            # Parse GO terms for each ontology
            mf_terms = self._parse_go_terms(parts[1]) if len(parts) > 1 else []
            bp_terms = self._parse_go_terms(parts[2]) if len(parts) > 2 else []
            cc_terms = self._parse_go_terms(parts[3]) if len(parts) > 3 else []
            
            if not (mf_terms or bp_terms or cc_terms):
                continue
            
            self.annotations[pdb_chain] = {
                'mf': mf_terms,
                'bp': bp_terms,
                'cc': cc_terms
            }
            
            # Update counts
            for term in mf_terms:
                self.mf_counts[term] += 1
            for term in bp_terms:
                self.bp_counts[term] += 1
            for term in cc_terms:
                self.cc_counts[term] += 1
        
        print(f"Parsed {len(self.annotations)} annotated proteins")
        print(f"Unique GO terms: MF={len(self.mf_counts)}, BP={len(self.bp_counts)}, CC={len(self.cc_counts)}")
    
    def _parse_go_terms(self, term_string: str) -> List[str]:
        """Parse comma-separated GO terms."""
        if not term_string or term_string.strip() == '':
            return []
        terms = [t.strip() for t in term_string.split(',')]
        return [t for t in terms if t.startswith('GO:')]
    
    def _build_vocabularies(self):
        """Build GO term vocabularies with optional filtering."""
        # Filter by minimum count and sort for reproducibility
        self.mf_terms = sorted([
            t for t, c in self.mf_counts.items() if c >= self.min_count
        ])
        self.bp_terms = sorted([
            t for t, c in self.bp_counts.items() if c >= self.min_count
        ])
        self.cc_terms = sorted([
            t for t, c in self.cc_counts.items() if c >= self.min_count
        ])
        
        # Build index mappings
        self.mf_to_idx = {t: i for i, t in enumerate(self.mf_terms)}
        self.bp_to_idx = {t: i for i, t in enumerate(self.bp_terms)}
        self.cc_to_idx = {t: i for i, t in enumerate(self.cc_terms)}
        
        print(f"Vocabulary sizes (after filtering): MF={len(self.mf_terms)}, "
              f"BP={len(self.bp_terms)}, CC={len(self.cc_terms)}")
    
    @property
    def num_mf(self) -> int:
        return len(self.mf_terms)
    
    @property
    def num_bp(self) -> int:
        return len(self.bp_terms)
    
    @property
    def num_cc(self) -> int:
        return len(self.cc_terms)
    
    def get_graph_ids(self) -> List[str]:
        """Get all annotated graph IDs."""
        return list(self.annotations.keys())
    
    def has_annotation(self, graph_id: str) -> bool:
        """Check if a graph has annotations."""
        return graph_id in self.annotations
    
    def get_labels(self, graph_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get multi-label binary targets for a graph.
        
        Args:
            graph_id: PDB-chain identifier (e.g., "1A2B-A")
            
        Returns:
            Tuple of (mf_labels, bp_labels, cc_labels) as float32 tensors
        """
        if graph_id not in self.annotations:
            # Return zero vectors if not annotated
            return (
                torch.zeros(self.num_mf, dtype=torch.float32),
                torch.zeros(self.num_bp, dtype=torch.float32),
                torch.zeros(self.num_cc, dtype=torch.float32)
            )
        
        annot = self.annotations[graph_id]
        
        # Build binary vectors
        mf_vec = torch.zeros(self.num_mf, dtype=torch.float32)
        bp_vec = torch.zeros(self.num_bp, dtype=torch.float32)
        cc_vec = torch.zeros(self.num_cc, dtype=torch.float32)
        
        for term in annot['mf']:
            if term in self.mf_to_idx:
                mf_vec[self.mf_to_idx[term]] = 1.0
        
        for term in annot['bp']:
            if term in self.bp_to_idx:
                bp_vec[self.bp_to_idx[term]] = 1.0
        
        for term in annot['cc']:
            if term in self.cc_to_idx:
                cc_vec[self.cc_to_idx[term]] = 1.0
        
        return mf_vec, bp_vec, cc_vec
    
    def get_raw_annotations(self, graph_id: str) -> Optional[Dict[str, List[str]]]:
        """Get raw GO term lists for a graph."""
        return self.annotations.get(graph_id)
    
    def compute_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute class weights for handling label imbalance.
        
        Uses inverse frequency weighting: weight = total_samples / (num_classes * class_count)
        
        Returns:
            Tuple of (mf_weights, bp_weights, cc_weights) tensors
        """
        total_samples = len(self.annotations)
        
        def compute_weights(terms, counts):
            weights = []
            for term in terms:
                count = counts.get(term, 1)
                # Avoid division by zero
                weight = total_samples / (len(terms) * max(count, 1))
                weights.append(weight)
            return torch.tensor(weights, dtype=torch.float32)
        
        mf_weights = compute_weights(self.mf_terms, self.mf_counts)
        bp_weights = compute_weights(self.bp_terms, self.bp_counts)
        cc_weights = compute_weights(self.cc_terms, self.cc_counts)
        
        return mf_weights, bp_weights, cc_weights
    
    def get_statistics(self) -> Dict:
        """Get annotation statistics."""
        # Compute label statistics per protein
        mf_per_protein = []
        bp_per_protein = []
        cc_per_protein = []
        
        for annot in self.annotations.values():
            mf_per_protein.append(len([t for t in annot['mf'] if t in self.mf_to_idx]))
            bp_per_protein.append(len([t for t in annot['bp'] if t in self.bp_to_idx]))
            cc_per_protein.append(len([t for t in annot['cc'] if t in self.cc_to_idx]))
        
        return {
            'total_proteins': len(self.annotations),
            'num_mf_terms': self.num_mf,
            'num_bp_terms': self.num_bp,
            'num_cc_terms': self.num_cc,
            'avg_mf_per_protein': np.mean(mf_per_protein) if mf_per_protein else 0,
            'avg_bp_per_protein': np.mean(bp_per_protein) if bp_per_protein else 0,
            'avg_cc_per_protein': np.mean(cc_per_protein) if cc_per_protein else 0,
            'mf_term_frequencies': dict(self.mf_counts),
            'bp_term_frequencies': dict(self.bp_counts),
            'cc_term_frequencies': dict(self.cc_counts)
        }
    
    def save_vocabulary(self, output_dir: str):
        """Save vocabularies to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vocab = {
            'mf_terms': self.mf_terms,
            'bp_terms': self.bp_terms,
            'cc_terms': self.cc_terms,
            'mf_to_idx': self.mf_to_idx,
            'bp_to_idx': self.bp_to_idx,
            'cc_to_idx': self.cc_to_idx
        }
        
        with open(output_dir / 'go_vocabulary.json', 'w') as f:
            json.dump(vocab, f, indent=2)
        
        print(f"Vocabulary saved to {output_dir / 'go_vocabulary.json'}")
    
    def load_vocabulary(self, vocab_file: str):
        """Load vocabularies from disk."""
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        
        self.mf_terms = vocab['mf_terms']
        self.bp_terms = vocab['bp_terms']
        self.cc_terms = vocab['cc_terms']
        self.mf_to_idx = vocab['mf_to_idx']
        self.bp_to_idx = vocab['bp_to_idx']
        self.cc_to_idx = vocab['cc_to_idx']
        
        print(f"Loaded vocabulary: MF={self.num_mf}, BP={self.num_bp}, CC={self.num_cc}")


class GOTermAnalyzer:
    """
    Analyze GO term distributions for subset selection.
    """
    
    def __init__(self, parser: GOAnnotationParser):
        self.parser = parser
    
    def get_term_distribution(self, ontology: str = 'mf') -> Dict[str, int]:
        """Get GO term frequency distribution."""
        if ontology == 'mf':
            return dict(self.parser.mf_counts)
        elif ontology == 'bp':
            return dict(self.parser.bp_counts)
        elif ontology == 'cc':
            return dict(self.parser.cc_counts)
        else:
            raise ValueError(f"Unknown ontology: {ontology}")
    
    def get_rare_terms(self, ontology: str = 'mf', threshold: int = 10) -> List[str]:
        """Get GO terms that appear in fewer than threshold proteins."""
        counts = self.get_term_distribution(ontology)
        return [t for t, c in counts.items() if c < threshold]
    
    def get_proteins_with_term(self, term: str) -> List[str]:
        """Get all proteins annotated with a specific GO term."""
        proteins = []
        for graph_id, annot in self.parser.annotations.items():
            all_terms = annot['mf'] + annot['bp'] + annot['cc']
            if term in all_terms:
                proteins.append(graph_id)
        return proteins
    
    def get_coverage_by_proteins(self, protein_ids: List[str]) -> Dict[str, float]:
        """
        Calculate what fraction of each GO term is covered by a protein subset.
        
        Returns:
            Dict mapping GO term to coverage fraction (0-1)
        """
        coverage = {}
        
        # Count terms in subset
        subset_mf = defaultdict(int)
        subset_bp = defaultdict(int)
        subset_cc = defaultdict(int)
        
        for pid in protein_ids:
            if pid in self.parser.annotations:
                annot = self.parser.annotations[pid]
                for t in annot['mf']:
                    subset_mf[t] += 1
                for t in annot['bp']:
                    subset_bp[t] += 1
                for t in annot['cc']:
                    subset_cc[t] += 1
        
        # Calculate coverage for each term
        for term in self.parser.mf_terms:
            total = self.parser.mf_counts.get(term, 1)
            subset_count = subset_mf.get(term, 0)
            coverage[term] = subset_count / total
        
        for term in self.parser.bp_terms:
            total = self.parser.bp_counts.get(term, 1)
            subset_count = subset_bp.get(term, 0)
            coverage[term] = subset_count / total
        
        for term in self.parser.cc_terms:
            total = self.parser.cc_counts.get(term, 1)
            subset_count = subset_cc.get(term, 0)
            coverage[term] = subset_count / total
        
        return coverage


if __name__ == '__main__':
    # Test the annotation parser
    import sys
    
    annotation_file = 'annotations/nrPDB-GO_2019.06.18_annot.tsv'
    
    if not Path(annotation_file).exists():
        print(f"Annotation file not found: {annotation_file}")
        sys.exit(1)
    
    # Parse annotations
    parser = GOAnnotationParser(annotation_file)
    
    # Print statistics
    stats = parser.get_statistics()
    print(f"\n{'='*60}")
    print("GO Annotation Statistics")
    print(f"{'='*60}")
    print(f"Total annotated proteins: {stats['total_proteins']}")
    print(f"\nGO Term counts:")
    print(f"  Molecular Function (MF): {stats['num_mf_terms']}")
    print(f"  Biological Process (BP): {stats['num_bp_terms']}")
    print(f"  Cellular Component (CC): {stats['num_cc_terms']}")
    print(f"\nAverage GO terms per protein:")
    print(f"  MF: {stats['avg_mf_per_protein']:.2f}")
    print(f"  BP: {stats['avg_bp_per_protein']:.2f}")
    print(f"  CC: {stats['avg_cc_per_protein']:.2f}")
    
    # Test label generation
    print(f"\n{'='*60}")
    print("Testing label generation")
    print(f"{'='*60}")
    
    sample_ids = list(parser.annotations.keys())[:3]
    for graph_id in sample_ids:
        mf, bp, cc = parser.get_labels(graph_id)
        raw = parser.get_raw_annotations(graph_id)
        print(f"\n{graph_id}:")
        print(f"  MF terms: {len(raw['mf'])} -> {mf.sum().item():.0f} in vocab")
        print(f"  BP terms: {len(raw['bp'])} -> {bp.sum().item():.0f} in vocab")
        print(f"  CC terms: {len(raw['cc'])} -> {cc.sum().item():.0f} in vocab")
    
    # Save vocabulary
    parser.save_vocabulary('.')
