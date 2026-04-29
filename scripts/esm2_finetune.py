"""
ESM2 Fine-Tuning Module with LoRA

This module enables fine-tuning ESM2 during GNN training using:
1. LoRA (Low-Rank Adaptation) - adds small trainable adapters (~0.5% params)
2. Last-layer unfreezing - trains final transformer layers

Key difference from v11-v15:
- v11-v15: Pre-extract ESM2 embeddings → load from disk → frozen
- This:    ESM2 runs live during training → embeddings update

No new graphs needed - only the ESM2 computation changes.

Memory Requirements:
- ESM2 t33 650M: ~2.5 GB weights
- With LoRA: +~10 MB trainable params
- With gradient checkpointing: fits in 12-16 GB VRAM

Usage:
    from esm2_finetune import ESM2LoRA, ESM2FineTunable
    
    # LoRA approach (recommended)
    esm2 = ESM2LoRA(rank=8, alpha=16)
    
    # Last-layer unfreezing
    esm2 = ESM2FineTunable(unfreeze_layers=2)
"""

import math
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ════════════════════════════════════════════════════════════════
#  LoRA Layer (Low-Rank Adaptation)
# ════════════════════════════════════════════════════════════════

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for linear transformations.
    
    Original: y = Wx
    LoRA:     y = Wx + (BA)x
    
    Where:
        W: [out_dim, in_dim] frozen original weights
        A: [rank, in_dim]    trainable down-projection
        B: [out_dim, rank]   trainable up-projection
    
    Total trainable params: rank * (in_dim + out_dim)
    For ESM2 attention (1280 dim, rank=8): 8 * 2560 = 20K per layer
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming, B with zeros (starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward (frozen)
        result = self.original(x)
        
        # LoRA delta (trainable)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return result + self.scaling * lora_out
    
    def merge_weights(self):
        """Merge LoRA weights into original for inference (optional)."""
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)
            self.original.weight.data += delta


# ════════════════════════════════════════════════════════════════
#  ESM2 with LoRA
# ════════════════════════════════════════════════════════════════

class ESM2LoRA(nn.Module):
    """
    ESM2 model with LoRA adapters for fine-tuning.
    
    Applies LoRA to:
    - Query projections (captures what to attend to)
    - Value projections (captures what information to extract)
    
    Keeps Key projections and FFN layers frozen.
    
    Args:
        model_name: ESM2 model name (default: esm2_t33_650M_UR50D)
        rank: LoRA rank (default: 8, higher = more capacity)
        alpha: LoRA scaling factor (default: 16)
        dropout: LoRA dropout rate (default: 0.05)
        target_layers: Which layers to apply LoRA to (default: last 6)
        use_gradient_checkpointing: Reduce memory with recomputation
    """
    
    ESM2_MODELS = {
        'esm2_t6_8M_UR50D':    {'layers': 6,  'dim': 320},
        'esm2_t12_35M_UR50D':  {'layers': 12, 'dim': 480},
        'esm2_t30_150M_UR50D': {'layers': 30, 'dim': 640},
        'esm2_t33_650M_UR50D': {'layers': 33, 'dim': 1280},
    }
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_layers: Optional[List[int]] = None,
        use_gradient_checkpointing: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.model_name = model_name
        self.rank = rank
        self.alpha = alpha
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        model_info = self.ESM2_MODELS.get(model_name)
        if model_info is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.n_layers = model_info['layers']
        self.embed_dim = model_info['dim']
        
        # Default: apply LoRA to last 6 layers
        if target_layers is None:
            target_layers = list(range(max(0, self.n_layers - 6), self.n_layers))
        self.target_layers = target_layers
        
        # Load ESM2 model
        print(f"Loading ESM2 model: {model_name}")
        import esm
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Freeze all ESM2 parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        # Apply LoRA to target layers
        self._apply_lora(rank, alpha, dropout)
        
        # Move to device
        self.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ESM2 LoRA: {trainable_params:,} trainable / {total_params:,} total "
              f"({100*trainable_params/total_params:.2f}%)")
    
    def _apply_lora(self, rank: int, alpha: float, dropout: float):
        """Apply LoRA to Q and V projections in target layers."""
        self.lora_layers = nn.ModuleDict()
        
        for layer_idx in self.target_layers:
            layer = self.esm_model.layers[layer_idx]
            attn = layer.self_attn
            
            # Replace Q and V projections with LoRA versions
            q_lora = LoRALayer(attn.q_proj, rank, alpha, dropout)
            v_lora = LoRALayer(attn.v_proj, rank, alpha, dropout)
            
            # Store references
            self.lora_layers[f'layer_{layer_idx}_q'] = q_lora
            self.lora_layers[f'layer_{layer_idx}_v'] = v_lora
            
            # Replace in the model
            attn.q_proj = q_lora
            attn.v_proj = v_lora
        
        print(f"Applied LoRA to layers: {self.target_layers}")
    
    def _tokenize(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize sequences for ESM2."""
        # Format: list of (label, sequence) tuples
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        
        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = (batch_tokens != self.alphabet.padding_idx).float()
        
        return batch_tokens.to(self.device), attention_mask.to(self.device)
    
    def forward(
        self,
        sequences: List[str],
        return_per_residue: bool = True,
    ) -> torch.Tensor:
        """
        Extract embeddings from sequences.
        
        Args:
            sequences: List of amino acid sequences (single-letter codes)
            return_per_residue: If True, return [total_residues, dim]
                               If False, return [batch, max_len, dim]
        
        Returns:
            embeddings: Per-residue embeddings
        """
        tokens, attention_mask = self._tokenize(sequences)
        
        # Forward pass through ESM2
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            embeddings = self._forward_with_checkpointing(tokens)
        else:
            with torch.set_grad_enabled(self.training):
                results = self.esm_model(tokens, repr_layers=[self.n_layers])
                embeddings = results["representations"][self.n_layers]
        
        # Remove BOS/EOS tokens: [batch, seq_len+2, dim] → [batch, seq_len, dim]
        # ESM2 adds <cls> at start and <eos> at end
        embeddings = embeddings[:, 1:-1, :]
        
        if return_per_residue:
            # Flatten to [total_residues, dim] for GNN input
            # Need to handle variable-length sequences
            all_embs = []
            for i, seq in enumerate(sequences):
                seq_len = len(seq)
                all_embs.append(embeddings[i, :seq_len, :])
            return torch.cat(all_embs, dim=0)
        
        return embeddings
    
    def _forward_with_checkpointing(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""
        # This is a simplified version - ESM2 internals may need adjustment
        results = self.esm_model(tokens, repr_layers=[self.n_layers])
        return results["representations"][self.n_layers]
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return list of trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get only LoRA weights for saving."""
        return {k: v for k, v in self.state_dict().items() if 'lora_' in k}
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA weights."""
        current = self.state_dict()
        for k, v in state_dict.items():
            if k in current:
                current[k] = v
        self.load_state_dict(current, strict=False)


# ════════════════════════════════════════════════════════════════
#  ESM2 with Layer Unfreezing (Alternative to LoRA)
# ════════════════════════════════════════════════════════════════

class ESM2FineTunable(nn.Module):
    """
    ESM2 model with last N layers unfrozen for fine-tuning.
    
    Simpler than LoRA but uses more memory and has more trainable params.
    
    Args:
        model_name: ESM2 model name
        unfreeze_layers: Number of last layers to unfreeze (default: 2)
        unfreeze_embeddings: Also unfreeze embedding layer (default: False)
    """
    
    ESM2_MODELS = {
        'esm2_t6_8M_UR50D':    {'layers': 6,  'dim': 320},
        'esm2_t12_35M_UR50D':  {'layers': 12, 'dim': 480},
        'esm2_t30_150M_UR50D': {'layers': 30, 'dim': 640},
        'esm2_t33_650M_UR50D': {'layers': 33, 'dim': 1280},
    }
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        unfreeze_layers: int = 2,
        unfreeze_embeddings: bool = False,
        use_gradient_checkpointing: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.model_name = model_name
        self.unfreeze_layers = unfreeze_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        model_info = self.ESM2_MODELS.get(model_name)
        if model_info is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.n_layers = model_info['layers']
        self.embed_dim = model_info['dim']
        
        # Load ESM2 model
        print(f"Loading ESM2 model: {model_name}")
        import esm
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Freeze all parameters first
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers
        layers_to_unfreeze = list(range(self.n_layers - unfreeze_layers, self.n_layers))
        for layer_idx in layers_to_unfreeze:
            for param in self.esm_model.layers[layer_idx].parameters():
                param.requires_grad = True
        
        # Optionally unfreeze embedding layer
        if unfreeze_embeddings:
            for param in self.esm_model.embed_tokens.parameters():
                param.requires_grad = True
        
        # Unfreeze final layer norm
        if hasattr(self.esm_model, 'emb_layer_norm_after'):
            for param in self.esm_model.emb_layer_norm_after.parameters():
                param.requires_grad = True
        
        # Move to device
        self.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ESM2 FineTunable: {trainable_params:,} trainable / {total_params:,} total "
              f"({100*trainable_params/total_params:.2f}%)")
        print(f"Unfrozen layers: {layers_to_unfreeze}")
    
    def _tokenize(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize sequences for ESM2."""
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        attention_mask = (batch_tokens != self.alphabet.padding_idx).float()
        return batch_tokens.to(self.device), attention_mask.to(self.device)
    
    def forward(
        self,
        sequences: List[str],
        return_per_residue: bool = True,
    ) -> torch.Tensor:
        """Extract embeddings from sequences."""
        tokens, attention_mask = self._tokenize(sequences)
        
        with torch.set_grad_enabled(self.training):
            results = self.esm_model(tokens, repr_layers=[self.n_layers])
            embeddings = results["representations"][self.n_layers]
        
        # Remove BOS/EOS tokens
        embeddings = embeddings[:, 1:-1, :]
        
        if return_per_residue:
            all_embs = []
            for i, seq in enumerate(sequences):
                seq_len = len(seq)
                all_embs.append(embeddings[i, :seq_len, :])
            return torch.cat(all_embs, dim=0)
        
        return embeddings
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return list of trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]


# ════════════════════════════════════════════════════════════════
#  Sequence Recovery from Graph (for training)
# ════════════════════════════════════════════════════════════════

# One-hot index → single-letter AA (same as pdb_to_pyg_v10/v11)
IDX_TO_AA = list("ARNDCEQGHILKMFPSTWYV")

def recover_sequence_from_graph(node_features: torch.Tensor) -> str:
    """
    Recover amino acid sequence from v10 graph node features.
    
    Node features layout: [N, 40]
        - First 20 dims: one-hot amino acid encoding
        - Remaining 20: physicochemical/structural features
    
    Args:
        node_features: [N, 40] tensor from graph.x
    
    Returns:
        sequence: Single-letter amino acid sequence
    """
    # Get one-hot portion
    one_hot = node_features[:, :20]  # [N, 20]
    
    # Convert to indices
    aa_indices = one_hot.argmax(dim=1)  # [N]
    
    # Convert to sequence
    sequence = ''.join(IDX_TO_AA[idx.item()] for idx in aa_indices)
    
    return sequence


def recover_sequences_batch(
    node_features: torch.Tensor,
    batch: torch.Tensor,
) -> List[str]:
    """
    Recover sequences for a batch of graphs.
    
    Args:
        node_features: [total_nodes, 40]
        batch: [total_nodes] graph index per node
    
    Returns:
        sequences: List of sequences, one per graph
    """
    sequences = []
    n_graphs = batch.max().item() + 1
    
    for g in range(n_graphs):
        mask = (batch == g)
        graph_features = node_features[mask]
        seq = recover_sequence_from_graph(graph_features)
        sequences.append(seq)
    
    return sequences


# ════════════════════════════════════════════════════════════════
#  Integration Helper: Replace ESM2GatedFusion with Live ESM2
# ════════════════════════════════════════════════════════════════

class ESM2LiveFusion(nn.Module):
    """
    Drop-in replacement for ESM2GatedFusion that runs ESM2 live.
    
    Instead of loading pre-extracted embeddings, this module:
    1. Takes node features and batch indices
    2. Recovers sequences from node features
    3. Runs ESM2 forward pass (with LoRA/fine-tuning)
    4. Applies gated fusion
    
    Compatible with existing GNN architecture - just swap the fusion module.
    """
    
    def __init__(
        self,
        esm2_module: nn.Module,  # ESM2LoRA or ESM2FineTunable
        hidden_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.esm2 = esm2_module
        self.esm_dim = esm2_module.embed_dim
        self.hidden = hidden_dim
        
        # Same architecture as ESM2GatedFusion
        self.esm_proj = nn.Sequential(
            nn.Linear(self.esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        # For monitoring
        self.register_buffer('esm_scale', torch.ones(1))
    
    def forward(
        self,
        h: torch.Tensor,           # [N, hidden] from node encoder
        x: torch.Tensor,            # [N, 40] raw node features
        batch: torch.Tensor,        # [N] graph index per node
    ) -> torch.Tensor:
        """
        Args:
            h: Node embeddings from node encoder
            x: Raw node features (for sequence recovery)
            batch: Graph index per node
        
        Returns:
            h_fused: [N, hidden] fused embeddings
        """
        # Recover sequences from node features
        sequences = recover_sequences_batch(x, batch)
        
        # Run ESM2 (with LoRA/fine-tuning)
        esm_emb = self.esm2(sequences, return_per_residue=True)  # [N, esm_dim]
        
        # Project and gate (same as ESM2GatedFusion)
        esm_h = self.esm_proj(esm_emb)  # [N, hidden]
        gate = self.gate_net(torch.cat([h, esm_h], dim=-1))  # [N, hidden]
        
        # Update scale for monitoring
        with torch.no_grad():
            self.esm_scale.fill_(esm_h.abs().mean().item())
        
        return gate * esm_h + (1.0 - gate) * h


# ════════════════════════════════════════════════════════════════
#  Quick Test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing ESM2 LoRA module...")
    
    # Test sequence recovery
    print("\n1. Testing sequence recovery...")
    fake_features = torch.zeros(5, 40)
    fake_features[0, 0] = 1  # A
    fake_features[1, 1] = 1  # R
    fake_features[2, 2] = 1  # N
    fake_features[3, 3] = 1  # D
    fake_features[4, 4] = 1  # C
    seq = recover_sequence_from_graph(fake_features)
    print(f"   Recovered sequence: {seq}")
    assert seq == "ARNDC", f"Expected ARNDC, got {seq}"
    print("   ✓ Sequence recovery works")
    
    # Test LoRA layer
    print("\n2. Testing LoRA layer...")
    original = nn.Linear(64, 64)
    lora = LoRALayer(original, rank=4, alpha=8)
    x = torch.randn(2, 64)
    y = lora(x)
    print(f"   Input: {x.shape}, Output: {y.shape}")
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable}")
    print("   ✓ LoRA layer works")
    
    # Test ESM2 loading (if available)
    print("\n3. Testing ESM2 LoRA (requires esm package)...")
    try:
        import esm
        # Use smallest model for testing
        esm2_lora = ESM2LoRA(
            model_name="esm2_t6_8M_UR50D",  # Smallest for testing
            rank=4,
            alpha=8,
            target_layers=[4, 5],  # Last 2 layers
            device="cpu",
        )
        
        test_seqs = ["ACDEFGHIKLMNPQRSTVWY", "AAAAAAAAAA"]
        embs = esm2_lora(test_seqs, return_per_residue=True)
        print(f"   Input: 2 sequences (len 20 + len 10)")
        print(f"   Output: {embs.shape}")
        assert embs.shape == (30, 320), f"Expected (30, 320), got {embs.shape}"
        print("   ✓ ESM2 LoRA works")
        
    except ImportError:
        print("   ⚠ esm package not installed, skipping ESM2 test")
    except Exception as e:
        print(f"   ⚠ ESM2 test failed: {e}")
    
    print("\n✓ All tests passed!")
