# V13 Architecture: Deep Learning Breakdown
## Protein Function Prediction using Hierarchical Chain-Aware GNN

---

## 1. Problem Formulation

**Task:** Multi-label classification on graph-structured data.

Given a protein structure graph G = (V, E) and per-node embeddings from a pretrained language model, predict a binary vector for each of three GO ontologies:

```
y_MF  ∈ {0,1}^489    (Molecular Function — 489 classes)
y_BP  ∈ {0,1}^1943   (Biological Process — 1943 classes)
y_CC  ∈ {0,1}^320    (Cellular Component — 320 classes)
```

This is multi-label (not multi-class) — a protein can have many GO terms simultaneously. Output activation is **sigmoid**, not softmax.

---

## 2. Input Representation

### 2.1 Protein Structure Graph

Each protein is represented as an undirected graph:

- **Nodes (V):** One node per amino acid residue
- **Edges (E):** Two residues are connected if their Cα atoms are within 8Å in 3D space (spatial contact graph)
- **Node features (x):** 20-dimensional one-hot amino acid type + biochemical properties
- **Edge features (e):** Distance between residues + geometric angles (direction cosines)

### 2.2 ESM2 Per-Residue Embeddings

- **Source:** ESM2 (esm2_t33_650M_UR50D) — 650M parameter protein language model trained on 250M sequences
- **Output:** 1280-dimensional vector per residue, encoding evolutionary context
- **Usage:** Precomputed offline, loaded as node-level features at training time
- **Key property:** Captures which residues are evolutionarily conserved, which substitutions are tolerated — information invisible to structural graphs alone

### 2.3 Chain Identity Embedding

- Each residue carries a chain ID (A, B, C...) for multi-chain protein complexes
- Learned embedding: `nn.Embedding(max_chains, 8)` → 8-dimensional chain identity vector
- Concatenated to node features before message passing

---

## 3. Layer-by-Layer Architecture

### Stage 1: Encoding

#### Node Encoder
```
Input:  x ∈ R^(N × 20)        (one-hot amino acid features)
Linear(20 → 192) + BatchNorm + ReLU + Dropout(0.1)
Output: h ∈ R^(N × 192)
```

#### Edge Encoder
```
Input:  e ∈ R^(E × 6)         (distance + geometric angles)
Linear(6 → 64) + ReLU → Linear(64 → 32)
Output: e' ∈ R^(E × 32)
```

#### ESM2 Gated Fusion
This is a key original contribution. Rather than blindly concatenating ESM2 features, the model learns how much to trust them per residue:

```
Input:  h_struct ∈ R^(N × 192)     (structural features)
        h_esm2  ∈ R^(N × 1280)     (ESM2 embeddings)

Projection: h_esm2_proj = Linear(1280 → 192)(h_esm2)

Gate:   g = sigmoid(Linear(192+192 → 192)([h_struct || h_esm2_proj]))
        g ∈ R^(N × 192), values in (0,1)

Fusion: h = g ⊙ h_esm2_proj + (1-g) ⊙ h_struct
        h ∈ R^(N × 192)
```

**Deep learning concept:** This is a learned soft selection between two feature sources. g=1 means "trust ESM2 completely", g=0 means "trust structure completely". The model learns this per residue during training via backpropagation.

---

### Stage 2: Message Passing (4× Dual-Conv Block)

Each block runs two GNN operations in parallel and combines them:

#### GCN Branch (Graph Convolutional Network)
```
h'_i = W · MEAN({h_j : j ∈ N(i) ∪ {i}})
```
- Cheap, symmetric aggregation
- Captures global neighborhood topology
- Every neighbor contributes equally — no discrimination

**Deep learning concept:** This is the graph equivalent of a standard convolution. Neighborhood averaging with learned weights.

#### GATv2 Branch (Graph Attention Network v2, 6 heads)
```
α_ij = softmax_j( a^T · LeakyReLU(W·[h_i || h_j]) )
h'_i = ||_{k=1}^{6} σ( Σ_j α_ij^k · W^k · h_j )
```
- Each head learns different attention patterns
- α_ij is the importance weight of neighbor j for node i
- 6 heads capture 6 different "views" of the neighborhood
- Outputs concatenated across heads, then projected back to 192-d

**Deep learning concept:** Multi-head self-attention applied to graphs. The model learns which structural contacts are functionally important — not all residue contacts matter equally for predicting function.

#### Block Combination
```
h_block = GCN(h) + GATv2(h)          (element-wise addition)
h_out   = LayerNorm(h_block + h)      (skip connection)
```

**Deep learning concept:** Skip connections (residual connections) prevent vanishing gradients across 4 stacked blocks. LayerNorm stabilizes training.

This block is applied **4 times sequentially**. Each application increases the receptive field — after 4 blocks, each node has seen information from residues up to 4 hops away in the graph.

---

### Stage 3: Key Innovation — Hierarchical Chain-Aware Pooling

This is where V13 fundamentally differs from all prior work. The goal is to go from node-level representations h ∈ R^(N × 192) to a single protein-level vector.

**Prior work (DeepFRI, GOBoost, GGN-GO):** GlobalMeanPool over all residues → one vector. Information about chain structure is destroyed.

**V13:** Three-stage hierarchy.

#### 3a. Residue-Level Triple Pooling (within each chain)
For each chain c containing residues {h_1, ..., h_k}:

```
h_gated = Σ_i α_i · h_i              (attention pool: α = softmax(Linear(h)))
h_mean  = (1/k) Σ_i h_i              (mean pool)
h_max   = MAX_i(h_i)                  (max pool)

h_triple = [h_gated || h_mean || h_max]   (concatenate → 576-d)
h_chain_input = Linear(576 → 192)(h_triple)
```

**Why three pooling operations?**
- Attention pool: focuses on the most important residues
- Mean pool: captures the average character of the chain
- Max pool: captures the strongest signal for each feature dimension
- Together they give complementary views of the same chain

#### 3b. Chain Attention Pool (residues → chain vector)
```
For each chain c:
  h_chain_c = Σ_i α_i · h_i
  where α_i = softmax(v^T · tanh(W · h_i))
  h_chain_c ∈ R^192
```
This produces one 192-d vector per chain.

#### 3c. Protein Chain Pool (chains → protein vector)
```
h_protein = Σ_c β_c · h_chain_c
where β_c = softmax(v^T · tanh(W · h_chain_c))
h_protein ∈ R^192
```
This produces one 192-d vector for the entire protein complex.

**Deep learning concept:** This is hierarchical attention pooling — the same idea as hierarchical attention networks in NLP (word → sentence → document), applied to protein biology (residue → chain → complex).

#### 3d. CC Cross-Attention Branch
Cellular Component prediction depends on where in the cell a protein is located — which is determined by exposed surface chains, membrane-spanning regions, etc.

```
Query:   h_protein ∈ R^192            (protein-level vector)
Keys:    H_chains ∈ R^(C × 192)       (all chain vectors)
Values:  H_chains ∈ R^(C × 192)

h_cc = MultiHeadAttention(Q=h_protein, K=H_chains, V=H_chains)
     = softmax(QK^T / √d) · V
```

**Deep learning concept:** This is the standard transformer cross-attention mechanism. The protein-level query attends over chain-level keys to extract localization-relevant context. Used only for CC prediction, not MF or BP.

#### 3e. Pool Fusion
```
h_final = LayerNorm([h_triple_global || h_protein || h_cc])
        ∈ R^(192×3 = 576)
```
All pooling paths concatenated and normalized.

---

### Stage 4: Prediction Heads

Three separate MLPs — one per ontology:

```
MF Head: Linear(576 → 256) → ReLU → Dropout(0.3) → Linear(256 → 489)  → sigmoid
BP Head: Linear(576 → 256) → ReLU → Dropout(0.3) → Linear(256 → 1943) → sigmoid
CC Head: Linear(576 → 256) → ReLU → Dropout(0.3) → Linear(256 → 320)  → sigmoid
```

Output: probability for each GO term independently. Threshold applied at test time to convert to binary predictions.

**Why separate heads?** MF, BP, CC are different biological concepts. Sharing parameters would force the model to learn a single representation serving three different tasks — empirically worse than specialization.

---

## 4. Loss Functions

### 4.1 Class-Balanced Asymmetric Loss (primary loss)

Standard Binary Cross-Entropy treats every example equally. This is catastrophic for protein function prediction — some GO terms appear in fewer than 0.1% of proteins. A model that never predicts rare terms gets a great BCE score but terrible Fmax.

**Asymmetric Loss** (Ridnik et al., ICCV 2021):

```
For positive labels (y=1):
  L_pos = -(1-p)^γ_pos · log(p)          (focal weight on hard positives)

For negative labels (y=0):
  p_clipped = max(p - clip, 0)            (ignore very easy negatives)
  L_neg = -(p_clipped)^γ_neg · log(1-p_clipped)

Total: L_ASL = Σ L_pos + Σ L_neg
```

**Parameters in V13:**
- `γ_pos = 0` (no focal weighting on positives — this is the bug we identified)
- `γ_neg = 4` (aggressive down-weighting of easy negatives)
- `clip = 0.05`

**What each parameter does:**
- Higher `γ_pos` → model penalized more for missing rare positive labels → better recall
- Higher `γ_neg` → model ignores easy negatives (very common non-labels) → focuses on hard examples
- `clip` → probability margin below which negatives are completely ignored

**Known issue identified in this project:** `γ_pos = 0` caused threshold collapse on DeepFRI splits. Increasing to `γ_pos = 2.0` is the recommended fix.

### 4.2 GO-DAG Hierarchical Consistency Loss

GO terms form a Directed Acyclic Graph — if a protein has GO:0004674 (protein serine/threonine kinase activity), it MUST also have GO:0004672 (protein kinase activity), its parent.

Without this loss, the model can predict P(child) = 0.9 and P(parent) = 0.1 — biologically impossible.

```
For each parent-child edge (child → parent) in GO DAG:
  violation = max(0, sigmoid(z_child) - sigmoid(z_parent) + margin)

L_hier = weight × Σ_edges violation

Default: weight=0.05, margin=0.0
```

**Deep learning concept:** This is a constraint-based loss — adding domain knowledge as a penalty term. The model is still free to violate the constraint, but pays a cost proportional to the degree of violation.

### 4.3 Co-occurrence Regularization

GO terms that frequently co-occur in the same proteins should have correlated predictions. Label embeddings are initialized from SVD of the co-occurrence matrix and regularized to stay semantically meaningful.

```
L_cooc = ||W_labels - W_cooc_svd||^2_F    (Frobenius norm)
```

### 4.4 Total Training Loss

```
L_total = L_ASL(MF) + L_ASL(BP) + L_ASL(CC)    (per-ontology asymmetric loss)
        + 0.05 × L_hier                          (hierarchy consistency)
        + λ × L_cooc                             (co-occurrence regularization)
```

---

## 5. Evaluation Metric: Fmax

Standard accuracy is meaningless for multi-label classification with class imbalance. Fmax is the standard metric in protein function prediction (used in CAFA challenges).

```
For threshold t ∈ [0, 1]:
  precision(t) = TP(t) / (TP(t) + FP(t))
  recall(t)    = TP(t) / (TP(t) + FN(t))
  F1(t)        = 2 · precision(t) · recall(t) / (precision(t) + recall(t))

Fmax = max_{t} F1(t)
```

This finds the threshold that achieves the best precision-recall tradeoff across the entire test set. A model that predicts everything as positive gets recall=1 but precision≈0. A model that predicts nothing gets precision undefined and recall=0. Fmax rewards models that make confident, correct predictions.

**Per-class Fmax** (what V13 reports): optimize threshold separately per GO term, then average. Better than global threshold for imbalanced classes.

---

## 6. Post-Processing: Ancestor Propagation

After predicting probabilities, enforce the GO hierarchy at test time:

```
For each GO term t, from leaves to roots:
  P_final(t) = max(P_raw(t), max_{c: child of t} P_raw(c))
```

If the model predicts P(child) = 0.8 but P(parent) = 0.3, propagation corrects parent to 0.8. This enforces biological validity without changing training.

**Finding from this project:** Propagation consistently hurt performance (raw > propagated Fmax across all versions). Reason: hierarchy loss during training already made predictions mostly consistent, so propagation over-corrected calibrated probabilities.

---

## 7. Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Optimizer | AdamW | Weight decay decoupled from gradient — better generalization |
| Learning rate | 5e-4 | OneCycleLR with warmup — avoids local minima early |
| Scheduler | OneCycleLR | Cosine annealing with 10% warmup — standard for GNNs |
| Batch size | 16 | Constrained by 8GB VRAM |
| Gradient accumulation | 1 | Effective batch = 16 |
| AMP | Enabled | Mixed precision — halves VRAM usage, 2× faster |
| Gradient checkpointing | Enabled | Trades compute for memory on large graphs |
| Early stopping patience | 20 epochs | Stops when validation Fmax plateaus |
| Total parameters | 2,276,359 | ~2.3M |

---

## 8. Key Deep Learning Concepts Summary

| Concept | Where Used | Purpose |
|---------|-----------|---------|
| Graph Neural Network | Message passing stage | Learn from non-Euclidean protein structure |
| Multi-head attention | GATv2 + Cross-attention | Learn which residues/chains matter |
| Gated fusion | ESM2 integration | Soft selection between feature sources |
| Residual connections | All GNN blocks | Prevent vanishing gradients |
| Hierarchical pooling | Chain-aware pooling | Preserve biological organization |
| Transfer learning | ESM2 embeddings | Leverage 650M param pretrained model |
| Focal loss variant | Asymmetric loss | Handle extreme class imbalance |
| Constraint loss | Hierarchy consistency | Encode domain knowledge |
| Multi-task learning | Three separate heads | Specialize per ontology |

---

## 9. What Makes This Architecture Novel

Every published method (DeepFRI, HEAL, GGN-GO, GOBoost) processes individual protein chains. The key novelty of this work:

1. **PDB-level multi-chain graphs** — Full protein complex with inter-chain edges encoding 3D proximity between chains. Biologically, protein function often emerges from chain-chain interactions, not individual chains.

2. **Hierarchical chain-aware pooling** — Residue → Chain → Protein pooling mirrors biological organization. No prior work pools at the chain level before protein-level aggregation.

3. **Chain-specific CC prediction** — Dedicated cross-attention for Cellular Component using chain-level context, motivated by the observation that localization signals live at the chain level (membrane-spanning chains, signal peptides).

4. **ESM2 gated fusion** — Rather than fixed concatenation, learned per-residue gating between evolutionary and structural features.

---

## 10. Identified Limitations and Future Work

| Limitation | Root Cause | Solution |
|-----------|-----------|---------|
| Generalization failure on DeepFRI splits | Frozen ESM2 = family-specific representations | ESM2 fine-tuning via LoRA |
| Threshold collapse on unseen families | γ_pos=0 → output calibration loss | Increase γ_pos to 2.0 |
| CC gap vs GOBoost (0.343 vs 0.745) | PDB-level aggregation dilutes CC signal | Chain-level processing |
| Cannot match GGN-GO/GOBoost | No AF2 data, no contrastive learning | AlphaFold2 augmentation + contrastive loss |
| Hardware constraint | ESM2 fine-tuning needs >8GB VRAM | Cloud GPU / smaller ESM2 variant |