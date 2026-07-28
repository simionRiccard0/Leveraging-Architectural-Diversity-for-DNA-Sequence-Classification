## Methodological Framework & Experimental Setup

This repository implements a multi-domain benchmark to evaluate how diverse deep learning paradigms generalize across heterogeneous genomic sequence classification tasks directly from raw nucleotide sequences without manual feature extraction.

---

### Research Objective

While domain-specific models demonstrate strong performance in specialized genomic tasks, there remains a critical need for a systematic comparative analysis evaluating the trade-offs between **convolutional local motif extraction** and **attention-based global contextual modeling**.

The primary objective of this project is to benchmark pure CNNs, multi-branch hybrid architectures, and Transformer models across:
* **Diverse Sequence Lengths & Biological Contexts**: Ranging from short 81 bp regulatory elements to 300 bp metagenomic fragments.
* **Out-of-Distribution (OOD) Generalization**: Assessing model performance when encountering completely unseen viral taxonomic families.
* **Sequencing Error Robustness**: Measuring accuracy degradation under synthetic mutation/corruption noise levels (1%, 5%, 10%).

---

### Evaluated Neural Architectures

We benchmark three distinct architectural paradigms, representing different trade-offs in motif locality, base frequency modeling, and attention mechanisms:

| Paradigm | Architecture | Core Structural Philosophy |
| :--- | :--- | :--- |
| **Hybrid Multi-Branch** | **ViraExplorer** | Combines multi-scale 1D convolutions (`PatternBranch` with max pooling and `FrequencyBranch` with average pooling) with a parallel 4-layer `TransformerBranch` to simultaneously capture local motifs, base frequencies, and global positional context via late MLP fusion. |
| **Multi-Branch CNN Ensemble** | **FullE** | An ensemble of 1D CNN subnetworks featuring multi-scale convolutions and parameterized $L_p$ power-average pooling—which dynamically interpolates between mean pooling ($p=1$) and max pooling ($p \to \infty$). Submodels and merger networks are combined via a 10-model Sum-Rule decision fusion. |
| **Convolutional Transformer** | **Nucleic Transformer** | Formulates sequence classification using 1D convolutional $k$-mer tokenization (kernels $k \in \{2..6\}$) to generate learnable $k$-mer tokens, followed by a 6-layer Transformer Encoder stack with global mean pooling. |

---

### Experimental Strategy: Configuration Cross-Benchmarking

To evaluate both **native optimization** and **cross-paradigm adaptability**, each architecture is systematically trained and evaluated under **three distinct configuration strategies ($C_1, C_2, C_3$)**. Each configuration reflects the native loss, optimization, learning rate schedule, and data augmentation strategy of one of the baseline models:

```text
                  ┌───────────────────────────────────────────────┐
                  │           Evaluation Architectures           │
                  │   [ ViraExplorer | FullE | Nucleic-Trans. ]   │
                  └───────────────────────┬───────────────────────┘
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       ▼                                  ▼                                  ▼
┌──────────────┐                   ┌──────────────┐                   ┌──────────────┐
│ Config A (C1)│                   │ Config B (C2)│                   │ Config C (C3)│
│ ViraExplorer │                   │    FullE     │                   │  Nucleic-Tr. │
│    Native    │                   │    Native    │                   │    Native    │
└──────────────┘                   └──────────────┘                   └──────────────┘
