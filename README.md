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

#### Configuration Specifications

##### 1. Configuration A ($C_1$ – ViraExplorer Strategy)
* **Optimization Goal**: Maximizes class-imbalance resilience and multi-branch feature learning.
* **Loss Function**: Focal Loss ($\alpha = 0.75, \gamma = 2.0$) with Label Smoothing ($\epsilon = 0.05$).
* **Optimizer & Schedule**: AdamW ($lr_{\text{backbone}} = 1\times 10^{-4}$, $lr_{\text{transformer}} = 5\times 10^{-5}$, weight decay $= 1\times 10^{-4}$) with a 5-epoch linear warmup followed by Cosine Annealing.
* **Data Augmentation**: Stochastic Reverse Complement ($p=0.5$) and random positional sequence shifts within $\pm 5\text{ bp}$.
* **Training Protocol**: 120 epochs maximum with Early Stopping (patience = 15).

##### 2. Configuration B ($C_2$ – FullE Ensemble Strategy)
* **Optimization Goal**: Standard multi-branch CNN training scheme.
* **Loss Function**: Binary Cross-Entropy (BCE) Loss.
* **Optimizer & Schedule**: Standard Adam ($lr = 1\times 10^{-3}$) with StepLR decay ($\gamma = 0.5$ every 5 epochs).
* **Data Augmentation**: None (clean baseline sequence input).
* **Training Protocol**: Fixed 30 epochs without early stopping.

##### 3. Configuration C ($C_3$ – Nucleic Transformer Strategy)
* **Optimization Goal**: Attention-focused optimization with token-level noise regularization.
* **Loss Function**: Cross-Entropy (CE) Loss.
* **Optimizer & Schedule**: Adam optimizer with Noam (AIAYN) inverse-square-root learning rate schedule and warmup steps.
* **Data Augmentation**: Online random nucleotide mutations ($n = 40$ mutated positions per sequence).
* **Training Protocol**: Fixed 60 epochs without early stopping.

---

### Benchmark Datasets & Tasks

Models trained under each configuration are evaluated across a diverse suite of biological tasks:

* **ViraMiner (VM) Metagenomic Benchmark**:
  * **Task**: Identification of viral contigs from 300 bp NGS human biospecimen reads.
  * **Challenge**: Severe real-world class imbalance ($\approx 2\%$ viral positive vs $98\%$ host/bacterial background).

* **Bacterial Promoter Recognition ($E.\ coli$)**:
  * **Task**: Binary classification of 81 bp promoter sequences centered around Transcription Start Sites (TSS), benchmarked on RegulonDB and extended datasets (DNABERT2-CAMP).

* **Human Enhancer Sequence Benchmark**:
  * **Task**: Functional annotation of 200 bp enhancer fragments across human cell lines (non-redundant set filtered via CD-HIT).

* **Out-of-Distribution (OOD) Anellovirus Benchmark**:
  * **Task**: Evaluation of model generalization when sequences from the *Anelloviridae* family are completely excluded from training/validation sets and evaluated exclusively during inference.

* **Synthetic Noise Robustness (SIM)**:
  * **Task**: Assessment of error tolerance by injecting uniform substitution noise at 1%, 5%, and 10% mutation levels into test sequences across 5 independent random seeds per corruption level (15 test sets total).
