# Leveraging Architectural Diversity for DNA Sequence Classification


This repository implements a multi-domain benchmark to evaluate how diverse deep learning paradigms generalize across heterogeneous genomic sequence classification tasks directly from raw nucleotide sequences without manual feature extraction.
The repository provides a dual execution framework: a decoupled workflow (strictly separating training/checkpointing from standalone inference) alongside monolithic end-to-end pipelines, ensuring complete experimental flexibility and post-hoc logit analysis across biological benchmarks.

---
## Research Objective

While domain-specific models demonstrate strong performance in specialized genomic tasks, there remains a critical need for a systematic comparative analysis evaluating the trade-offs between **convolutional local motif extraction** and **attention-based global contextual modeling**.

The primary objective of this project is to benchmark pure CNNs, multi-branch hybrid architectures, and Transformer models across:

* **Diverse Sequence Lengths & Biological Contexts**: Ranging from short 81 bp regulatory elements to 300 bp metagenomic fragments.
* **Out-of-Distribution (OOD) Generalization**: Assessing model performance when encountering completely unseen viral taxonomic families.
* **Sequencing Error Robustness**: Measuring accuracy degradation under synthetic mutation/corruption noise levels (1%, 5%, 10%).

---

## Evaluated Neural Architectures

We benchmark four distinct architectural paradigms, exploring the trade-offs between local motif discovery, nucleotide frequency modeling, and attention mechanisms:

| Paradigm | Architecture | Core Structural Philosophy |
| :--- | :--- | :--- |
| **Hybrid Multi-Branch** | **GenExplorer** | Combines multi-scale 1D convolutions (`PatternBranch` with max pooling and `FrequencyBranch` with average pooling) with a parallel 4-layer `TransformerBranch` to simultaneously capture local motifs, base frequencies, and global positional context via late MLP fusion. |
| **Multi-Branch CNN Ensemble** | **FullE** | An ensemble of 1D CNN subnetworks featuring multi-scale convolutions and parameterized $L_p$ power-average pooling—which dynamically interpolates between mean pooling ($p=1$) and max pooling ($p \to \infty$). Submodels and merger networks are combined via a 10-model Sum-Rule decision fusion. |
| **Convolutional Transformer** | **Nucleic Transformer** | Formulates sequence classification using 1D convolutional $k$-mer tokenization (kernels $k \in \{2..6\}$) to generate learnable $k$-mer tokens, followed by a 6-layer Transformer Encoder stack with global mean pooling. |
|**Genomic Foundation Model** | **DNABERT2-CAMP** | Leverages the pre-trained DNABERT-2 (117M) transformer backbone fine-tuned alongside a CAMP (Convolutional Attention Motif Profiler) module—a 4-layer 1D CNN cascade coupled with Multi-Head Self-Attention for localized motif refinement. |
---

## Experimental Strategy: Configuration Cross-Benchmarking

To evaluate both **native optimization** and **cross-paradigm adaptability**, each architecture is systematically trained and evaluated under **three distinct configuration strategies ($C_1, C_2, C_3$)**.

Each configuration reflects the native loss, optimization, learning rate schedule, and data augmentation strategy of one of the baseline models:

```text
                 ┌───────────────────────────────────────────────┐
                 │           Evaluation Architectures            │
                 │   [ ViraExplorer | FullE | Nucleic-Trans. ]   │
                 └───────────────────────┬───────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌──────────────┐                  ┌──────────────┐                  ┌──────────────┐
│ Config A (C1)│                  │ Config B (C2)│                  │ Config C (C3)│
│ ViraExplorer │                  │    FullE     │                  │  Nucleic-Tr. │
│    Native    │                  │    Native    │                  │    Native    │
└──────────────┘                  └──────────────┘                  └──────────────┘
````

### Configuration Specifications

### 1. Configuration A ($C_1$ – ViraExplorer Strategy)

* **Optimization Goal**: Maximizes class-imbalance resilience and multi-branch feature learning.
* **Loss Function**: Focal Loss ($\alpha = 0.75, \gamma = 2.0$) with Label Smoothing ($\epsilon = 0.05$).
* **Optimizer & Schedule**: AdamW ($lr_{\text{backbone}} = 1\times 10^{-4}$, $lr_{\text{transformer}} = 5\times 10^{-5}$, weight decay $= 1\times 10^{-4}$) with a 5-epoch linear warmup followed by Cosine Annealing.
* **Data Augmentation**: Stochastic Reverse Complement ($p=0.5$) and random positional sequence shifts within $\pm 5\text{ bp}$.
* **Training Protocol**: 120 epochs maximum with Early Stopping (patience = 15).

---

### 2. Configuration B ($C_2$ – FullE Ensemble Strategy)

* **Optimization Goal**: Standard multi-branch CNN training scheme.
* **Loss Function**: Binary Cross-Entropy (BCE) Loss.
* **Optimizer & Schedule**: Standard Adam ($lr = 1\times 10^{-3}$) with StepLR decay ($\gamma = 0.5$ every 5 epochs).
* **Data Augmentation**: None (clean baseline sequence input).
* **Training Protocol**: Fixed 30 epochs without early stopping.

---

### 3. Configuration C ($C_3$ – Nucleic Transformer Strategy)

* **Optimization Goal**: Attention-focused optimization with token-level noise regularization.
* **Loss Function**: Cross-Entropy (CE) Loss.
* **Optimizer & Schedule**: Adam optimizer with Noam (AIAYN) inverse-square-root learning rate schedule and warmup steps.
* **Data Augmentation**: Stochastic mutation augmentation ($K = \min(40, \lfloor 0.13 \cdot L \rfloor)$ nucleotide substitutions per sequence).
* **Training Protocol**: Fixed 60 epochs without early stopping.

---

## Benchmark Datasets & Tasks

Models trained under each configuration are evaluated across 5 biological benchmarks:

### DS-VM (Viral Metagenomic Benchmark)

* **Task**: Identification of viral contigs from 300 bp NGS human biospecimen reads under severe class imbalance ($\approx 2\%$ viral positive vs $98\%$ host/bacterial background). 

### DS-ECO (Bacterial Promoters - E. coli)

* **Task**: Binary classification of 81 bp promoter sequences centered around Transcription Start Sites (TSS), benchmarked on RegulonDB and extended datasets (DNABERT2-CAMP).

### DS-ENH (Human Enhancers)

* **Task**: Functional annotation of 200 bp enhancer fragments across human cell lines.

### DS-OOD (Out-of-Distribution Anellovirus)

* **Task**: Sequences from the Anelloviridae family are excluded from training/validation and tested exclusively at inference to measure out-of-distribution taxonomic transfer.

### DS-SIM (Synthetic Noise Robustness)

* **Task**: Assessment of error tolerance by injecting uniform substitution noise at 1%, 5%, and 10% mutation levels into test sequences across 5 independent random seeds per corruption level.

---

## Dual Execution Framework

The repository supports two workflows depending on experimental and computational requirements:

```bash
                        ┌───────────────────────────────┐
                        │       EXECUTION OPTIONS       │
                        └───────────────┬───────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌─────────────────────────────┐                       ┌─────────────────────────────┐
│ 1. DECOUPLED WORKFLOW       │                       │ 2. ALL-IN-ONE PIPELINES     │
│ pipelines/train_<arch>.py   │                       │ pipelines/train_*_pipe.py   │
└────────────┬────────────────┘                       └────────────┬────────────────┘
             │ Saves Checkpoint (.pt)                              │
             ▼                                                     │
┌─────────────────────────────┐                                    │
│ evaluation/evaluate.py      │                                    │
└────────────┬────────────────┘                                    │
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        ▼ Exports Raw Logits (.txt)
                             ┌─────────────────────┐
                             │    global_logits/   │
                             └──────────┬──────────┘
                                        ▼ Input for Benchmark Synthesis
                             ┌─────────────────────┐
                             │ compute_metrics.py  │
                             └─────────────────────┘
```
### Option 1: Decoupled Workflow (Modular)
Isolates training from inference. Models save only validated checkpoints in checkpoints/, while evaluate.py independently computes test predictions:
```bash
# 1. Train GenExplorer
python pipelines/train_genexplorer.py --config A --task ecoli --data_folder ./datasets --checkpoint_dir ./checkpoints

# 2. Standalone Inference
python evaluation/evaluate.py --arch genexplorer --task ecoli --config A --checkpoint ./checkpoints/best_genexplorer_ecoli_A.pt --data_folder ./datasets --logits_dir ./global_logits

```

### Option 2:All-in-One End-to-End Pipelines
Executes training, validation, checkpoint selection, and automated cascade evaluation across all benchmark test sets in a single script run:
```bash
# End-to-End GenExplorer Execution
python pipelines/train_genex_pipeline.py --config A --task ecoli --data_folder ./datasets

# End-to-End Nucleic Transformer Execution
python pipelines/train_nt_pipeline.py --config A --task ecoli --data_folder ./datasets

# End-to-End DNABERT2-CAMP Execution
python pipelines/train_dnabert2_camp_pipeline.py --config A --task ecoli --data_folder ./datasets

# End-to-End FullE Ensemble Execution
python pipelines/train_fulle_pipeline.py --config A --task ecoli --data_folder ./datasets

```
---

## Cross-Configuration Evaluation

Each architecture can be executed using any of the three optimization strategies described above.

Consequently, the benchmark supports two complementary analyses:

* **Intra-architecture comparison**: Evaluates how a single architecture behaves under different optimization strategies.
* **Cross-architecture comparison**: Evaluates different architectures trained under the same optimization strategy.

This design enables a fair assessment of both architectural inductive biases and optimization strategies.

---

## Global Benchmark Synthesis

Both workflows export raw logits to
```
global_logits/
```
using the unified naming convention:
```
{architecture}_cfg{CONFIG}_{task}_{evaluation_dataset}.txt
```
directory using a standardized organization based on:

* **Architecture**

  * `genexplorer`
  * `fulle`
  * `nucleictransformer`
  * `dnabert2camp`

* **Training Configuration**

  * `cfgA`
  * `cfgB`
  * `cfgC`

* **Training Task**

  * `viral`
  * `ecoli`
  * `enhancer`
  * `anellovirus`

* **Evaluation Dataset**

  * `test_VM`
  * `test_ecoli`
  * `test_enhancer`
  * `test_anellovirus`
  * `test_sim_noise_5_seed_5002`

Saving raw logits rather than only final metrics allows:

* Reproducible post-hoc analyses.
* Computation of new evaluation metrics without rerunning inference.
* ROC and Precision–Recall curve generation.
* Confidence interval estimation via bootstrapping.
* Decision-level ensemble experiments.
* Statistical significance testing between models.

The analysis scripts operate directly on the stored logits, making experimentation significantly faster and fully reproducible.

---

## Determinism & Reproducibility
Global random seed is fixed to 2020 across all data splitters, tokenizers, and PyTorch backends to guarantee reproducible replication of all experimental results.

