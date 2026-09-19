# Cross-Center Parkinson’s Disease Spectrum Detection from Tear Proteomics Using Knowledge-Guided Domain Adaptation and LoRA

A research implementation for **Parkinson’s disease (PD) spectrum detection from tear-fluid proteomics** using biological prior knowledge, cross-center domain-adversarial representation learning, and parameter-efficient **Low-Rank Adaptation (LoRA)**.

The framework is designed to learn disease-relevant molecular representations while reducing acquisition-center dependence and preserving external transportability across independent cohorts.

---

## Overview

The proposed pipeline combines:

- **Tear-fluid proteomics** with 1,011 protein-level features
- **PrimeKG-guided biological prior information**
- A **knowledge-gated shared molecular encoder**
- **Domain-adversarial learning** across the CRUCES and DONOSTIA acquisition centers
- **Low-Rank Adaptation (LoRA)** for parameter-efficient PD-spectrum refinement
- **Leakage-controlled preprocessing**
- **Locked external validation** on an independent cohort without retraining or threshold recalibration

The model is first trained on the PXD068184 development cohort and then evaluated on the independent **PXD028811** cohort.

---

## Framework Architecture

<p align="center">
  <img src="figures/framework_architecture.png" width="700" alt="Overall architecture of the proposed tear-proteomic Parkinson's disease spectrum framework">
</p>

The framework architecture integrates leakage-controlled preprocessing, PrimeKG-guided biological prior alignment, knowledge-gated representation learning, cross-center domain-adversarial learning, LoRA-based PD-spectrum adaptation, and independent external evaluation.

---

## End-to-End Experimental Pipeline

<p align="center">
  <img src="figures/experimental_pipeline.png" width="1200" alt="End-to-end experimental pipeline for tear-proteomic Parkinson's disease spectrum modeling">
</p>

The complete experimental workflow separates model development from external evaluation to reduce information leakage and provide a stricter assessment of model transportability.

---

## Datasets

### PXD068184 — Development Cohort

PXD068184 is used as the primary development cohort.

For the initial V2 representation-learning stage:

- **130 participants** were used for training
- **33 participants** were retained for internal validation
- CRUCES and DONOSTIA were treated as separate acquisition domains
- Genetic PD cases were withheld during the initial representation-learning stage

After V2 training, **7 LRRK2-associated PD cases** were added to the training branch for PD-spectrum LoRA adaptation.

The final LoRA training cohort contained:

- **70 controls**
- **67 PD-spectrum cases**
- **137 participants in total**

### PXD028811 — Independent External Cohort

The final model was evaluated on PXD028811 as an independent external cohort:

- **27 controls**
- **27 PD-spectrum cases**
- **54 participants in total**

No external retraining, normalization refitting, feature reselection, center adaptation, LoRA modification, or threshold optimization was performed.

---

## Methodology

### 1. Leakage-Controlled Preprocessing

Protein abundance values are transformed using a participant-wise percentile-rank procedure. Standardization parameters are estimated only from the original development-training subset and are reused unchanged for validation, genetic PD samples, and external inference.

### 2. PrimeKG-Guided Molecular Prior

The 1,011 proteomic features are aligned with PD-related biological information derived from **PrimeKG**. A compact disease-relevance prior is used to guide the contribution of individual proteins.

### 3. Knowledge-Gated Shared Encoder

The gated proteomic representation is compressed through a shared neural encoder:

```text
1011 -> 128 -> 64
