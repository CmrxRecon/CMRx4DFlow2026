
## Repository Structure

The demos are split into two folders:

- **`./ForRecon/`** — everything a participant needs to:
  1) load data,  
  2) simulate undersampling,  
  3) reconstruct,  
  4) save results in a **submission-ready format**.  

- **`./ForEvaluation/`** — shows how the submissions evaluated:
  - applies **background phase correction** to the submitted reconstructions,
  - computes metrics by comparing against the **fully sampled ground truth**,
  - includes batch reconstruction and batch evaluation scripts.

---

## Contents

### `ForRecon/` (Participant Workflow)

#### 0) DataLoading Demo
Demonstrates how to load 4D Flow MRI data:
- read raw **multi-coil k-space** and metadata,
- coil combination using **coil sensitivity maps**,
- compute velocity maps from complex 4D Flow data.

#### 1) Undersampling Demo
Demonstrates the undersampling simulation workflow:
- generate an undersampling mask (using provided mask-generation utilities),
- apply the mask to fully sampled multi-coil k-space to simulate accelerated acquisition,
- perform a **zero-filled** reconstruction (IFFT + coil combination).

#### 2.1) Compressed Sensing Reconstruction (TV / LLR)
Demonstrates compressed sensing reconstruction from undersampled multi-coil k-space:
- set up the reconstruction problem,
- run an iterative solver with **total variation (TV)** and **locally low-rank (LLR)** regularization,
- output complex 4D Flow images for downstream velocity analysis.

#### 2.2) Flow Variational Network (FlowVN) Reconstruction
Introduces the Flow Variational Network (FlowVN) pipeline, including both training and inference for 4D Flow MRI reconstruction from undersampled k-space.

This implementation is adapted and modified from the original FlowMRI-Net codebase:
https://gitlab.ethz.ch/ibt-cmr/publications/flowmri_net

#### 3) DataSaving Demo
Shows how to export reconstructions to the **submission-ready format** expected by the evaluation pipeline.

---

### `ForEvaluation/` (Evaluation Workflow)

#### 4) PostProcessing Demo (Background Phase Correction)
Demonstrates evaluation-specific post-processing:
- apply **background phase correction** to reconstructed phase/velocity,
- ensures phase offsets do not bias velocity measurements.

> The evaluation pipeline applies background phase correction **before** computing metrics.

#### 5) Evaluation Demo
Describes the evaluation procedure:
- validation workflow,
- metrics computation using corrected reconstructions vs. fully sampled reference.

#### Batch Scripts
`ForEvaluation/` also includes utilities for bulk processing:

- **`BatchRecon_FlowVN.py`**  
  Batch reconstruction using FlowVN.

- **`BatchRecon_CS.py`**  
  Batch reconstruction using CS (TV/LLR).

- **`BatchEval.py`**  
  Batch evaluation: applies background phase correction and computes metrics for a set of reconstructed results.

---