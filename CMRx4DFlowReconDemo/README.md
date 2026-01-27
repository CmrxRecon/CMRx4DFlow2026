
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

#### 2) Compressed Sensing Reconstruction (TV / LLR)
Demonstrates compressed sensing reconstruction from undersampled multi-coil k-space:
- set up the reconstruction problem,
- run an iterative solver with **total variation (TV)** and **locally low-rank (LLR)** regularization,
- output complex 4D Flow images for downstream velocity analysis.

#### 3) DataSaving Demo
Shows how to export reconstructions to the **submission-ready format** expected by the evaluation pipeline.


#### Ex) Flow Variational Network (FlowVN) Reconstruction
Introduces the Flow Variational Network (FlowVN) pipeline, including both training and inference for 4D Flow MRI reconstruction from undersampled k-space.

This implementation is adapted and modified from the original FlowMRI-Net codebase:
https://gitlab.ethz.ch/ibt-cmr/publications/flowmri_net


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

## BatchRecon_CS.py
Batch reconstruction using CS (TV/LLR).

### What it does
- Recursively scans `--path_recon` for case folders containing `kdata_ktGaussian{R}.mat` (and required side files).
- Runs CS-LLR reconstruction for each case and each `R`.
- Saves outputs to `--path_save` while preserving the same relative folder structure.

### Required files per case (for each R)
- `kdata_ktGaussian{R}.mat`, `usmask_ktGaussian{R}.mat`, `segmask.mat`, `coilmap.mat`

### Outputs (per case, per R) saved under
`<path_save>/<same relative path as in path_recon>/`
- `img_ktGaussian{R}.npz` (recon, masked by segmask)
- `recontime_ktGaussian{R}.csv` (single value: seconds)

### Example
``` bash
python BatchRecon_CS.py \
  --path_recon "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1&R2/ValidationSet/Aorta/Center012/Philips_15T_Ambition" \
  --path_save  "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_CS/TaskR1&R2/ValidationSet/Aorta/Center012/Philips_15T_Ambition" \
  --device cuda:0 \
  --Rs 10 20 \
  --lamb_llr 10=0.5 20=1.0
```

## BatchRecon_FlowVN.py
Batch reconstruction using FlowVN (test mode).

### What it does
- Recursively scans `--path_recon` for valid case folders for each `R`.
- Calls FlowVN `main.py --mode test` for each case.
- Saves FlowVN outputs to `--path_save` with the same relative folder structure.

### Required files per case (for each R)
- `kdata_ktGaussian{R}.mat`, `usmask_ktGaussian{R}.mat`, `segmask.mat`, `coilmap.mat`, `params.csv`

### Outputs
Saved under:
`<path_save>/<same relative path as in path_recon>/`
- FlowVN test outputs written by FlowVN into `--save_dir` (script sets `--save_dir` to this folder)

### Example
``` bash
python BatchRecon_FlowVN.py \
  --path_recon "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1&R2/ValidationSet/Aorta/Center012/Philips_15T_Ambition" \
  --path_save  "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_FlowVN/TaskR1&R2/ValidationSet/Aorta/Center012/Philips_15T_Ambition" \
  --cuda_visible_devices 0 \
  --Rs 10 20 \
  --ckpt_path "../FlowVN/weights/12-epoch=0.ckpt" \
  --network FlowVN --features_in 1 --T_size 5 --features_out 8 \
  --kernel_size 5 --num_stages 10 --loss supervised
```

## BatchEval.py
Batch evaluation for recon results (phase correction + metrics).

### What it does
- Recursively finds recon files: `**/img_ktGaussian*.npz` under `--path_recon`.
- For each case, loads GT from `--path_gt/<same relative path>/img_gt.npz` and `segmask.mat`.
- Applies background phase correction, then computes metrics:
  - magnitude: SSIM, nRMSE
  - flow: RelErr, AngErr (deg)
- Reads runtime from `recontime_ktGaussian{R}.csv` if present.

### Inputs
- Recon files:
  `<path_recon>/**/img_ktGaussian{R}.npz`
- GT files:
  `<path_gt>/<same rel_dir>/img_gt.npz` and `<path_gt>/<same rel_dir>/segmask.mat`

### Output
- Writes a single CSV table to `--out_csv` (one row per case per R)

### Example
```bash
python BatchEval.py \
  --path_recon "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_FlowVN/TaskR1&R2" \
  --path_gt   "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/TaskR1&R2" \
  --out_csv   "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_FlowVN/TaskR1&R2/results.csv"
```
---