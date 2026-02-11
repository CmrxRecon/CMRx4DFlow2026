## 0. ChallengeData Directory Structure
```
ChallengeData/
├─ TaskR1R2/
│  ├─ TrainSet/
│  │  └─ Aorta/
│  │     └─ Center012/
│  │        └─ Philips_30T_Ingenia/
│  │           └─ P005/
│  │              ├─ kdata_full.mat
│  │              ├─ coilmap.mat
│  │              ├─ segmask.mat
│  │              └─ params.csv
│  └─ ValidationSet/
│     └─ Aorta/
│        └─ Center012/
│           └─ Philips_30T_Ingenia/
│              └─ P006/
│                 ├─ kdata_ktGaussian10.mat
│                 ├─ usmask_ktGaussian10.mat
│                 ├─ ...
│                 ├─ coilmap.mat
│                 ├─ segmask.mat
│                 └─ params.csv
```
## 1. Data Description

**Dimension definitions:**
- **FE**: number of frequency-encoding samples
- **PE**: number of phase-encoding samples
- **SPE**: number of slice phase-encoding samples
- **Nc**: number of coils
- **Nt**: number of cardiac phases (time frames)
- **Nv**: number of velocity encodings

### Files

#### `kdata_full.mat`
Fully sampled k-space data.
- **Key**: `kdata_full`
- **Shape**: `(Nv, Nt, Nc, SPE, PE, FE)`

#### `coilmap.mat`
Coil sensitivity maps.
- **Key**: `coilmap`
- **Shape**: `(Nc, SPE, PE, FE)`

#### `segmask.mat`
Segmentation mask defining the ROI.
- **Key**: `segmask`
- **Shape**: `(SPE, PE, FE)`

#### `params.csv`
Acquisition and encoding parameters (missing values are left blank in the CSV):
- **FOV**: field of view (mm) for **FE**, **PE**, **SPE**
- **RR**: cardiac cycle duration (ms)
- **TR**: repetition time (ms)
- **TE**: echo time (ms)
- **FA**: flip angle (degrees)
- **field_strength**: magnetic field strength (T)
- **resolution**: resolution (mm) for **FE**, **PE**, **SPE**
- **spatial_order**: anatomical directions for **FE**, **PE**, **SPE**
  - Example: `[HF, AP, RL]` means FE is Head→Foot, PE is Anterior→Posterior, SPE is Right→Left
- **VENC_order**: velocity-encoding directions for the **Nv** dimension
  - Example: `[HF, AP, RL]` means encoding 1 is Head→Foot, 2 is Anterior→Posterior, 3 is Right→Left
- **VENC**: VENC values listed in the order of `VENC_order`

#### `usmask_ktGaussian{R}.mat`
Undersampling mask with acceleration factor **R**.
- **Key**: `usmask_ktGaussian`
- **Shape**: `(1, Nt, 1, SPE, PE, 1)`

#### `kdata_ktGaussian{R}.mat`
Undersampled k-space data with acceleration factor **R**.
- **Key**: `kdata_ktGaussian`
- **Shape**: `(Nv, Nt, Nc, SPE, PE, FE)`

#### `img_ktGaussian{R}.npz`
Reconstructed image generated **inside `segmask`** (from undersampled k-space with acceleration factor **R**). Stored as a **sparse NPZ**.
- **Key**: `img_ktGaussian`
- **Shape**: `(Nv, Nt, SPE, PE, FE)`

**Storage format (sparse in `.npz`)**  
The image is saved in NPZ using a sparse representation to reduce file size. For a complete, working example (including detailed function call instructions), please refer to:

- [`CMRx4DFlowReconDemo.ForRecon.3_DataSavingDemo`](https://github.com/CmrxRecon/CMRx4DFlow2026/blob/main/CMRx4DFlowReconDemo/ForRecon/3_DataSavingDemo.ipynb)
