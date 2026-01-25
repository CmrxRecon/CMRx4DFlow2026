# CMRx4DFlowReconDemo

This demo contains several Jupyter notebooks (`*.ipynb`) organized into the following parts:

Demo data can be downloaded from the following page : [Demo Data Downloading](TODO)

### 0) DataLoading Demo

This demo focuses on data loading for 4D Flow MRI. It shows how to read the raw multi-coil k-space and metadata, perform basic coil combination with coil sensitivity maps, and compute velocity maps from the complex 4D Flow data.

### 1) Undersampling Demo

This demo introduces the undersampling workflow for 4D Flow MRI.
It shows how to generate an undersampling mask (via the provided mask-generation functions), how to apply the mask to fully-sampled multi-coil k-space to simulate accelerated acquisition, and how to reconstruct a zero-filled image by inverse FFT followed by coil combination with sensitivity maps.

### 2.1) Compressed Sensing Reconstruction
This demo shows how to reconstruct undersampled 4D Flow MRI using total variation (TV) and locally low-rank (LLR) compressed sensing.
It walks through setting up the reconstruction problem for undersampled multi-coil k-space and running the iterative solver to recover complex 4D Flow images for subsequent velocity analysis.

### 2.2) Flow Variational Network (FlowVN) Reconstruction (WIP)
This demo introduces the Flow Variational Network (FlowVN) pipeline.
It explains how to train FlowVN on 4D Flow MRI data (data preparation, network configuration, and training loop), and how to run FlowVN inference to reconstruct images from undersampled k-space, producing reconstructions that can be used directly for downstream flow quantification.

### 3) DataSaving Demo
This demo shows how to save reconstructed images into the submission-ready format. 

### 4) PostProcessing Demo
This demo focuses on post-processing steps specific to 4D Flow MRI, in particular background phase correction.
Because phase offsets can bias velocity measurements, the evaluation pipeline applies background phase correction to uploaded reconstructions first, and then compares the corrected results against the fully-sampled image when computing metrics.

### 5) Evaluation Demo
This demo describes the evaluation procedure, including the metrics used and the validation workflow.
