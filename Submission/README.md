## Submission Guidelines

Please keep the original directory structure and save your results to the corresponding subfolders as specified.

Name your zip file as `Submission.zip`.

## Directory Structure

For each case, use the following path:

`{Task}/{ValidationSet or TestSet}/{Anatomy}/{Center}/{Vendor}/PXXX/img_ktGaussian{R}.npz`

The directory layout of a submission follows the reference package: `./Submission_demo.zip`.

## `.npz` Format
To avoid potential saving-format and compatibility issues, we **strongly recommend** exporting reconstructed images in **`.npz`** format using the utility function provided in this repository

This helps ensure consistent serialization and better cross-environment compatibility when loading results for downstream evaluation and visualization.

For a complete, working example (including detailed function call instructions), please refer to:

- [`CMRx4DFlowReconDemo.ForRecon.3_DataSavingDemo`](https://https://github.com/CmrxRecon/CMRx4DFlow2026/tree/main/CMRx4DFlowReconDemo/ForRecon/3_DataSavingDemo)
