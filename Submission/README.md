## Submission Guidelines

Please keep the original directory structure and save your results to the corresponding subfolders as specified.

Name your zip file as `Submission.zip`.

## Directory Structure

For each case, use the following path:

`{Task}/{Anatomy}/ValidationSet/{Center}/{Vendor}/PXXX/img_ktGaussian{R}.mat`

## `.mat` Format

Each `img_ktGaussian{R}.mat` file must contain a single variable with:

- **Key**: `img_ktGaussian`
- **Array shape**: `(Nv, Nt, Nc, SPE, PE, FE)`

To avoid potential `.mat` formatting/compatibility issues, we strongly recommend saving reconstructed images with `CMRx4DFlowReconDemo.Utils.dataloading.save_mat` provided in this repository.