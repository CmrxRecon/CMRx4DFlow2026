Thoracic aorta (FlowMRI-Net)|Cerebrovascular arteries (FlowMRI-Net)
--|--
<img src="./gifs/aorta.gif" height="250" />|<img src="./gifs/ICA.gif" height="250" />


# FlowMRI-Net: A Generalizable Self-Supervised Physics-Driven 4D Flow MRI Reconstruction Network

by Luuk Jacobs, Marco Piccirelli, Valery Vishnevskiy, and Sebastian Kozerke

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

## About
The implementation code, data, and pre-trained weights for the work in <https://doi.org/10.1016/j.jocmr.2025.101913> 

## Prerequisites
Required packages can be installed via conda:
```
conda env create -f environment.yml
```

## Data availability
Aortic and cerebrovascular 4D flow MRI data can be freely downloaded from <https://doi.org/10.3929/ethz-b-000705347>. What data are used for training or testing is specified using the root_dir and input arguments in `scripts/train.sh` and `scripts/test.sh`. Which volunteers are used for training, validation, and testing is specified in `utils/dataloader.py`.

## Pre-trained weights
The pre-trained weights in `weights` can be used for transfer learning or inference purposes by specifying the desired checkpoint path (specifying aorta or brain, undersampling factor, and FlowMRI-Net or FlowVN) as ckpt_path argument in `scripts/train.sh` or `scripts/test.sh`, respectively. 

## Citation 
If you use the code and/or data for your work, please cite the following:
```
@article{jacobs2025flowmri,
  title={FlowMRI-Net: A Generalizable Self-Supervised 4D Flow MRI Reconstruction Network},
  author={Jacobs, Luuk and Piccirelli, Marco and Vishnevskiy, Valery and Kozerke, Sebastian},
  journal={Journal of Cardiovascular Magnetic Resonance},
  pages={101913},
  year={2025},
  publisher={Elsevier}
}
```

```
@misc{jacobs_flowmri-net_2025,
	title = {FlowMRI-Net dataset, for aortic and cerebrovascular 4D flow MRI},
	url = {https://www.research-collection.ethz.ch/handle/20.500.11850/705347},
	doi = {10.3929/ethz-b-000705347},
	publisher = {ETH Zurich},
	author = {Jacobs, Luuk and Piccirelli, Marco and Vishnevskiy, Valery and Kozerke, Sebastian},
	year = {2025},
}
```

## Correspondence
For any questions, please contact [Luuk Jacobs](mailto:ljacobs@ethz.ch)
