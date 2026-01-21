from torch.utils.data import Dataset
from pathlib import Path
from utils.partitioning import *
from utils.misc_utils import *
from einops import rearrange
import numpy as np
import random
import sys
sys.path.append('../')
from Utils.dataloading import load_data
from Utils.utils_flow import k2i_numpy, i2k_numpy
sys.path.append('../../')
from CMRx4DFlowMaskGeneration import *
from pathlib import Path
from typing import Iterable, List, Sequence, Union

REQUIRED_FILES = ("kdata_full.mat", "coilmap.mat", "segmask.mat", "params.csv")

PathLike = Union[str, Path]
# Specify here what subjects you want to use
DEFAULT_OPTS = {
    "train_roots": [
        "/mnt/nas/nas3/openData/rawdata/4dFlow/Aorta/Center003",
        "/mnt/nas/nas3/openData/rawdata/4dFlow/Aorta/Center007",
        "/mnt/nas/nas3/openData/rawdata/4dFlow/Aorta/Center012",
    ],
    "val_roots": [
        "/mnt/nas/nas3/openData/rawdata/4dFlow/Aorta/Center004",
        "/mnt/nas/nas3/openData/rawdata/4dFlow/Aorta/Center010",
        "/mnt/nas/nas3/openData/rawdata/4dFlow/Aorta/Center011",
    ],
    "test_roots": [
        "/mnt/nas/nas3/openData/rawdata/4dFlow/Aorta/Center008",
    ],
}
def find_valid_cases(
    roots: Sequence[PathLike],
    required_files: Iterable[str] = REQUIRED_FILES,
    *,
    anchor: str = "kdata_full.mat",
) -> List[str]:
    req = tuple(required_files)
    out: List[str] = []
    seen = set()

    for r in roots:
        root = Path(r)
        if not root.exists():
            continue

        for kpath in root.rglob(anchor):
            case_dir = kpath.parent
            if all((case_dir / f).is_file() for f in req):
                p = str(case_dir)
                if p not in seen:
                    seen.add(p)
                    out.append(p)

    out.sort()
    return out
class CMRx4DFlowDataSet(Dataset): 
	def __init__(self, **kwargs):
		options = DEFAULT_OPTS
		for key in kwargs.keys():
			options[key] = kwargs[key]
		self.options = options.copy() 
		self.usrate_list = [10, 20, 30, 40, 50]

		self.D_size = options['D_size']
		self.T_size = options['T_size']
		self.input = options['input'] 
		self.loss = options['loss']  

		self.filename = []
		mode = options["mode"] 

		roots = options.get(f"{mode}_roots", [])
		subjects = find_valid_cases(roots)
		self.filename = []
		for patient_dir in subjects:
			kdata = load_data(patient_dir + '/kdata_full.mat', 'kdata_full')
			Nx = kdata.shape[-1]
			slices = range(Nx-self.D_size+1)
			if options['mode'] == "test":
				slices = slices[::self.D_size]

			for i in slices:
				if options['mode'] == 'train':
					self.filename.append([patient_dir, i, None])
				elif options['mode'] == 'val':
					for u in self.usrate_list:
						self.filename.append([patient_dir, i, u])
				else:  # test 或其他
					self.filename.append([patient_dir, i, self.usrate_list[0]])

	def __len__(self):
		return len(self.filename)

	def get_order(self):
		return self.filename

	def __getitem__(self, idx):
		if self.options['mode'] != 'train':
			np.random.seed(0)

		f = load_data(self.filename[idx][0] + '/kdata_full.mat', 'kdata_full')
		c = load_data(self.filename[idx][0] + '/coilmap.mat', 'coilmap')
		s = load_data(self.filename[idx][0] + '/segmask.mat', 'segmask')

		Nv, Nt, Nc, SPE ,PE, FE = f.shape

		if self.options['network'] == "FlowVN":  # segment-wise reconstruction
			seg_idx = np.random.randint(Nv) if self.options['mode'] == 'train' else 0  # need to run inference for segment 0, 1, 2, and 3
		if self.T_size == -1:
			cardiac_bins = range(Nt)
		else:
			first_bin = random.randint(-self.T_size+1, Nt-self.T_size)  # any cardiac bin can be in the center (allows wrapping)
			cardiac_bins = list(range(first_bin, first_bin+self.T_size)) if self.options['mode'] == 'train' else list(range(Nt))
		cardiac_bins = np.mod(cardiac_bins,Nt)
		order = np.argsort(cardiac_bins)
		tmp = f[seg_idx:seg_idx+1, cardiac_bins[order]]
		inv = np.empty_like(order)
		inv[order] = np.arange(order.size)
		f = tmp[:, inv]
		f = f['real'] + 1j * f['imag']
		f = f.astype('complex64')
		f = k2i_numpy(f, ax=[-1])[..., self.filename[idx][1]:self.filename[idx][1]+self.D_size]

		c = c[..., self.filename[idx][1]:self.filename[idx][1]+self.D_size]
		c = c['real'] + 1j * c['imag']
		c = c.astype('complex64')

		s = s[self.filename[idx][1]:self.filename[idx][1]+self.D_size]
		Nv, Nt, Nc, SPE, PE, FE = f.shape

		im = np.sum(k2i_numpy(f, ax=[-2, -3]) * np.conj(c), axis=-4)
		if self.options['mode'] == 'train':
			usrate = random.choice(self.usrate_list)
		else:
			usrate = self.filename[idx][2] 
		total_points = PE * SPE // usrate
		mask_size = (PE, SPE)
		center_radius_x = 0.5
		center_radius_y = 0.5
		sigma_x = PE / 5
		sigma_y = SPE / 5
		masks_spe_pe_t = fun_mask_gen_2d(
			mask_size=mask_size,
			center_radius_x=center_radius_x,
			center_radius_y=center_radius_y,
			total_points=total_points,
			pattern_num=Nt,
			sigma_x=sigma_x,
			sigma_y=sigma_y,
			min_dist_factor=3,
			rep_decay_factor=0.5,
		)

		f = rearrange(f, 'nv nt nc spe pe fe -> nv nc nt fe pe spe')
		c = rearrange(c, 'nc spe pe fe -> nc fe pe spe')
		s = rearrange(s, 'spe pe fe -> fe pe spe')
		mask = rearrange(masks_spe_pe_t, 'spe pe t -> 1 1 t 1 pe spe')
		f *= mask
		im = rearrange(im, 'nv nt spe pe fe -> nv nt fe pe spe')

		# V, C, T, FE, PE, SPE
		# C, FE, PE, SPE
		# FE, PE, SPE
		# V, 1, T, 1, PE, SPE

		if self.loss == 'ssdu':
			if self.options['mode'] == 'train':
				mask_p1, mask_p2 = uniform_disjoint_selection(mask, rho=0.2, r2=9, seed=None, venc_coherence=True)
			elif self.options['mode'] == 'val':
				mask_p1, mask_p2 = uniform_disjoint_selection(mask, rho=0.2, r2=9, seed=444, venc_coherence=True)
			else:
				mask_p1, mask_p2 = mask, np.zeros_like(mask)  # all the data is in the input, so none left for loss

			# partition data
			kdata_p1 = f * mask_p1[:, np.newaxis, :, np.newaxis] 
			kdata_p2 = f * mask_p2[:, np.newaxis, :, np.newaxis] 

			# compute initial image
			imdata_p1 = mriAdjointOp(kdata_p1, c[np.newaxis, :, np.newaxis, :, :, :], mask_p1[:, np.newaxis, :, np.newaxis]).astype(np.complex64)

			# normalize data
			norm = np.linalg.norm(f) / np.linalg.norm(f != 0)
			imdata_p1 /= norm
			kdata_p1 /= norm
			kdata_p2 /= norm

			return {'imdata_p1': imdata_p1, 'kdata_p2': kdata_p2, 'kdata_p1': kdata_p1, 'coil_sens': c, 'norm': norm, "segmentation": s.astype(bool), "subj": str(self.filename[idx][0]).split('/')[-1]}

		elif self.loss == 'supervised':  
			# compute initial image
			imdata_p1 = mriAdjointOp(f, c[np.newaxis, :, np.newaxis, :, :, :], mask).astype(np.complex64)

			# normalize data
			norm = np.linalg.norm(f) / np.linalg.norm(f != 0)
			imdata_p1 /= norm
			im /= norm
			f /= norm
			return {'imdata_p1': imdata_p1, 'gt': im, 'kdata_p1': f, 'coil_sens': c, 'norm': norm, "segmentation": s.astype(bool), "subj": str(self.filename[idx][0]).split('/')[-1]}
		
		else:
			raise ValueError("loss must be either 'ssdu' or 'supervised'")
