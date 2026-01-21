from torch.utils.data import Dataset
from pathlib import Path
from utils.partitioning import *
from utils.misc_utils import *
import numpy as np
import random

DEFAULT_OPTS_BRAIN = {'train_subjects':['vol001', 'vol002', 'vol003', 'vol004', 'vol005', 'vol006', 'vol007', 'vol008', 'vol009'],
					  'test_subjects':['vol010']}

DEFAULT_OPTS_AORTA = {'train_subjects':['vol001', 'vol002', 'vol003', 'vol004', 'vol005', 'vol006', 'vol007', 'vol008', 'vol009'],
					  'val_subjects': ['vol010'],
					  'test_subjects':['vol011', 'vol012', 'vol013', 'vol014', 'vol015']}

# Specify here what subjects you want to use
DEFAULT_OPTS = DEFAULT_OPTS_BRAIN

class OwnDataset(Dataset): 
	def __init__(self, **kwargs):
		options = DEFAULT_OPTS
		for key in kwargs.keys():
			options[key] = kwargs[key]
		self.options = options.copy() 

		self.D_size = options['D_size']
		self.T_size = options['T_size']
		self.input = options['input'] 
		self.loss = options['loss']  
		self.is_brain = 'brain' in options['root_dir']

		self.filename = []
		for patient in options[options['mode']+'_subjects']:
			patient_dir = Path(self.options['root_dir']) / str(patient)
			file_path = patient_dir / self.input 
			Nx = np.load(file_path, mmap_mode='c').shape[-3]
			slices = range(Nx-self.D_size+1)  # you always include the D_size in front
			# you can speed up the validation by selecting subset of slices
			#if options['mode'] == "val":
			#	slices = [70]
			if options['mode'] == "test":  
				slices = slices[::self.D_size]
			for i in slices:
				self.filename.append([patient_dir, i])

	def __len__(self):
		return len(self.filename)

	def get_order(self):
		return self.filename

	def __getitem__(self, idx):
		if self.options['mode'] != 'train':
			np.random.seed(0)
	
		total_bins = np.load(self.filename[idx][0] / self.input, mmap_mode='c').shape[2]
		if self.T_size == -1:
			cardiac_bins = range(total_bins)
		else:
			first_bin = random.randint(-self.T_size+1, total_bins-self.T_size)  # any cardiac bin can be in the center (allows wrapping)
			cardiac_bins = list(range(first_bin, first_bin+self.T_size)) if self.options['mode'] == 'train' else list(range(total_bins))
		# V, C, T, FE, PE, SPE
		# C, FE, PE, SPE
		# FE, PE, SPE
		# V, 1, T, 1, PE, SPE
		f = np.load(self.filename[idx][0] / self.input, mmap_mode='c')[:, :, cardiac_bins, 0, self.filename[idx][1]:self.filename[idx][1]+self.D_size, :, :].astype(np.complex64)
		c = np.load(self.filename[idx][0] / 'coilmap_full.npy', mmap_mode='c')[:, self.filename[idx][1]:self.filename[idx][1]+self.D_size, :, :].astype(np.complex64)
		s = np.load(self.filename[idx][0] / 'segmentation_full.npy', mmap_mode='c')[self.filename[idx][1]:self.filename[idx][1]+self.D_size, :, :] if self.is_brain else np.array([0])
			
		retro_us = False  # retrospectively undersample the prospectively-undersampled brain scans
		if retro_us:
			m = np.load(self.filename[idx][0] / 'retro_mask_R24.npy')
			f *= m[:, np.newaxis, cardiac_bins, np.newaxis, :, :] 

		mask = (abs(f[:,0,:,0,:,:]) != 0).astype(np.float32)

		if self.options['network'] == "FlowVN":  # segment-wise reconstruction
			seg_idx = np.random.randint(f.shape[0]) if self.options['mode'] == 'train' else 0  # need to run inference for segment 0, 1, 2, and 3
			f = f[[seg_idx]]
			mask = mask[[seg_idx]]
		
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

			return {'imdata_p1': imdata_p1, 'kdata_p2': kdata_p2, 'kdata_p1': kdata_p1, 'coil_sens': c, 'norm': norm, "segmentation": s.astype(bool), "subj": str(self.filename[idx][0].stem).split('/')[-1]}

		elif self.loss == 'supervised':  
			# compute initial image
			imdata_p1 = mriAdjointOp(f, c[np.newaxis, :, np.newaxis, :, :, :], mask[:, np.newaxis, :, np.newaxis]).astype(np.complex64)

			# load ground truth
			im = np.load(str(self.filename[idx][0] / 'gt_imdata.npy'), mmap_mode='c')[seg_idx,cardiac_bins,self.filename[idx][1]:self.filename[idx][1]+self.D_size][np.newaxis, np.newaxis] 
			
			# normalize data
			norm = np.linalg.norm(f) / np.linalg.norm(f != 0)
			imdata_p1 /= norm
			im /= norm
			f /= norm

			return {'imdata_p1': imdata_p1, 'gt': im, 'kdata_p1': f, 'coil_sens': c, 'norm': norm, "segmentation": s.astype(bool), "subj": str(self.filename[idx][0].stem).split('/')[-1]}
		
		else:
			raise ValueError("loss must be either 'ssdu' or 'supervised'")
