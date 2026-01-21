import numpy as np
import h5py
from glob import glob
import SimpleITK as sitk
from skimage import filters
from misc_utils import *
from unwrapFlow import unwrap_data
from execute_MSAC import execute_MSAC

save_h5 = False
f_out = './recon.h5'
venc = 100  # aorta: venc=100 cm/s, brain: venc=150 cm/

# load network recon
recon = './results/exp/vol013_1990-epoch=42'
Nslices = max([int(xx.split('.')[-2].split('slice')[-1]) for xx in glob(recon+"/*", recursive = True)])+1
for i in range(Nslices):
    if i == 0:
        Nv, Nt, Dsize, Ny, Nz = np.load(recon+'/slice0.npy', mmap_mode='c').shape
        vol = np.zeros((Nv, Nt, Nslices*Dsize, Ny, Nz), dtype=np.complex64)  # Nv, Nt, Nx, Ny, Nz  
    vol[:, :, i*Dsize:(i+1)*Dsize, :, :] = np.load(recon+'/slice{}.npy'.format(i)) 

# alternatively load LLR recon or ground truth
#vol = np.load('./philips_aorta/vol013/gt_imdata.npy')

# compute bias field for averaged magniude and apply to all segments and cardiac bins
mag_avg = np.mean(np.sqrt(np.sum(np.square(np.abs(vol)), axis=0)), axis=0)
raw_img_sitk = sitk.GetImageFromArray(mag_avg)
transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)
fg_mask = sitk.GetImageFromArray((mag_avg > filters.threshold_multiotsu(mag_avg, classes=5)[0]).astype(np.uint8))
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(raw_img_sitk, fg_mask)
log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
vol = abs(vol) / sitk.GetArrayFromImage(sitk.Exp(log_bias_field))[np.newaxis, np.newaxis] * np.exp(1j*np.angle(vol))

# divide the flow segments
data = np.transpose(vol, (2, 3, 4, 0, 1))[:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:,np.newaxis]
data = np.conjugate(data[:, :, :, :, :, :, :, :, :, 1:, :, :]) * np.exp(1j * np.angle(data[:, :, :, :, :, :, :, :, :, 0:1, :, :]))

# Background phase correction
data = execute_MSAC(np.squeeze(data).transpose(0,1,2,4,3), corr_fit_order=3) 

# Unwrap aliased voxels
data = np.squeeze(data)
corrected_angles = np.zeros_like(data, dtype=np.float32)
for i in range(data.shape[3]):  # unfold each velocity encoding separately
    phi_w = np.angle(data[:,:,:,i,:])  # x,y,z,t
    corrected_angles[:,:,:,i,:] = unwrap_data(phi_w, mode='lap4D', tfc=True) 
max_phase = np.max(np.abs(corrected_angles))  # new phases do not have to range between -pi and pi anymore
corrected_angles = corrected_angles/max_phase*np.pi  # rescale for complex data 
venc = venc*max_phase/np.pi  # increase the venc accordingly
data = abs(data) * np.exp(1j*corrected_angles)  
data = data[:,:,:,np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, np.newaxis]

# lower memory 
data = data.astype(np.complex64)

if save_h5:
    data = np.transpose(np.squeeze(data), (3, 4, 0, 1, 2))

    hf = h5py.File(f_out, 'w')

    head = hf.create_dataset('Header',(1,))
    head.attrs[u'Venc'] = venc

    hfdat = hf.create_group('data')
    hfdat.create_dataset('real', data=np.real(data)) 
    hfdat.create_dataset('imag', data=np.imag(data)) 

    # compute velocity fields from this via
    # np.angle(data)/np.pi*venc
