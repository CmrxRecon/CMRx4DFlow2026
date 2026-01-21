import os 
import time 
import torch
import einops
import argparse
import numpy as np
from CS_LLR_utils import *


def CS_LLR(lamb_tv, lamb_llr, input_dir, input, output_dir, vol):
    """
    LLR and spatiotemporal TV are implementated here (for the FlowMRI_net paper lamb_tv=0)

    You could reconstruct all segments (velocity-encodings) at once by removing the seg loop and indexing
    This will be faster but requires more GPU memory
    """

    for seg in [0,1,2,3]: 
        ksp = np.load(input_dir+vol+'/'+input, mmap_mode='c')[[seg], :, :, 0, :, :, :].astype(np.complex64)  # (V, C, T, D, H, W) complex64
        ksp = np.fft.ifftshift(np.fft.fft(np.fft.ifftshift(ksp, axes=3), axis=3, norm="ortho"), axes=3)  
        retro_prosp = False  # retrospectively undersampling of prospectively-undersampled data (for brain)
        if retro_prosp:
            m = np.load(input_dir + vol + '/retro_mask_R24.npy')[[seg], :, :, :]
            ksp *= m[:, np.newaxis, :, np.newaxis, :, :] 
                                
        coils = np.load(input_dir+vol+'/coilmap_full.npy', mmap_mode='c')[:, np.newaxis, :, :, :].astype(np.complex64)  # (C, 1, D, H, W) complex64
        mask = ksp[:,0:1,:,0:1,:,:] !=  0 #(V, 1, T, 1, H, W) bool

        V, C, T, X, Y, Z = ksp.shape
        norm_k = np.median(np.abs(ksp[:,:,:, X//2, Y//2, Z//2])) / 10
        ksp = ksp / norm_k

        with torch.no_grad():
            ksp = torch.from_numpy(ksp).to(torch.complex64)
            ksp = k2i(ksp, -3)
            coils = torch.from_numpy(coils).to(torch.complex64)
            mask = torch.from_numpy(mask).to(torch.float32)
            mask = (ksp.abs().sum(dim=(1, -3), keepdims=True) > 0).float()

        # fixed CS-LLR hyperparameters
        lamb_tv_deps = 0.005            # epsilon to prevent zero gradient and smooth the optimization
        lamb_tv_w = (1, 0.1, 0.1, 0.1)  # weightings for gradients over (t, x, y, z)
        bsz = 16                        # LLR block/patch size

        vtv_reg_multi_v_f_lowmem = get_vtv_reg_multi_v_f_lowmem(V, T, lamb_tv, lamb_tv_deps, lamb_tv_w)
                
        with torch.no_grad():
            dtype = torch.complex64
            dev = 'cuda'
            coils = coils.to(dev).to(dtype)
            kspc = ksp.to(dev).to(dtype)
            
            kspc = einops.rearrange(kspc, 'V C T X Y Z -> C (V T) X Y Z')
            mask = (einops.reduce(kspc.abs(), 'C VT X Y Z -> 1 VT 1 Y Z', 'sum') > 0.).float()
            mask = mask.to(dev)
            
            acq_op = return_cartesian_acq_operator(mask, kspc, dims=[-1, -2], coils=coils, force_fftn_device=None)

            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
            start_max_memory = torch.cuda.max_memory_allocated()
            tm0 = time.time()
            rec, itinfo = pgd_recon(kspc, acq_op, lamb_llr, alpha=0.3, tau=0.97, 
                                                    bsz=bsz, debug=True, maxiters=70, 
                                                    custom_regularizer_fn=vtv_reg_multi_v_f_lowmem,
                                                    force_eig_on_cpu=True)
            torch.cuda.synchronize()
            print(f'Elapsed {time.time() - tm0}')
            end_max_memory = torch.cuda.max_memory_allocated()
            
            rec = rec.cpu()
            rec = einops.rearrange(rec, '(V T) X Y Z -> V T X Y Z', V=V, T=T)
            kspc = acq_op = None
            
            rec = rec * norm_k

            print('Mem: ', (end_max_memory - start_max_memory)/1024/1024, end_max_memory/1024/1024, start_max_memory/1024/1024)

        # save reconstruction
        np.save(output_dir + 'rec{}.npy'.format(seg), rec)

    pass
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS-LLR arguments')

    parser.add_argument('--lambda_llr', type=float, default=0.0, help='weighting of LLR regularization')
    parser.add_argument('--lambda_tv',  type=float, default=0.0, help='weighting of TV regularization')
    parser.add_argument('--input_dir',	type=str,   default='',  help='directory of the input data')
    parser.add_argument('--input',	    type=str,   default='',  help='name of network input file (e.g. rawdata_full_1_R16.npy)')
    parser.add_argument('--output_dir',	type=str,   default='',  help='directory of the output data')
    parser.add_argument('--vol',	    type=str,   default='',  help='volunteer to reconstruct')  
    
    args = parser.parse_args()
    args = vars(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms = True

    CS_LLR(lamb_tv=args["lambda_tv"], lamb_llr=args["lambda_llr"], input_dir=args["input_dir"], input=args["input"], output_dir=args["output_dir"], vol=args["vol"])
