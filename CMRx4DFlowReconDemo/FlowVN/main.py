import os
from pathlib import Path
import argparse
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils.misc_utils import *
# from utils.dataloader import OwnDataset
from utils.dataloader_CMRx4DFlow import CMRx4DFlowDataSet
import sys
import time
import csv
sys.path.append('../')
from Utils.utils_datasl import save_coo_npz
from networks.flowvn import FlowVN
class CMRSaveCallback(pl.Callback):
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}  # key -> {recon: {seg_idx: np}, seg: {seg_idx: np}, meta: dict, recon_ms_sum: float}

    def _key(self, meta: dict):
        subj = meta.get("subj", "unknown")
        R = int(meta.get("usrate", 0))
        return (subj, R)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        meta = {k: outputs.get(k) for k in outputs.keys() if k != "recon"}

        seg_idx = int(meta.get("seg_idx", 0))
        recon_ms = meta.get("recon_ms", 0.0)

        k = self._key(meta)
        if k not in self._cache:
            self._cache[k] = {"recon": {}, "seg": {}, "meta": meta, "recon_ms_sum": 0.0}

        if recon_ms is not None:
            self._cache[k]["recon_ms_sum"] += float(recon_ms)

        recon = outputs["recon"].detach().cpu()
        x = recon
        if x.ndim == 5:
            x = x[0]
        if x.ndim != 4:
            raise RuntimeError(f"Unexpected recon shape {tuple(recon.shape)} -> {tuple(x.shape)}")
        x_np = x.numpy()

        seg = batch["segmentation"]
        if hasattr(seg, "detach"):
            seg = seg.detach().cpu().numpy()
        seg = seg.astype(bool)[0]
        if seg.ndim != 3:
            raise RuntimeError(f"Unexpected segmentation shape {tuple(seg.shape)} (expected (fe,pe,spe))")

        self._cache[k]["recon"][seg_idx] = x_np
        self._cache[k]["seg"][seg_idx] = seg
        self._cache[k]["meta"] = meta

    def on_test_epoch_end(self, trainer, pl_module):
        for (subj, R), pack in self._cache.items():
            recon_map = pack["recon"]
            seg_map = pack["seg"]
            if len(recon_map) == 0:
                continue

            seg_ids = sorted(recon_map.keys())

            img = np.stack([recon_map[i] for i in seg_ids], axis=0)
            img = np.transpose(img, (0, 1, 4, 3, 2))

            pick = seg_ids[0]
            s = seg_map[pick]
            s = np.transpose(s, (2, 1, 0))
            s = s[None, None, :, :, :]

            save_coo_npz(str(self.save_dir / f"img_ktGaussian{R}.npz"), img * s)

            csv_path = self.save_dir / f"recontime_ktGaussian{R}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["recontime"])
                w.writerow([pack["recon_ms_sum"] / 1000.0])

        self._cache.clear()
class UnrolledNetwork(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        self.log_img_count = 0

        if self.options['network'] == "FlowVN":
            self.network = FlowVN(**self.options)      
        elif self.options['network'] == "FlowMRI_Net":
            self.network = FlowMRI_Net(**self.options)
        else:
            raise ValueError('Network does not exist')
        
        self.L1_loss = nn.L1Loss()

    def training_step(self, batch):
        if self.options['loss'] == 'ssdu':
            kdata_p2 = batch['kdata_p2']  # heldout disjoint partition for loss
            loss_mask = abs(kdata_p2[:, :, 0, :, 0, :, :]) != 0
        
            #with torch.autograd.graph.save_on_cpu(pin_memory=True):  # you can add cpu offloading to lower peak gpu memory
            recon_img_p1 = self.network(Variable(batch['imdata_p1'], requires_grad=True), batch['kdata_p1'], batch['coil_sens'])
            kdata_p1 = mri_forward_op(recon_img_p1, batch['coil_sens'], loss_mask.float())
            loss = 0.5 * torch.norm(torch.view_as_real(kdata_p2) - torch.view_as_real(kdata_p1), p=2) / torch.norm(torch.view_as_real(kdata_p2), p=2) + \
                   0.5 * torch.norm(torch.view_as_real(kdata_p2) - torch.view_as_real(kdata_p1), p=1) / torch.norm(torch.view_as_real(kdata_p2), p=1)
            
        elif self.options['loss'] == 'supervised':
            recon_img_p1 = self.network(Variable(batch['imdata_p1'], requires_grad=True), batch['kdata_p1'], batch['coil_sens'])

            if self.options['exp_loss']:
                tau = self.current_epoch/10
                w = torch.exp(torch.Tensor([-tau*(self.options['num_stages']-k+1) for k in range(self.options['num_stages'])]).to(self.device))
                w /= torch.sum(w)
                loss = torch.sum(w*torch.norm(recon_img_p1 - batch['gt'], p=1, dim=[1,2,3,4,5,6]))/40000
            else:
                loss = self.L1_loss(recon_img_p1 - batch['gt'][:, 0], torch.zeros_like(recon_img_p1))
            if (not torch.isfinite(loss)) or (loss.item() > 1e3):
                print("[LOSS SPIKE]", loss.item(),
                    batch["case_dir"], batch["slice_start"],
                    "seg", batch["seg_idx"], "usrate", batch["usrate"])
        # tensorboard logging
        self.log_dict({"train_loss_epoch": loss, "step": self.current_epoch*1.}, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.options['loss'] == 'ssdu':
            kdata_p2 = batch['kdata_p2']
            loss_mask = abs(kdata_p2[:, :, 0, :, 0, :, :]) != 0
            
            if self.options['T_size'] == -1 or self.options['network'] == "FlowVN":
                recon_img_p1 = self.network(batch['imdata_p1'], batch['kdata_p1'], batch['coil_sens'])
            else:
                recon_img_p1 = torch.zeros_like(batch['imdata_p1'])
                for t in range(-self.options['T_size']+1,recon_img_p1.shape[2]-self.options['T_size']+1):
                    cardiac_bins = list(range(t, t+self.options['T_size']))
                    recon_img_p1[:,:,t+self.options['T_size']//2] = self.network(batch['imdata_p1'][:,:,cardiac_bins], batch['kdata_p1'][:,:,:,cardiac_bins], batch['coil_sens'])[:,:,self.options['T_size']//2]

            kdata_p1 = mri_forward_op(recon_img_p1, batch['coil_sens'], loss_mask.float())  # NxCxTxDxHxW

            center_slice = int(self.options['D_size']//2)
            loss = 0.5 * torch.norm(torch.view_as_real(kdata_p2)[:,:,:,:,center_slice] - torch.view_as_real(kdata_p1)[:,:,:,:,center_slice], p=2) / torch.norm(torch.view_as_real(kdata_p2)[:,:,:,:,center_slice], p=2) + \
                   0.5 * torch.norm(torch.view_as_real(kdata_p2)[:,:,:,:,center_slice] - torch.view_as_real(kdata_p1)[:,:,:,:,center_slice], p=1) / torch.norm(torch.view_as_real(kdata_p2)[:,:,:,:,center_slice], p=1)

        elif self.options['loss'] == 'supervised':
            center_slice = int(self.options['D_size']//2)

            if self.options['T_size'] == -1 or self.options['network'] == "FlowVN":
                recon_img_p1 = self.network(batch['imdata_p1'], batch['kdata_p1'], batch['coil_sens'])
            else:
                recon_img_p1 = torch.zeros_like(batch['imdata_p1'])
                for t in range(-self.options['T_size']+1,recon_img_p1.shape[2]-self.options['T_size']+1):
                    cardiac_bins = list(range(t, t+self.options['T_size']))
                    recon_img_p1[:,:,t+self.options['T_size']//2] = self.network(batch['imdata_p1'][:,:,cardiac_bins], batch['kdata_p1'][:,:,:,cardiac_bins], batch['coil_sens'])[:,:,self.options['T_size']//2]
            
            loss = torch.norm(recon_img_p1[:,:,:,center_slice] - batch['gt'][:,:,:,center_slice], p=1)/torch.numel(recon_img_p1[:,:,:,center_slice])

        # self.log_dict({"val_loss_epoch": loss, "step": self.current_epoch*1.}, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if batch_idx == 0:  # only save first image
            recon_np = recon_img_p1[[0],0,0,center_slice].detach().cpu().numpy()  
            sample_img = 255 * np.abs(recon_np)/np.max(np.abs(recon_np))
            self.logger.experiment.add_image('sample_recon',np.rot90(sample_img.astype(np.uint8), k=-1, axes=(1, 2)).astype(np.uint8),self.log_img_count)
            self.log_img_count += 1
        return {'avg_val_loss': loss}
    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        recon_ms = None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        if self.options['T_size'] == -1 or self.options['network'] == "FlowVN":
            recon_img = self.network(batch['imdata_p1'], batch['kdata_p1'], batch['coil_sens'])
        else:
            window_t = self.options['T_size'] - 2
            n_bins = batch['imdata_p1'].shape[2]
            recon_img = torch.zeros_like(batch['imdata_p1'][:,:,0:1,:,:,:]).repeat(
                1, 1, int(np.ceil(n_bins/window_t))*window_t, 1, 1, 1
            )
            for t in range(0, n_bins, window_t):
                cardiac_bins = list(range(t-self.options['T_size']+1, t+1))
                recon_img[:,:,t:t+window_t] = self.network(
                    batch['imdata_p1'][:,:,cardiac_bins],
                    batch['kdata_p1'][:,:,:,cardiac_bins],
                    batch['coil_sens']
                )[:,:,1:-1]
            recon_img = torch.roll(recon_img[:,:,:n_bins], shifts=-window_t, dims=2)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            recon_ms = float(start_event.elapsed_time(end_event))

        recon_img_complex = (recon_img[0] * batch['norm']).detach()

        usrate = int(batch["usrate"][0]) if hasattr(batch["usrate"], "__len__") else int(batch["usrate"])
        seg_idx = int(batch["seg_idx"][0]) if hasattr(batch["seg_idx"], "__len__") else int(batch["seg_idx"])
        subj = batch["subj"][0] if isinstance(batch["subj"], (list, tuple)) else batch["subj"]

        return {
            "recon": recon_img_complex,
            "subj": subj,
            "seg_idx": seg_idx,
            "usrate": usrate,
            "recon_ms": recon_ms,
        }
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),lr=self.options['lr']) 
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.options['epoch'])}}

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network arguments')

    # Data IO
    parser.add_argument('--D_size',     type=int,  	default=1,	    			help='number of slices per volume (FlowVN only)')
    parser.add_argument('--T_size',     type=int,  	default=5,	    			help='number of cardiac bins per volume')
    parser.add_argument('--root_dir',   type=str,	default='data/own_card3d',	help='directory of the data')
    parser.add_argument('--save_dir',	type=str,   default='results/exp',		help='directory of the experiment')
    parser.add_argument('--ckpt_path',	type=str,   default=None,				help='checkpoint to test or restart training')
    parser.add_argument('--input',	    type=str,   default='', 				help='name of network input file')

    # General configuration 
    parser.add_argument('--grad_check', 	    type=bool, 	default=False,	  help='use gradient checkpointing')
    parser.add_argument('--network',   		  	type=str,  	default='',    	  help='model to use (FlowVN or FlowMRI_Net)')
    parser.add_argument('--num_stages',       	type=int,  	default=10,    	  help='number of stages in the network')
    parser.add_argument('--features_in',      	type=int,  	default=1,    	  help='number of input dimensions')
    parser.add_argument('--features_out',    	type=int,  	default=24,    	  help='number of filters for convolutional kernel')

    # FlowVN specific configuration
    parser.add_argument('--kernel_size',       	type=int,  	default=7,    	  help='xyz kernel size')
    parser.add_argument('--act',            	type=str,  	default='linear', help='what activation to use, rbf or linear (faster but more unstable due to feature range)')
    parser.add_argument('--num_act_weights',	type=int,  	default=71,    	  help='number of basis functions for activation')
    parser.add_argument('--grid',    	        type=float, default=0.25,	  help='grid size for linear act')
    parser.add_argument('--weight',           	type=float, default=0.025,	  help='scale weights for RBF kernel')
    parser.add_argument('--vmin',    		  	type=float,	default=-3.5,	  help='min value of filter response for rbf activation')
    parser.add_argument('--vmax',    		  	type=float, default=3.5, 	  help='max value of filter response for rbf activation')
    parser.add_argument('--sgd_momentum',	    type=bool, 	default=True,	  help='use sgd momentum')
    parser.add_argument('--exp_loss',    	    type=bool, 	default=False,	  help='use exponentially weighted loss')

    # Training and Testing Configuration
    parser.add_argument('--mode',           		type=str,   default='train',	help='train or test')
    parser.add_argument('--lr',             		type=float, default=1e-4,     	help='learning rate')
    parser.add_argument('--epoch',          		type=int,   default=100,      	help='number of training epoch')
    parser.add_argument('--batch_size',     		type=int,   default=1,        	help='batch size')
    parser.add_argument('--loss',       		    type=str,   default='',       	help='type of loss (ssdu or supervised)')
    parser.add_argument('--usrate', type=int, default=None, help='test only: ktGaussian undersampling rate')
    parser.add_argument('--devices', type=int, nargs='+', default=[0],
                        help='GPU device ids, e.g. --devices 0 or --devices 0 1 2 3')
    args = parser.parse_args()
    print_options(parser,args)
    args = vars(args)

    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms = True

    save_dir = Path(args['save_dir'])
    save_dir.mkdir(parents=True,exist_ok=True)
    logger = TensorBoardLogger("./results/lightning_logs", name="") if args['mode'] == 'train' else None

    n_run = str(max((int(p.split("_")[-1]) for p in glob("./results/lightning_logs/*")), default=0) + 1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, save_weights_only=True, dirpath=save_dir, filename=n_run+'-{epoch}')  # save last checkpoint only

    save_callback = CMRSaveCallback(
            save_dir=args["save_dir"],
        )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args["devices"],
        strategy='ddp_find_unused_parameters_true' if len(args["devices"]) > 1 else "auto",
        max_epochs=args["epoch"],
        accumulate_grad_batches=1,
        logger=logger,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, save_callback],
    )
    dataset = CMRx4DFlowDataSet(**args)
    if args['mode'] == 'train':
        args_val = vars(parser.parse_args())
        args_val['mode'] = 'val'
        val_dataset = CMRx4DFlowDataSet(**args_val)
        
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)     
        dataloader_val = DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=True)

        if args['ckpt_path'] is not None:
            model = UnrolledNetwork.load_from_checkpoint(args['ckpt_path'], **args)
        else:
            model = UnrolledNetwork(**args)
        trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader_val)
        
    elif args['mode'] == 'test':
        if args.get("usrate", None) is None:
            raise ValueError("test mode requires --usrate, e.g. --usrate 10")

        dataset = CMRx4DFlowDataSet(**args)  # dataset 内部会用 args['usrate']
        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)

        save_callback = CMRSaveCallback(save_dir=args["save_dir"])

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args["devices"],
            max_epochs=args["epoch"],
            logger=False,
            callbacks=[save_callback],
        )

        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda(0)
        else:
            map_location = "cpu"

        model = UnrolledNetwork.load_from_checkpoint(
            args["ckpt_path"],
            map_location=map_location,
            **args
        )
        trainer.test(model, dataloaders=dataloader)
