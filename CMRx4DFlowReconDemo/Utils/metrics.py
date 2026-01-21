import numpy as np
import torch
from .pytorch_ssim import *

def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

def SSIM(pred, gt):
    """
    pred/gt supported shapes:
      - (Nv, Nt, SPE, PE, FE)  -> treated as (N, C, D, H, W) with N=Nv, C=Nt
      - (Nt, SPE, PE, FE)      -> treated as N=1, C=Nt
      - (SPE, PE, FE)          -> treated as N=1, C=1
    """
    ssim = SSIM3D(window_size=11)

    pred = _to_tensor(pred).float()
    gt   = _to_tensor(gt).float()

    if pred.ndim == 5:          # (Nv, Nt, SPE, PE, FE)
        pred_t = pred
        gt_t   = gt
    elif pred.ndim == 4:        # (Nt, SPE, PE, FE)
        pred_t = pred.unsqueeze(0)
        gt_t   = gt.unsqueeze(0)
    elif pred.ndim == 3:        # (SPE, PE, FE)
        pred_t = pred.unsqueeze(0).unsqueeze(0)
        gt_t   = gt.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported ndim={pred.ndim}, expected 3/4/5.")

    return ssim(pred_t, gt_t).item()

def nRMSE(pred, gt, segmask=None, eps=1e-12):
    """
    Normalized RMSE within segmask.

    pred, gt: shape (Nv, Nt, SPE, PE, FE)
    segmask : shape (SPE, PE, FE)
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt, dtype=np.float32)

    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)

    # (SPE,PE,FE) -> (1,1,SPE,PE,FE) -> broadcast to (Nv,Nt,SPE,PE,FE)
    while segmask.ndim < gt.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, gt.shape).astype(np.float32)

    n = np.sum(mask)
    mse = np.sum(((pred - gt) ** 2) * mask) / (n + eps)
    denom = np.max(gt * mask) + eps
    return np.sqrt(mse) / denom

def RelErr(pred, gt, segmask=None, eps=1e-12):
    """
    Relative magnitude error for vector fields.

    pred, gt: shape (Nv, Nt, SPE, PE, FE)
    segmask : shape (SPE, PE, FE)

    Computes || |v_gt| - |v_pred| ||_2 / || |v_gt| ||_2 within mask, aggregated over (Nt,SPE,PE,FE).
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt, dtype=np.float32)

    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)

    gt_mag   = np.linalg.norm(gt, axis=0)    # (Nt,SPE,PE,FE)
    pred_mag = np.linalg.norm(pred, axis=0)  # (Nt,SPE,PE,FE)

    # (SPE,PE,FE) -> (1,SPE,PE,FE) -> broadcast to (Nt,SPE,PE,FE)
    while segmask.ndim < gt_mag.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, gt_mag.shape).astype(np.float32)

    numerator = np.sum(((gt_mag - pred_mag) ** 2) * mask)
    denominator = np.sum((gt_mag ** 2) * mask) + eps
    return np.sqrt(numerator / denominator)

def AngErr(pred, gt, segmask=None, eps=1e-8):
    """
    Angular error (degrees) for vector fields.

    pred, gt: shape (Nv, Nt, SPE, PE, FE)
    segmask : shape (SPE, PE, FE)

    Angle per voxel/time:
        acos( |<p,g>| / (||p||·||g|| + eps) )
    then averaged inside mask over (Nt,SPE,PE,FE).
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt, dtype=np.float32)

    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)

    dot = np.abs(np.sum(pred * gt, axis=0))  # (Nt,SPE,PE,FE)
    norm_p = np.linalg.norm(pred, axis=0)
    norm_g = np.linalg.norm(gt, axis=0)
    cos_sim = np.clip(dot / (norm_p * norm_g + eps), 0.0, 1.0)

    error_map = np.arccos(cos_sim)           # radians

    while segmask.ndim < error_map.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, error_map.shape).astype(np.float32)

    n_valid = np.sum(mask) + 1e-12
    return (np.sum(error_map * mask) / n_valid) / np.pi * 180.0