# =========================
# 1) DataSet: in test mode run only one --usrate, and emit samples per seg_idx
#    In test mode filename stores [case_dir, slice_start(fixed 0), usrate(fixed), seg_idx]
# =========================
import os
import sys
import random
from pathlib import Path

import numpy as np
from einops import rearrange
from torch.utils.data import Dataset

sys.path.append('../')
from Utils.utils_datasl import load_mat
from Utils.utils_flow import k2i_numpy
from utils.partitioning import uniform_disjoint_selection

sys.path.append('../../')
from CMRx4DFlowMaskGeneration import fun_mask_gen_2d
from utils.misc_utils import mriAdjointOp

REQUIRED_FILES = ("kdata_full.mat", "coilmap.mat", "segmask.mat", "params.csv")

DEFAULT_OPTS = {
    "train_roots": [
        "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1&R2/TrainSet/Aorta/",
        # "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1&R2/TrainSet/Aorta/Center012/Philips_15T_Ambition",
    ],
    "val_roots": [
        "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1&R2/ValidationSet/Aorta/"
        # "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1&R2/ValidationSet/Aorta/Center012/Philips_15T_Ambition"
    ],
}

def find_valid_cases(roots, required_files=REQUIRED_FILES, anchor="kdata_full.mat"):
    req = tuple(required_files)
    out, seen = [], set()
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

def load_usmask_ktGaussian(case_dir, usrate, Nt, SPE, PE):
    mask_path = Path(case_dir) / f"usmask_ktGaussian{usrate}.mat"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    m = load_mat(str(mask_path), "usmask_ktGaussian")[()]
    expected = (1, Nt, 1, SPE, PE, 1)
    if m.shape != expected:
        raise ValueError(f"Unexpected mask shape {m.shape}, expected {expected}")
    m = np.squeeze(m, axis=(0, 2, 5))
    m = np.transpose(m, (1, 2, 0))
    mask = rearrange(m, "spe pe t -> 1 1 t 1 pe spe").astype(np.float32)
    return mask

def sorted_read_then_gather(x, axis, idx, fixed_slices=None):
    idx = np.asarray(idx, dtype=np.int64)
    order = np.argsort(idx)
    idx_sorted = idx[order]
    key = [slice(None)] * len(x.shape)
    if fixed_slices:
        for ax, sl in fixed_slices.items():
            key[ax] = sl
    key[axis] = idx_sorted
    out_sorted = x[tuple(key)]
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return np.take(out_sorted, inv, axis=axis)

class CMRx4DFlowDataSet(Dataset):
    def __init__(self, **kwargs):
        options = DEFAULT_OPTS.copy()
        options.update(kwargs)
        self.options = options

        self.usrate_list = [10, 20, 30, 40, 50]
        self.D_size = int(options["D_size"])
        self.T_size = int(options["T_size"])
        self.input = options.get("input", None)
        self.loss = options["loss"]
        self.network = options.get("network", "")

        mode = options["mode"]
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be one of ['train','val','test'], got {mode}")

        if mode in ("val", "test"):
            self.D_size = -1

        if self.loss not in ("ssdu", "supervised"):
            raise ValueError("loss must be either 'ssdu' or 'supervised'")

        self.test_usrate = options.get("usrate", None)
        if mode == "test":
            if self.test_usrate is None:
                raise ValueError("In test mode you must pass --usrate")
            self.test_usrate = int(self.test_usrate)

        self.filename = []

        if self.input is not None and str(self.input) != "":
            case_dir = Path(self.input)
            if not case_dir.exists():
                raise FileNotFoundError(f"Input path does not exist: {case_dir}")
            case_dir = str(case_dir)

            if mode == "test":
                kdata = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
                Nv_test = int(kdata.shape[0])
                for seg_i in range(Nv_test):
                    self.filename.append([case_dir, 0, self.test_usrate, int(seg_i)])
            else:
                kdata = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
                Nx = int(kdata.shape[-1])
                slice_starts = [0] if mode == "val" else list(range(0, Nx - self.D_size + 1)) if self.D_size != -1 else [0]
                for i in slice_starts:
                    if mode == "train":
                        self.filename.append([case_dir, int(i), None, None])
                    else:
                        for u in self.usrate_list:
                            self.filename.append([case_dir, int(i), int(u), 0])
        else:
            roots = options.get(f"{mode}_roots", [])
            subjects = find_valid_cases(roots)

            for patient_dir in subjects:
                if mode == "test":
                    seg = load_mat(str(Path(patient_dir) / "segmask.mat"), "segmask")
                    Nv_test = int(seg.shape[0]) if seg.ndim == 4 else 1
                    for seg_i in range(Nv_test):
                        self.filename.append([patient_dir, 0, self.test_usrate, int(seg_i)])
                else:
                    kdata = load_mat(str(Path(patient_dir) / "kdata_full.mat"), "kdata_full")
                    Nx = int(kdata.shape[-1])
                    slice_starts = [0] if mode == "val" else list(range(0, Nx - self.D_size + 1)) if self.D_size != -1 else [0]
                    for i in slice_starts:
                        if mode == "train":
                            self.filename.append([patient_dir, int(i), None, None])
                        else:
                            for u in self.usrate_list:
                                self.filename.append([patient_dir, int(i), int(u), 0])

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):
        mode = self.options["mode"]
        if mode != "train":
            np.random.seed(0)
            random.seed(0)

        case_dir = str(self.filename[idx][0])
        slice_start = int(self.filename[idx][1])
        stored_usrate = self.filename[idx][2]
        stored_seg_idx = self.filename[idx][3]

        c = load_mat(str(Path(case_dir) / "coilmap.mat"), "coilmap")
        s = load_mat(str(Path(case_dir) / "segmask.mat"), "segmask")

        if stored_seg_idx is None:
            if mode == "train" and self.network == "FlowVN":
                f_tmp = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
                Nv_tmp = int(f_tmp.shape[0])
                seg_idx = np.random.randint(Nv_tmp)
                del f_tmp
            else:
                seg_idx = 0
        else:
            seg_idx = int(stored_seg_idx)

        slice_end = None if self.D_size == -1 else slice_start + self.D_size

        if mode == "test":
            usrate = int(stored_usrate)
            kpath = Path(case_dir) / f"kdata_ktGaussian{usrate}.mat"
            if not kpath.is_file():
                raise FileNotFoundError(f"K-space not found: {kpath}")
            f = load_mat(str(kpath), "kdata_ktGaussian")
        else:
            f = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
            usrate = int(stored_usrate) if stored_usrate is not None else -1

        Nv, Nt, Nc, SPE, PE, FE = f.shape

        if self.T_size == -1:
            cardiac_bins = list(range(Nt))
        else:
            if mode == "train":
                first_bin = random.randint(-self.T_size + 1, Nt - self.T_size)
                cardiac_bins = list(range(first_bin, first_bin + self.T_size))
            else:
                cardiac_bins = list(range(Nt))
        bins = np.mod(cardiac_bins, Nt).astype(np.int64)

        f = sorted_read_then_gather(f, axis=1, idx=bins, fixed_slices={0: slice(seg_idx, seg_idx + 1)})

        f = k2i_numpy(f, ax=[-1])[..., slice_start:slice_end]

        c = c[..., slice_start:slice_end].astype("complex64")
        s = s[..., slice_start:slice_end]

        Nv, Nt, Nc, SPE, PE, FE = f.shape

        im = np.sum(k2i_numpy(f, ax=[-2, -3]) * np.conj(c), axis=-4)

        if mode == "train":
            usrate = random.choice(self.usrate_list)
            total_points = (PE * SPE) // usrate
            masks_spe_pe_t = fun_mask_gen_2d(
                mask_size=(PE, SPE),
                center_radius_x=0.5, center_radius_y=0.5,
                total_points=total_points,
                pattern_num=Nt,
                sigma_x=PE/5, sigma_y=SPE/5,
                min_dist_factor=3,
                rep_decay_factor=0.5,
            )
            mask = rearrange(masks_spe_pe_t, "spe pe t -> 1 1 t 1 pe spe").astype(np.float32)
        elif mode == "val":
            mask = load_usmask_ktGaussian(case_dir, usrate=int(stored_usrate), Nt=Nt, SPE=SPE, PE=PE)
        else:
            mask = np.ones((1, 1, Nt, 1, PE, SPE), dtype=np.float32)

        f = rearrange(f, "nv nt nc spe pe fe -> nv nc nt fe pe spe")
        c = rearrange(c, "nc spe pe fe -> nc fe pe spe")
        s = rearrange(s, "spe pe fe -> fe pe spe")
        f *= mask
        im = rearrange(im, "nv nt spe pe fe -> nv nt fe pe spe")

        imdata_p1 = mriAdjointOp(f, c[np.newaxis, :, np.newaxis, :, :, :], mask).astype(np.complex64)

        denom = np.linalg.norm(np.abs(f) != 0)
        norm = np.linalg.norm(f) / (denom if denom != 0 else 1.0)

        imdata_p1 /= norm
        im /= norm
        f /= norm

        return {
            "imdata_p1": imdata_p1,
            "gt": im,
            "kdata_p1": f,
            "coil_sens": c,
            "norm": norm,
            "segmentation": s.astype(bool),
            "case_dir": case_dir,
            "subj": Path(case_dir).name,
            "slice_start": int(slice_start),
            "seg_idx": int(seg_idx),
            "usrate": int(usrate),
            "bins": bins.astype(np.int64),
            "Nt": int(Nt),
            "SPE": int(SPE),
            "PE": int(PE),
            "FE": int(FE),
        }