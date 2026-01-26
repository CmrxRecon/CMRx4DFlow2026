import os
import sys
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import numpy as np
from einops import rearrange
from torch.utils.data import Dataset

from utils.partitioning import *
from utils.misc_utils import *

sys.path.append("../")
from Utils.utils_datasl import load_mat
from Utils.utils_flow import k2i_numpy, i2k_numpy

sys.path.append("../../")
from CMRx4DFlowMaskGeneration import *

REQUIRED_FILES = ("kdata_full.mat", "coilmap.mat", "segmask.mat", "params.csv")

PathLike = Union[str, Path]

DEFAULT_OPTS = {
    "train_roots": [
        "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1&R2/TrainSet",
    ],
    "val_roots": [
        "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1&R2/ValidationSet",
    ],
    "test_roots": [
        "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1&R2/TestSet",
    ],
}


def find_valid_cases(
    roots: Sequence[PathLike],
    required_files: Iterable[str] = REQUIRED_FILES,
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


def _get_rank_world_from_env():
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world


def load_usmask_ktGaussian(case_dir: str, usrate: int, Nt: int, SPE: int, PE: int):
    """
    Reads: case_dir/usmask_ktGaussian{usrate}.mat
    Expected shape: (1, Nt, 1, SPE, PE, 1)
    Returns shape: (1, 1, Nt, 1, PE, SPE) float32 in {0,1}
    """
    mask_path = Path(case_dir) / f"usmask_ktGaussian{usrate}.mat"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    m = load_mat(str(mask_path), "usmask_ktGaussian")[()]

    expected = (1, Nt, 1, SPE, PE, 1)
    if m.shape != expected:
        raise ValueError(f"Unexpected mask shape {m.shape}, expected {expected}")

    m = np.squeeze(m, axis=(0, 2, 5))  # (Nt, SPE, PE)
    m = np.transpose(m, (1, 2, 0))     # (SPE, PE, Nt)
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

        self.D_size = options["D_size"]
        self.T_size = options["T_size"]
        self.input = options.get("input", None)
        self.loss = options["loss"]

        self.filename = []
        mode = options["mode"]

        # input handling: if input is a directory, treat it as case_dir and use it as self.filename[idx][0]
        if self.input is not None:
            case_dir = Path(self.input)
            if not case_dir.exists():
                raise FileNotFoundError(f"Input path does not exist: {case_dir}")
            case_dir = str(case_dir)

            kdata = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
            Nx = kdata.shape[-1]
            slices = range(Nx - self.D_size + 1)
            if mode == "test":
                slices = slices[:: self.D_size]

            for i in slices:
                if mode == "train":
                    self.filename.append([case_dir, i, None])
                elif mode == "val":
                    for u in self.usrate_list:
                        self.filename.append([case_dir, i, u])
                else:
                    self.filename.append([case_dir, i, self.usrate_list[0]])
        else:
            roots = options.get(f"{mode}_roots", [])
            subjects = find_valid_cases(roots)

            for patient_dir in subjects:
                kdata = load_mat(str(Path(patient_dir) / "kdata_full.mat"), "kdata_full")
                Nx = kdata.shape[-1]
                slices = range(Nx - self.D_size + 1)
                if mode == "test":
                    slices = slices[:: self.D_size]

                for i in slices:
                    if mode == "train":
                        self.filename.append([patient_dir, i, None])
                    elif mode == "val":
                        for u in self.usrate_list:
                            self.filename.append([patient_dir, i, u])
                    else:
                        self.filename.append([patient_dir, i, self.usrate_list[0]])

        if self.options.get("ddp_split_by_case", True) and (mode == "train"):
            self._split_by_case_ddp()

    def _split_by_case_ddp(self):
        rank, world = _get_rank_world_from_env()
        if world <= 1:
            return

        cases = sorted({x[0] for x in self.filename})
        if len(cases) < world:
            raise RuntimeError(
                f"Cannot split by case: num_cases={len(cases)} < world_size={world}"
            )

        my_cases = set(cases[rank::world])
        self.filename = [x for x in self.filename if x[0] in my_cases]

        if self.options.get("debug_case_split", False):
            print(f"[rank {rank}/{world}] cases={len(my_cases)} samples={len(self.filename)}")

    def __len__(self):
        return len(self.filename)

    def get_order(self):
        return self.filename

    def __getitem__(self, idx):
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))

        if self.options["mode"] != "train":
            np.random.seed(0)

        case_dir = self.filename[idx][0]
        f = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
        c = load_mat(str(Path(case_dir) / "coilmap.mat"), "coilmap")
        s = load_mat(str(Path(case_dir) / "segmask.mat"), "segmask")

        Nv, Nt, Nc, SPE, PE, FE = f.shape

        seg_idx = 0
        if self.options["network"] == "FlowVN":
            seg_idx = np.random.randint(Nv) if self.options["mode"] == "train" else 0

        if self.T_size == -1:
            cardiac_bins = range(Nt)
        else:
            first_bin = (
                random.randint(-self.T_size + 1, Nt - self.T_size)
                if self.options["mode"] == "train"
                else 0
            )
            cardiac_bins = (
                list(range(first_bin, first_bin + self.T_size))
                if self.options["mode"] == "train"
                else list(range(Nt))
            )

        bins = np.mod(cardiac_bins, Nt).astype(np.int64)

        f = sorted_read_then_gather(
            f,
            axis=1,
            idx=bins,
            fixed_slices={0: slice(seg_idx, seg_idx + 1)},
        )

        f = k2i_numpy(f, ax=[-1])[..., self.filename[idx][1] : self.filename[idx][1] + self.D_size]

        c = c[..., self.filename[idx][1] : self.filename[idx][1] + self.D_size].astype("complex64")
        s = s[self.filename[idx][1] : self.filename[idx][1] + self.D_size]

        Nv, Nt, Nc, SPE, PE, FE = f.shape

        im = np.sum(k2i_numpy(f, ax=[-2, -3]) * np.conj(c), axis=-4)

        if self.options["mode"] == "train":
            usrate = random.choice(self.usrate_list)

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
            mask = rearrange(masks_spe_pe_t, "spe pe t -> 1 1 t 1 pe spe").astype(np.float32)
        else:
            usrate = self.filename[idx][2]
            mask = load_usmask_ktGaussian(
                case_dir=case_dir,
                usrate=usrate,
                Nt=Nt,
                SPE=SPE,
                PE=PE,
            )

        f = rearrange(f, "nv nt nc spe pe fe -> nv nc nt fe pe spe")
        c = rearrange(c, "nc spe pe fe -> nc fe pe spe")
        s = rearrange(s, "spe pe fe -> fe pe spe")
        f *= mask
        im = rearrange(im, "nv nt spe pe fe -> nv nt fe pe spe")

        if self.loss == "ssdu":
            if self.options["mode"] == "train":
                mask_p1, mask_p2 = uniform_disjoint_selection(
                    mask, rho=0.2, r2=9, seed=None, venc_coherence=True
                )
            elif self.options["mode"] == "val":
                mask_p1, mask_p2 = uniform_disjoint_selection(
                    mask, rho=0.2, r2=9, seed=444, venc_coherence=True
                )
            else:
                mask_p1, mask_p2 = mask, np.zeros_like(mask)

            kdata_p1 = f * mask_p1[:, np.newaxis, :, np.newaxis]
            kdata_p2 = f * mask_p2[:, np.newaxis, :, np.newaxis]

            imdata_p1 = mriAdjointOp(
                kdata_p1,
                c[np.newaxis, :, np.newaxis, :, :, :],
                mask_p1[:, np.newaxis, :, np.newaxis],
            ).astype(np.complex64)

            norm = np.linalg.norm(f) / np.linalg.norm(f != 0)
            imdata_p1 /= norm
            kdata_p1 /= norm
            kdata_p2 /= norm

            return {
                "imdata_p1": imdata_p1,
                "kdata_p2": kdata_p2,
                "kdata_p1": kdata_p1,
                "coil_sens": c,
                "norm": norm,
                "segmentation": s.astype(bool),
                "subj": Path(case_dir).name,
            }

        if self.loss == "supervised":
            imdata_p1 = mriAdjointOp(
                f, c[np.newaxis, :, np.newaxis, :, :, :], mask
            ).astype(np.complex64)

            norm = np.linalg.norm(f) / np.linalg.norm(np.abs(f) != 0)

            if (not np.isfinite(norm)) or (norm < 1e-6):
                print(
                    f"[BAD NORM] rank {rank} case {case_dir} "
                    f"subj {Path(case_dir).name} "
                    f"slice {self.filename[idx][1]} seg {seg_idx} usrate {usrate} "
                    f"bins {bins} norm {norm} "
                    f"f_absmax {np.abs(f).max()} nnz {np.count_nonzero(f)}"
                )

            imdata_p1 /= norm
            im /= norm
            f /= norm

            meta = {
                "case_dir": case_dir,
                "subj": Path(case_dir).name,
                "slice_start": int(self.filename[idx][1]),
                "seg_idx": int(seg_idx),
                "usrate": int(usrate),
                "bins": bins.astype(np.int64),
                "Nt": int(Nt),
                "SPE": int(SPE),
                "PE": int(PE),
                "FE": int(FE),
            }

            return {
                "imdata_p1": imdata_p1,
                "gt": im,
                "kdata_p1": f,
                "coil_sens": c,
                "norm": norm,
                "segmentation": s.astype(bool),
                "subj": Path(case_dir).name,
                **meta,
            }

        raise ValueError("loss must be either 'ssdu' or 'supervised'")