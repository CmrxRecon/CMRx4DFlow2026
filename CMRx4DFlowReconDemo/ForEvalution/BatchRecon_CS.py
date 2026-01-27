import os
import glob
import sys
import argparse
import time
import csv 
import torch
from tqdm import tqdm
from einops import rearrange

sys.path.append("../")
from Utils.CS_LLR_exec import CS_LLR
from Utils.utils_datasl import read_params_csv, save_coo_npz, load_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_recon", type=str, required=True)
    parser.add_argument("--path_save", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--Rs", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--lamb_tv", type=float, default=0.0)
    parser.add_argument("--lamb_llr", type=str, nargs="*", default=[])
    args = parser.parse_args()

    device_type = torch.device(args.device).type
    is_cuda = device_type == "cuda"

    lamb_llr_by_R = {}
    for item in args.lamb_llr:
        if "=" not in item:
            raise ValueError(f"Invalid --lamb_llr entry '{item}', expected R=value")
        k, v = item.split("=", 1)
        lamb_llr_by_R[int(k)] = float(v)

    groups_by_R = {}
    total = 0

    for R in args.Rs:
        pattern = os.path.join(args.path_recon, "**", f"kdata_ktGaussian{R}.mat")
        kdata_files = glob.glob(pattern, recursive=True)

        case_dirs = []
        for kf in kdata_files:
            case_dir = os.path.dirname(kf)
            need = [
                os.path.join(case_dir, f"kdata_ktGaussian{R}.mat"),
                os.path.join(case_dir, f"usmask_ktGaussian{R}.mat"),
                os.path.join(case_dir, "segmask.mat"),
                os.path.join(case_dir, "coilmap.mat"),
            ]
            if all(os.path.exists(p) for p in need):
                case_dirs.append(case_dir)

        case_dirs = sorted(set(case_dirs))
        groups_by_R[R] = case_dirs
        total += len(case_dirs)

    print(f"Total cases: {total}")

    for R in args.Rs:
        if R not in lamb_llr_by_R:
            continue

        lamb_llr = lamb_llr_by_R[R]
        case_dirs = groups_by_R.get(R, [])

        pbar = tqdm(case_dirs, desc=f"R={R}", unit="case", dynamic_ncols=True)
        for case_dir in pbar:
            try:
                kdata_us = load_mat(os.path.join(case_dir, f"kdata_ktGaussian{R}.mat"), key="kdata_ktGaussian")[()]
                coilmap = load_mat(os.path.join(case_dir, "coilmap.mat"), key="coilmap")[()]
                segmask = load_mat(os.path.join(case_dir, "segmask.mat"), key="segmask")[()]

                ksp = rearrange(kdata_us, "nv nt nc spe pe fe -> nv nc nt fe pe spe")
                coils = rearrange(coilmap, "nc spe pe fe -> nc 1 fe pe spe")

                if is_cuda:
                    torch.cuda.synchronize(args.device)
                
                t_start = time.time()
                
                img_csllr = CS_LLR(args.lamb_tv, lamb_llr, ksp, coils, seg=True, dev=args.device)
                
                if is_cuda:
                    torch.cuda.synchronize(args.device)
                
                t_end = time.time()
                recon_time_sec = t_end - t_start

                img_csllr = rearrange(img_csllr, "nv nt fe pe spe -> nv nt spe pe fe")
                img_csllr_masked = img_csllr * segmask[None, None]

                rel = os.path.relpath(case_dir, args.path_recon)
                save_dir = os.path.join(args.path_save, rel)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"img_ktGaussian{R}.npz")
                save_coo_npz(save_path, img_csllr_masked)

                csv_path = os.path.join(save_dir, f"recontime_ktGaussian{R}.csv")
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["recontime"])
                    w.writerow([recon_time_sec])

                pbar.set_postfix_str(f"Time: {recon_time_sec:.2f}s")

            except Exception as e:
                pbar.write(f"[ERROR] {case_dir}: {e}")
                continue