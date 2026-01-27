import os
import glob
import sys
import argparse
import csv
import re
import numpy as np
from tqdm import tqdm

sys.path.append("../")
from Utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_recon", type=str, required=True, help="Root where reconstructed npz are saved")
    parser.add_argument("--path_gt", type=str, required=True, help="Root where GT img_gt.npz is located")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--recon_name_tpl", type=str, default="img_ktGaussian{R}.npz")
    parser.add_argument("--gt_name", type=str, default="img_gt.npz")
    parser.add_argument("--corr_fit_order", type=int, default=3)
    parser.add_argument("--corr_th", type=float, default=0.1)
    args = parser.parse_args()

    rows = []
    fieldnames = [
        "rel_dir", "recon_file", "gt_file", "R",
        "ssim_mag", "nrmse_mag", "relerr_flow", "angerr_flow_deg",
        "recontime"
    ]

    pattern = os.path.join(args.path_recon, "**", "img_ktGaussian*.npz")
    recon_files = sorted(glob.glob(pattern, recursive=True))

    if len(recon_files) == 0:
        print(f"found 0 recon files with pattern: {pattern}")
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        print(f"\nWrote 0 rows -> {args.out_csv}")
        raise SystemExit(0)

    groups = {}
    all_Rs = set()
    for recon_path in recon_files:
        fname = os.path.basename(recon_path)
        m = re.search(r"ktGaussian(\d+)\.npz$", fname)
        if m is None:
            continue
        R = int(m.group(1))
        rel_dir = os.path.relpath(os.path.dirname(recon_path), args.path_recon)
        groups.setdefault(rel_dir, {})[R] = recon_path
        all_Rs.add(R)

    rel_dirs = sorted(groups.keys())
    print(f"Found {len(rel_dirs)} cases, Rs={sorted(all_Rs)}")

    corr_cache = {}

    pbar = tqdm(rel_dirs, desc="Scoring cases", unit="case", dynamic_ncols=True)
    for rel_dir in pbar:
        gt_dir = os.path.join(args.path_gt, rel_dir)
        gt_path = os.path.join(gt_dir, args.gt_name)
        seg_path = os.path.join(gt_dir, "segmask.mat")

        if not os.path.exists(gt_path):
            pbar.write(f"[WARN] missing GT: {gt_path}")
            continue
        if not os.path.exists(seg_path):
            pbar.write(f"[WARN] missing segmask: {seg_path}")
            continue

        try:
            segmask = load_mat(seg_path, key="segmask")[()]
        except Exception as e:
            pbar.write(f"[ERROR] {seg_path}: {repr(e)}")
            continue

        try:
            if rel_dir not in corr_cache:
                gt_for_corr = load_coo_npz(gt_path, as_dense=True)
                corr_cache[rel_dir] = execute_MSAC(
                    gt_for_corr,
                    corr_fit_order=args.corr_fit_order,
                    th=args.corr_th,
                )
            corr_maps = corr_cache[rel_dir]
        except Exception as e:
            pbar.write(f"[ERROR] corr_maps failed for {rel_dir}: {repr(e)}")
            continue

        try:
            gt = load_coo_npz(gt_path, as_dense=True)
            gt[1:] *= np.exp(-1j * corr_maps)
            mag_gt, flow_gt = complex2magflow(gt)
        except Exception as e:
            pbar.write(f"[ERROR] GT processing failed for {rel_dir}: {repr(e)}")
            continue

        for R, recon_path in sorted(groups[rel_dir].items()):
            try:
                # --- Read reconstruction time from CSV ---
                recon_dir = os.path.dirname(recon_path)
                time_csv_path = os.path.join(recon_dir, f"recontime_ktGaussian{R}.csv")
                recon_time_val = 0.0
                
                if os.path.exists(time_csv_path):
                    with open(time_csv_path, "r", newline="") as tf:
                        reader = csv.reader(tf)
                        header = next(reader, None)  # skip "recontime" header
                        time_row = next(reader, None)
                        if time_row:
                            recon_time_val = float(time_row[0])
                else:
                    pbar.write(f"[WARN] missing time CSV: {time_csv_path}")
                # ----------------------------------------

                csllr = load_coo_npz(recon_path, as_dense=True)
                csllr[1:] *= np.exp(-1j * corr_maps)

                mag_csllr, flow_csllr = complex2magflow(csllr)

                ssim_val = SSIM(mag_csllr, mag_gt, segmask)
                nrmse_val = nRMSE(mag_csllr, mag_gt, segmask)
                relerr_val = RelErr(flow_csllr, flow_gt, segmask)
                angerr_val = AngErr(flow_csllr, flow_gt, segmask)

                rows.append({
                    "rel_dir": rel_dir,
                    "recon_file": os.path.basename(recon_path),
                    "gt_file": os.path.basename(gt_path),
                    "R": int(R),
                    "ssim_mag": float(ssim_val),
                    "nrmse_mag": float(nrmse_val),
                    "relerr_flow": float(relerr_val),
                    "angerr_flow_deg": float(angerr_val),
                    "recontime": recon_time_val
                })
            except Exception as e:
                pbar.write(f"[ERROR][R={R}] {recon_path}: {repr(e)}")
                continue

        pbar.set_postfix_str(f"{os.path.basename(rel_dir)} Rs={sorted(groups[rel_dir].keys())}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {args.out_csv}")