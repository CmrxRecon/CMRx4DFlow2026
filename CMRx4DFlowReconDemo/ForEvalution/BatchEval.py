import os
import glob
import sys
import argparse
import csv
from tqdm import tqdm

sys.path.append("../")
from CMRx4DFlowReconDemo.Utils.utils_datasl import load_coo_npz
from CMRx4DFlowReconDemo.Utils.utils_flow import load_data
from CMRx4DFlowReconDemo.Utils.utils_flow import SSIM, nRMSE, RelErr, AngErr  # adjust if these live elsewhere


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_recon", type=str, required=True, help="Root where reconstructed npz are saved")
    parser.add_argument("--path_gt", type=str, required=True, help="Root where GT img_gt.npz is located")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--Rs", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--recon_name_tpl", type=str, default="img_ktGaussian{R}.npz")
    parser.add_argument("--gt_name", type=str, default="img_gt.npz")
    parser.add_argument("--seg_name", type=str, default="segmask.mat")
    args = parser.parse_args()

    rows = []
    fieldnames = ["rel_dir", "recon_file", "gt_file", "R", "ssim_mag", "nrmse_mag", "relerr_flow", "angerr_flow_deg"]

    for R in args.Rs:
        recon_name = args.recon_name_tpl.format(R=R)
        pattern = os.path.join(args.path_recon, "**", recon_name)
        recon_files = sorted(glob.glob(pattern, recursive=True))

        if len(recon_files) == 0:
            print(f"[R={R}] found 0 recon files with pattern: {pattern}")
            continue

        pbar = tqdm(recon_files, desc=f"Scoring R={R}", unit="case", dynamic_ncols=True)
        for recon_path in pbar:
            try:
                rel_dir = os.path.relpath(os.path.dirname(recon_path), args.path_recon)
                gt_dir = os.path.join(args.path_gt, rel_dir)

                gt_path = os.path.join(gt_dir, args.gt_name)
                seg_path = os.path.join(gt_dir, args.seg_name)

                if not os.path.exists(gt_path):
                    pbar.write(f"[WARN][R={R}] missing GT: {gt_path}")
                    continue
                if not os.path.exists(seg_path):
                    pbar.write(f"[WARN][R={R}] missing segmask: {seg_path}")
                    continue

                # Load recon and GT (dense)
                csllr = load_coo_npz(recon_path, as_dense=True)
                gt = load_coo_npz(gt_path, as_dense=True)

                # Load segmask
                segmask = load_data(seg_path, key="segmask")[()]

                mag_csllr, flow_csllr = csllr
                mag_gt, flow_gt = gt

                ssim_val = SSIM(mag_csllr, mag_gt, segmask)
                nrmse_val = nRMSE(mag_csllr, mag_gt, segmask)
                relerr_val = RelErr(flow_csllr, flow_gt, segmask)
                angerr_val = AngErr(flow_csllr, flow_gt, segmask)

                rows.append({
                    "rel_dir": rel_dir,
                    "recon_file": os.path.basename(recon_path),
                    "gt_file": os.path.basename(gt_path),
                    "R": R,
                    "ssim_mag": float(ssim_val),
                    "nrmse_mag": float(nrmse_val),
                    "relerr_flow": float(relerr_val),
                    "angerr_flow_deg": float(angerr_val),
                })

                pbar.set_postfix_str(f"{os.path.basename(os.path.dirname(recon_path))} ssim={ssim_val:.3f}")

            except Exception as e:
                pbar.write(f"[ERROR][R={R}] {recon_path}: {repr(e)}")
                continue

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {args.out_csv}")