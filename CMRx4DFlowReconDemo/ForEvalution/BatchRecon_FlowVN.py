import os
import glob
import sys
import argparse
import subprocess
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_recon", type=str, required=True)
    parser.add_argument("--path_save", type=str, required=True)

    # FlowVN
    parser.add_argument("--flowvn_main", type=str, default="../FlowVN/main.py")
    parser.add_argument("--ckpt_path", type=str, default="../FlowVN/weights/12-epoch=0.ckpt")

    # Rs
    parser.add_argument("--Rs", type=int, nargs="+", default=[10, 20, 30, 40, 50])

    # FlowVN args (match your single-case test command)
    parser.add_argument("--network", type=str, default="FlowVN")
    parser.add_argument("--features_in", type=int, default=1)
    parser.add_argument("--T_size", type=int, default=5)
    parser.add_argument("--features_out", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--num_stages", type=int, default=10)
    parser.add_argument("--loss", type=str, default="supervised")

    # Optional: GPU selection if FlowVN respects CUDA_VISIBLE_DEVICES
    parser.add_argument("--cuda_visible_devices", type=str, default=None)

    # Optional: pass-through extra args to FlowVN
    parser.add_argument("--extra", type=str, nargs="*", default=[])

    args = parser.parse_args()

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

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
                os.path.join(case_dir, "params.csv"),
            ]
            if all(os.path.exists(p) for p in need):
                case_dirs.append(case_dir)

        case_dirs = sorted(set(case_dirs))
        groups_by_R[R] = case_dirs
        total += len(case_dirs)
        print(f"[R={R}] found {len(case_dirs)} cases")

    print(f"Total cases across Rs={args.Rs}: {total}")

    for R in args.Rs:
        case_dirs = groups_by_R.get(R, [])
        if not case_dirs:
            continue

        pbar = tqdm(case_dirs, desc=f"FlowVN testing R={R}", unit="case", dynamic_ncols=True)
        for case_dir in pbar:
            try:
                rel = os.path.relpath(case_dir, args.path_recon)
                save_dir = os.path.join(args.path_save, rel)
                os.makedirs(save_dir, exist_ok=True)

                cmd = [
                    "python", args.flowvn_main,
                    "--mode", "test",
                    "--ckpt_path", args.ckpt_path,
                    "--input", case_dir,
                    "--network", args.network,
                    "--features_in", str(args.features_in),
                    "--T_size", str(args.T_size),
                    "--features_out", str(args.features_out),
                    "--kernel_size", str(args.kernel_size),
                    "--num_stages", str(args.num_stages),
                    "--loss", args.loss,
                    "--save_dir", save_dir,
                    "--usrate", str(R),
                ]
                if args.extra:
                    cmd += args.extra

                subprocess.run(cmd, check=True, env=env)

                pbar.set_postfix_str(os.path.basename(case_dir) + " -> " + os.path.basename(save_dir))

            except subprocess.CalledProcessError as e:
                pbar.write(f"[ERROR][R={R}] {case_dir}: FlowVN failed: {e}")
                continue
            except Exception as e:
                pbar.write(f"[ERROR][R={R}] {case_dir}: {repr(e)}")
                continue