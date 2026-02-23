import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--flowvn_main", type=str, default="main.py")

    parser.add_argument("--test_roots", type=str, required=True,
                        help="e.g. /mnt/.../ChallengeData/TaskS2/")
    parser.add_argument("--in_base_dir", type=str, required=True,
                        help="e.g. /mnt/.../ChallengeData/TaskS2/")
    parser.add_argument("--out_base_dir", type=str, required=True,
                        help="e.g. /mnt/.../ChallengeData_FlowVN/TaskS2")

    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--loss", type=str, default="supervised")
    parser.add_argument("--network", type=str, default="FlowVN")

    parser.add_argument("--features_in", type=int, default=1)
    parser.add_argument("--D_size", type=int, default=5)
    parser.add_argument("--T_size", type=int, default=15)
    parser.add_argument("--num_act_weights", type=int, default=71)
    parser.add_argument("--features_out", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--num_stages", type=int, default=10)

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--act", type=str, default="linear_flowvn")

    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--usrate", type=int, nargs="+", default=[10, 20, 30, 40, 50])

    parser.add_argument("--cuda_visible_devices", type=str, default=None)

    parser.add_argument("--extra", type=str, nargs="*", default=[])

    parser.add_argument("--lowmem", action="store_true",
                        help="use FlowVNLowMem for inference (forward low memory)")

    args = parser.parse_args()

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = [
        "python", "-u", args.flowvn_main,
        "--mode", args.mode,
        "--loss", args.loss,
        "--network", args.network,
        "--features_in", str(args.features_in),
        "--D_size", str(args.D_size),
        "--T_size", str(args.T_size),
        "--num_act_weights", str(args.num_act_weights),
        "--features_out", str(args.features_out),
        "--kernel_size", str(args.kernel_size),
        "--num_stages", str(args.num_stages),
        "--epoch", str(args.epoch),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--act", args.act,

        "--devices", *[str(d) for d in args.devices],

        "--test_roots", args.test_roots,
        "--in_base_dir", args.in_base_dir,
        "--out_base_dir", args.out_base_dir,

        "--usrate", *[str(r) for r in args.usrate],
        "--ckpt_path", args.ckpt_path,
    ]

    if args.lowmem:
        cmd += ["--lowmem"]

    if args.extra:
        cmd += args.extra

    print("Running command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()