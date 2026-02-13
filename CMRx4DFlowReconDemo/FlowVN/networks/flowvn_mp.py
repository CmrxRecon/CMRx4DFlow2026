# networks/flowvn_mp.py
from __future__ import annotations
from typing import Optional, List, Sequence, Dict, Tuple, Union
import torch
from torch import nn

from networks.flowvn import VnMriReconCell


def _force_move_cell_tensors(cell: nn.Module, device: torch.device):
    """
    Move any tensor-like attributes inside the cell to device, even if they are
    not registered as parameters/buffers.
    Test-only hack to handle custom-stored kernels.
    """
    for name, val in vars(cell).items():
        if isinstance(val, torch.Tensor):
            setattr(cell, name, val.to(device))
        elif isinstance(val, (list, tuple)):
            moved = False
            new_seq = []
            for x in val:
                if isinstance(x, torch.Tensor):
                    new_seq.append(x.to(device))
                    moved = True
                else:
                    new_seq.append(x)
            if moved:
                setattr(cell, name, type(val)(new_seq))
        elif isinstance(val, dict):
            moved = False
            new_d = {}
            for k, x in val.items():
                if isinstance(x, torch.Tensor):
                    new_d[k] = x.to(device)
                    moved = True
                else:
                    new_d[k] = x
            if moved:
                setattr(cell, name, new_d)

    cell.to(device)


def _balanced_stage_ranges(num_stages: int, num_devices: int) -> List[Tuple[int, int]]:
    """
    Return ranges [start, end) for each device, balanced as evenly as possible.
    Example: num_stages=10, num_devices=3 -> [(0,4),(4,7),(7,10)]
    """
    if num_devices <= 0:
        raise ValueError("num_devices must be >= 1")
    if num_stages <= 0:
        raise ValueError("num_stages must be >= 1")
    if num_devices > num_stages:
        raise ValueError(f"num_devices cannot exceed num_stages. got {num_devices} > {num_stages}")

    q, r = divmod(num_stages, num_devices)  # base, remainder
    ranges: List[Tuple[int, int]] = []
    start = 0
    for di in range(num_devices):
        n = q + (1 if di < r else 0)
        end = start + n
        ranges.append((start, end))
        start = end
    return ranges


class FlowVNModelParallel(nn.Module):
    """
    split: number of devices to use.
    stages will be distributed across devices as evenly as possible.

    You can pass:
      - split=N and devices=["cuda:0","cuda:1",...]
    Or (backward-compatible-ish for N=2):
      - split=2 and device0="cuda:0", device1="cuda:1"
    """

    def __init__(
        self,
        *,
        split: int,
        devices: Optional[Sequence[Union[str, torch.device]]] = None,
        device0: str = "cuda:0",
        device1: str = "cuda:1",
        **options,
    ):
        super().__init__()
        self.options = dict(options)
        self.nc = int(options["num_stages"])
        self.num_devices = int(split)

        if not (1 <= self.num_devices <= self.nc):
            raise ValueError(
                f"split (num_devices) must be in [1, num_stages]. "
                f"got split={split}, num_stages={self.nc}"
            )

        if devices is None:
            if self.num_devices == 1:
                devices = [device0]
            elif self.num_devices == 2:
                devices = [device0, device1]
            else:
                # default to cuda:0..cuda:(N-1)
                devices = [f"cuda:{i}" for i in range(self.num_devices)]

        if len(devices) != self.num_devices:
            raise ValueError(f"len(devices) must equal split. got {len(devices)} vs {self.num_devices}")

        self.devs: List[torch.device] = [torch.device(d) for d in devices]
        self.ranges: List[Tuple[int, int]] = _balanced_stage_ranges(self.nc, self.num_devices)

        self.cell_list = nn.ModuleList([VnMriReconCell(block_id=i, **options) for i in range(self.nc)])

        # place cells on their assigned devices (force-move inner tensor attrs)
        for dev_idx, (s, e) in enumerate(self.ranges):
            dev = self.devs[dev_idx]
            for i in range(s, e):
                _force_move_cell_tensors(self.cell_list[i], dev)

        self.exp_loss = bool(options.get("exp_loss", False))

    @torch.no_grad()
    def forward(self, x, f, c, usrate):
        # Prepare per-device copies (stable & simple; uses more VRAM)
        x_real = torch.view_as_real(x)

        f_on: Dict[torch.device, torch.Tensor] = {}
        c_on: Dict[torch.device, torch.Tensor] = {}
        us_on: Dict[torch.device, Union[torch.Tensor, float, int]] = {}

        for dev in self.devs:
            f_on[dev] = f.to(dev, non_blocking=True)
            c_on[dev] = c.to(dev, non_blocking=True)
            us_on[dev] = usrate.to(dev, non_blocking=True) if torch.is_tensor(usrate) else usrate

        # Start on device 0 (first in list)
        dev0 = self.devs[0]
        x_cur = x_real.to(dev0, non_blocking=True)

        S_prev: Optional[torch.Tensor] = None

        for dev_idx, (s, e) in enumerate(self.ranges):
            dev = self.devs[dev_idx]

            # Move state to current device when switching
            if x_cur.device != dev:
                x_cur = x_cur.to(dev, non_blocking=True)
            if S_prev is not None and S_prev.device != dev:
                S_prev = S_prev.to(dev, non_blocking=True)

            f_dev = f_on[dev]
            c_dev = c_on[dev]
            us_dev = us_on[dev]

            for i in range(s, e):
                # cells already on correct device by __init__
                x_cur, S_prev = self.cell_list[i](x_cur, f_dev, c_dev, us_dev, S_prev)

        return torch.view_as_complex(x_cur)