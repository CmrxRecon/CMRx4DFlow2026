import os
from dataclasses import dataclass

import h5py
import hdf5storage
import numpy as np
import pandas as pd


# -------------------------
# Params (CSV) utilities
# -------------------------
def read_params_csv(filepath):
    """
    Read acquisition/reconstruction parameters from a CSV file.
    Returns dict or None.
    """
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        if df.empty:
            return None

        row = df.iloc[0]

        array_fields = {"resolution", "FOV", "VENC", "matrix_size", "spatial_order", "venc_order"}
        float_list_fields = {"resolution", "FOV", "VENC"}
        int_array_fields = {"matrix_size"}
        float_scalar_fields = {"RR", "FA", "TE", "TR", "field_strength"}

        params = {}

        for key, value in row.items():
            if pd.isna(value):
                params[key] = None
                continue

            if key in array_fields:
                items = [s.strip() for s in str(value).split(";") if s.strip() != ""]
                if key in int_array_fields:
                    params[key] = np.array([int(float(v)) for v in items], dtype=int)
                elif key in float_list_fields:
                    params[key] = [float(v) for v in items]
                else:
                    params[key] = items
                continue

            if key in float_scalar_fields:
                params[key] = float(value)
                continue

            params[key] = value

        return params

    except Exception as exc:
        raise RuntimeError(f"Failed to read params CSV: {filepath}") from exc

# -------------------------
# Lazy HDF5 dataset wrapper
# -------------------------
@dataclass
class LazyH5:
    """
    Lightweight wrapper around an HDF5 dataset that supports lazy slicing.
    Keep file open; call .close() when done.
    """
    file: h5py.File
    dset: h5py.Dataset

    @property
    def shape(self):
        return self.dset.shape

    @property
    def dtype(self):
        return self.dset.dtype

    def __getitem__(self, idx):
        return self.dset[idx]

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


# -------------------------
# Load one dataset (lazy) - save_mat style
# -------------------------
def load_data(data_path, key):
    """
    Open ONE dataset from a .mat/.h5 file lazily.

    Parameters
    ----------
    data_path : str
        Path to .mat (v7.3/HDF5) file.
    key : str
        Dataset name inside file.

    Returns
    -------
    LazyH5
        Lazy dataset wrapper (file kept open; call .close()).
    """
    f = h5py.File(data_path, "r")  # keep open
    try:
        if key not in f:
            keys = list(f.keys())
            raise KeyError(f"Dataset '{key}' not found in: {data_path}. Top-level keys: {keys}")
        return LazyH5(file=f, dset=f[key])
    except Exception:
        f.close()
        raise


# -------------------------
# Save to .mat (v7.3 style)
# -------------------------
def save_mat(save_path, key, data):
    """
    Save data to an HDF5 (.h5/.mat v7.3-style) file using h5py.
    """
    with h5py.File(save_path, "w") as f:
        if key in f:
            del f[key]
        if np.iscomplexobj(data):
            g = f.create_group(key)
            g.create_dataset("real", data=np.asanyarray(data).real, compression="gzip", compression_opts=4)
            g.create_dataset("imag", data=np.asanyarray(data).imag, compression="gzip", compression_opts=4)
        else:
            f.create_dataset(key, data=np.asanyarray(data), compression="gzip", compression_opts=4)