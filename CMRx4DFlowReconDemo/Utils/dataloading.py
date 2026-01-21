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


@dataclass
class LazyComplexH5:
    """
    Lazy complex wrapper backed by two HDF5 datasets: real + 1j*imag.
    Slicing triggers reading only the sliced region.
    """
    file: h5py.File
    real: h5py.Dataset
    imag: h5py.Dataset

    @property
    def shape(self):
        return self.real.shape

    @property
    def dtype(self):
        if self.real.dtype == np.float32 and self.imag.dtype == np.float32:
            return np.complex64
        return np.complex128

    def __getitem__(self, idx):
        return self.real[idx] + 1j * self.imag[idx]

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def load_data(data_path, key, *, real_key="real", imag_key="imag"):
    """
    Open ONE dataset from a .mat/.h5 file lazily.
    If `key` points to a group that contains `real_key` and `imag_key`,
    return a lazy complex wrapper.
    """
    f = h5py.File(data_path, "r")  # keep open
    try:
        if key not in f:
            keys = list(f.keys())
            raise KeyError(f"Dataset '{key}' not found in: {data_path}. Top-level keys: {keys}")

        obj = f[key]

        if isinstance(obj, h5py.Dataset):
            return LazyH5(file=f, dset=obj)

        if isinstance(obj, h5py.Group) and (real_key in obj) and (imag_key in obj):
            real = obj[real_key]
            imag = obj[imag_key]
            # optional shape check
            if real.shape != imag.shape:
                raise ValueError(f"real.shape {real.shape} != imag.shape {imag.shape} for group '{key}'")
            return LazyComplexH5(file=f, real=real, imag=imag)

        raise TypeError(
            f"Key '{key}' is a {type(obj).__name__} but not a dataset and not a (real/imag) group. "
            f"Available keys under '{key}': {list(obj.keys()) if isinstance(obj, h5py.Group) else 'N/A'}"
        )

    except Exception:
        f.close()
        raise

# -------------------------
# Save to .mat (v7.3 style)
# -------------------------
def save_mat(save_path, key, data, real_key="real", imag_key="imag"):
    """
    Save data to an HDF5 (.h5/.mat v7.3-style) file using h5py.
    """
    with h5py.File(save_path, "w") as f:
        if key in f:
            del f[key]
        arr = np.asanyarray(data)
        if np.iscomplexobj(arr):
            dt = np.dtype([(real_key, arr.real.dtype), (imag_key, arr.imag.dtype)])
            out = np.empty(arr.shape, dtype=dt)
            out[real_key] = arr.real
            out[imag_key] = arr.imag

            f.create_dataset(key, data=out, compression="gzip", compression_opts=4)
        else:
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)