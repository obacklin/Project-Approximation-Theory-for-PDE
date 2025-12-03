import torch
import numpy as np
import h5py
from torch.utils.data import Dataset

import os, math, h5py, numpy as np, torch
from torch.utils.data import Dataset

## h5_fields_dataset_buffered.py
import h5py, numpy as np, torch
from torch.utils.data import Dataset
from typing import Optional

class H5FieldsDatasetBuffered(Dataset):
    """
    Map-style dataset with a RAM buffer of contiguous samples to reduce disk I/O.
    Yields (a,u) as torch.float32 with shape (1,H,W). One channel enforced.

    Assumes HDF5 datasets at keys `key_a`, `key_u` with shapes:
      - (M, 1, H, W)  or
      - (M, H, W)     (channel dim will be added)
    """
    def __init__(
        self,
        h5_path: str,
        key_a: str = "data/a",
        key_u: str = "data/u",
        indices: Optional[np.ndarray] = None,  # e.g., np.arange(0, N_train) or last 2000 for val
        buffer_size: int = 1000,               # number of samples kept in RAM at once
        normalize_a: Optional[str] = "per_sample",  # None or "per_sample"
        transpose_hw: bool = False,            # set True only if last two dims are swapped in file
        seed: int = 42,                        # controls block shuffle order
        dtype = np.float32,
    ):
        self.h5_path = h5_path
        self.key_a = key_a
        self.key_u = key_u
        self.normalize_a = normalize_a
        self.transpose_hw = transpose_hw
        self.dtype = dtype
        self.seed = int(seed)

        # Shape metadata
        with h5py.File(h5_path, "r") as f:
            A = f[key_a]; U = f[key_u]
            assert A.shape[0] == U.shape[0], "Mismatched number of samples."
            self.M = int(A.shape[0])
            if len(A.shape) == 4:
                _, C, H, W = A.shape
                assert C == 1, f"Expected one channel, found {C}."
            elif len(A.shape) == 3:
                _, H, W = A.shape
            else:
                raise ValueError(f"Unsupported shape for {key_a}: {A.shape}")
        self.H, self.W = int(H), int(W)

        # Selection (default: all)
        self.indices = np.arange(self.M, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.L = int(self.indices.size)

        # Blocking
        self.buffer_size = int(max(1, buffer_size))
        self.block_offsets = np.arange(0, self.L, self.buffer_size, dtype=np.int64)  # starts in dataset-order space
        self.n_blocks = int(self.block_offsets.size)
        self._rng = np.random.RandomState(self.seed)
        self._block_perm = np.arange(self.n_blocks, dtype=np.int64)  # will be permuted per-epoch

        # Active buffer (per-process/worker)
        self._file = None
        self._a_ds = None
        self._u_ds = None
        self._active_block_id = None    # position in (0..n_blocks-1) after permutation
        self._buf_pos_start = None      # position range in [0..L)
        self._buf_pos_end = None
        self._buf_a = None              # torch.Tensor (Bbuf,1,H,W) on CPU
        self._buf_u = None

    # ---------- file handling ----------
    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._a_ds = self._file[self.key_a]
            self._u_ds = self._file[self.key_u]

    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass

    # ---------- epoch/block shuffling ----------
    def set_epoch(self, epoch: int):
        """Call this once per epoch (from your training loop) to reshuffle block order."""
        self._rng = np.random.RandomState(self.seed + int(epoch))
        self._block_perm = self._rng.permutation(self.n_blocks)
        # force buffer reload on next __getitem__
        self._active_block_id = None
        self._buf_a = self._buf_u = None

    # ---------- dataset protocol ----------
    def __len__(self):
        return self.L

    def _load_block(self, perm_block_id: int):
        """Load one contiguous block (by positions in self.indices) into RAM."""
        self._ensure_open()

        # Map permuted block id -> position range [pos_start, pos_end)
        base_block_id = int(self._block_perm[perm_block_id])
        pos_start = int(self.block_offsets[base_block_id])
        pos_end = int(min(pos_start + self.buffer_size, self.L))
        pos_slice = slice(pos_start, pos_end)

        # File sample indices for this block
        file_idx_block = self.indices[pos_slice]  # (Bbuf,)
        Bbuf = int(file_idx_block.size)

        # If contiguous in file, slice; else fancy-index (slower but robust)
        contiguous = (file_idx_block[-1] - file_idx_block[0] + 1 == Bbuf) and np.all(np.diff(file_idx_block) == 1)
        if contiguous:
            a_np = self._a_ds[file_idx_block[0]: file_idx_block[0] + Bbuf]
            u_np = self._u_ds[file_idx_block[0]: file_idx_block[0] + Bbuf]
        else:
            a_np = self._a_ds[file_idx_block]
            u_np = self._u_ds[file_idx_block]

        # Ensure shapes (Bbuf,1,H,W) and dtype
        if a_np.ndim == 3:  # (Bbuf,H,W)
            a_np = a_np[:, None, ...]
        if u_np.ndim == 3:
            u_np = u_np[:, None, ...]
        if self.transpose_hw:
            a_np = a_np.swapaxes(-1, -2)
            u_np = u_np.swapaxes(-1, -2)

        a_np = np.ascontiguousarray(a_np, dtype=self.dtype)
        u_np = np.ascontiguousarray(u_np, dtype=self.dtype)

        # Convert to torch (CPU). Normalize per-sample if requested.
        a_t = torch.from_numpy(a_np)       # (Bbuf,1,H,W)
        u_t = torch.from_numpy(u_np)

        if self.normalize_a == "per_sample":
            # (B,1,1,1) mean/std broadcasting over H,W
            mu = a_t.mean(dim=(-1, -2), keepdim=True)
            sd = a_t.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
            a_t = (a_t - mu) / sd

        self._buf_a = a_t
        self._buf_u = u_t
        self._active_block_id = perm_block_id
        self._buf_pos_start = pos_start
        self._buf_pos_end = pos_end

    def __getitem__(self, idx: int):
        """
        idx: 0..L-1 (position in the *epoch order* = permuted blocks).
        We compute which permuted block contains this position, load it if needed,
        then return the entry inside the RAM buffer.
        """
        if idx < 0 or idx >= self.L:
            raise IndexError(idx)

        # Determine which permuted block this idx falls into
        # Each block contributes up to `buffer_size` positions in order.
        perm_block_id = idx // self.buffer_size
        if (self._active_block_id is None) or (perm_block_id != self._active_block_id):
            self._load_block(perm_block_id)

        # Map global idx -> position in dataset-order space
        # Compute start position of this permuted block in the *epoch order*
        perm_block_start_pos = perm_block_id * self.buffer_size
        local_offset = idx - perm_block_start_pos                  # 0..buffer_size-1
        pos = self._buf_pos_start + local_offset
        if pos >= self._buf_pos_end:
            # idx points beyond the last element of the (short) tail block
            # Adjust within the valid tail length
            pos = self._buf_pos_end - 1
            local_offset = pos - self._buf_pos_start

        a = self._buf_a[local_offset]  # (1,H,W)
        u = self._buf_u[local_offset]
        return a, u
