#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def _read_sample(ds, idx: int):
    """
    ds is expected to be (N, C, H, W). Returns (H, W) for C=1, otherwise (C, H, W).
    """
    x = ds[idx]  # (C,H,W)
    if x.ndim != 3:
        raise ValueError(f"Expected per-sample shape (C,H,W), got {x.shape}")
    if x.shape[0] == 1:
        return x[0]
    return x


def main():
    ap = argparse.ArgumentParser(description="Visualize 4 random (a,u) pairs from an HDF5 Darcy dataset.")
    ap.add_argument("--file", type=str, required=True, help="Path to the .h5 file (e.g. threshold_dataset_256_1000.h5)")
    ap.add_argument("--k", type=int, default=4, help="Number of random samples to plot (default: 4)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (default: None)")
    ap.add_argument("--save", type=str, default=None, help="If set, save figure to this path instead of just showing it.")
    ap.add_argument("--u-shared-scale", action="store_true",
                    help="Use a shared color scale for u across the selected samples.")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rng = np.random.default_rng(args.seed)

    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError("Expected group '/data' in the HDF5 file.")
        g = f["data"]
        if "a" not in g or "u" not in g:
            raise KeyError("Expected datasets '/data/a' and '/data/u' in the HDF5 file.")

        a_ds = g["a"]  # (N, C_a, H, W)
        u_ds = g["u"]  # (N, C_u, H, W)
        cg_ds = g["cg_info"] if "cg_info" in g else None

        if a_ds.shape[0] != u_ds.shape[0]:
            raise ValueError(f"Mismatched sample counts: a has {a_ds.shape[0]}, u has {u_ds.shape[0]}")

        N = a_ds.shape[0]
        k = min(int(args.k), N)
        idxs = rng.choice(N, size=k, replace=False)

        # Preload u samples if we want a shared scale
        u_samples = None
        u_vmin = u_vmax = None
        if args.u_shared_scale:
            u_samples = [np.asarray(_read_sample(u_ds, int(i))) for i in idxs]
            u_vmin = min(u.min() for u in u_samples)
            u_vmax = max(u.max() for u in u_samples)

        fig, axes = plt.subplots(k, 2, figsize=(10, 2.8 * k), constrained_layout=True)
        if k == 1:
            axes = np.array([axes])  # force shape (1,2)

        im_a_first = None
        im_u_first = None

        for row, idx in enumerate(idxs):
            idx = int(idx)

            a = np.asarray(_read_sample(a_ds, idx))
            u = u_samples[row] if u_samples is not None else np.asarray(_read_sample(u_ds, idx))

            cg_info = None
            if cg_ds is not None:
                cg_info = int(cg_ds[idx])

            ax_a = axes[row, 0]
            ax_u = axes[row, 1]

            # Coefficient is thresholded (3 or 12) in your generator; fix scale for comparability.
            im_a = ax_a.imshow(a, origin="lower")#, vmin=3.0, vmax=12.0)
            ax_a.set_title(f"a (idx={idx}" + (f", cg={cg_info}" if cg_info is not None else "") + ")")
            ax_a.set_xticks([])
            ax_a.set_yticks([])

            if args.u_shared_scale:
                im_u = ax_u.imshow(u, origin="lower", vmin=u_vmin, vmax=u_vmax)
            else:
                im_u = ax_u.imshow(u, origin="lower")
            ax_u.set_title("u")
            ax_u.set_xticks([])
            ax_u.set_yticks([])

            if im_a_first is None:
                im_a_first = im_a
            if im_u_first is None:
                im_u_first = im_u

        # One colorbar per column
        fig.colorbar(im_a_first, ax=axes[:, 0], fraction=0.046, pad=0.02)
        fig.colorbar(im_u_first, ax=axes[:, 1], fraction=0.046, pad=0.02)

        if args.save:
            out = Path(args.save)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=200)
            print(f"Saved figure to: {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
    # usage python dataset_viz.py --file threshold_dataset_train_nodes257_1024.h5 --seed 0 --u-shared-scale
    #threshold_dataset_train_nodes257_1024.h5
    #
    # python dataset_viz.py --file lognormal_dataset_train_nodes257_1025.h5 --seed 0 --u-shared-scale