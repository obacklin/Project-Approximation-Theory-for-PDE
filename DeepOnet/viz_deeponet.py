# viz_deeponet.py (paper-ready, horizontal layout, random index)

import os
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from DeepONet2 import (DeepONet, DeepONetBranchCNN, Trunk)
from data_reformat import H5FieldsDatasetBuffered

# --------- Full-grid prediction (unchanged) ----------
@torch.no_grad()
def predict_full_grid(model, a, H, W, device=None, chunk_pts=16384):
    """
    a: (B,1,H,W)  -> returns (B,H,W)
    Builds an (x,y) grid in [0,1]^2 with ij-indexing and evaluates in chunks.
    """
    device = device or a.device
    B = a.size(0)
    ys = torch.linspace(0, 1, H, device=device)  # y <- row i
    xs = torch.linspace(0, 1, W, device=device)  # x <- col j
    Y, X = torch.meshgrid(ys, xs, indexing="ij")              # (H,W)
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)   # (H*W,2)
    XY = XY.unsqueeze(0).expand(B, -1, -1)                    # (B,H*W,2)

    phi = model.branch(a.to(device))                           # (B,p)
    preds = []
    for i in range(0, H*W, chunk_pts):
        xy_chunk = XY[:, i:i+chunk_pts, :]
        psi = model.trunk(xy_chunk)                            # (B,nPts,p)
        preds.append(torch.einsum("bp,bnp->bn", phi, psi) + model.trunk.bias)
    u_flat = torch.cat(preds, dim=1)                           # (B,H*W)
    return u_flat.view(B, H, W)

# --------- Single-row plotting (paper-friendly) ----------
def plot_row(a_1hw, u_true_hw, u_pred_hw, sample_id=None,
             same_scale_tp=True, savepath=None, use_extent=True):
    """
    a_1hw: (1,H,W), u_true_hw: (H,W), u_pred_hw: (H,W)
    Lays out [Input | Ground Truth | Approximation | Error] horizontally, paper-friendly.
    Each panel gets its own RIGHT-SIDE colorbar; Truth & Pred share vmin/vmax (separate bars).
    """
    a_img   = a_1hw.squeeze(0).cpu().numpy()
    u_true  = u_true_hw.cpu().numpy()
    u_pred  = u_pred_hw.cpu().numpy()
    diff    = u_pred - u_true  # error = pred - true

    # Metrics
    e = u_pred - u_true
    rel_l2 = np.linalg.norm(e.ravel()) / max(1e-12, np.linalg.norm(u_true.ravel()))
    mse    = float(np.mean(e**2))

    # Shared scale for true/pred (but separate bars)
    if same_scale_tp:
        tp_vmin = float(min(u_true.min(), u_pred.min()))
        tp_vmax = float(max(u_true.max(), u_pred.max()))
        tp_kwargs = dict(vmin=tp_vmin, vmax=tp_vmax)
    else:
        tp_kwargs = {}

    # Difference symmetric scale
    dmax = float(np.abs(diff).max()) or 1e-8
    diff_kwargs = dict(vmin=-dmax, vmax=dmax)

    # Paper-friendly size: ~two-column width, compact height
    fig_w, fig_h = 10.8, 3  # inches
    fig, axs = plt.subplots(1, 4, figsize=(fig_w, fig_h), constrained_layout=True)

    # Plot extent on [0,1]^2 (optional but nice for papers)
    extent = (0, 1, 0, 1) if use_extent else None
    common = dict(origin="lower", aspect="equal")
    if extent is not None:
        common["extent"] = extent

    # Panels
    im0 = axs[0].imshow(a_img, **common)
    axs[0].set_title("Input", fontsize=9)

    im1 = axs[1].imshow(u_true, **common, **tp_kwargs)
    axs[1].set_title("Ground Truth", fontsize=9)

    im2 = axs[2].imshow(u_pred, **common, **tp_kwargs)
    axs[2].set_title("Approximation", fontsize=9)

    im3 = axs[3].imshow(diff, **common, **diff_kwargs)
    axs[3].set_title("Error", fontsize=9)

    # Clean axes for paper
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        if extent is not None:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # --- Individual colorbars on the RIGHT of each axes (vertical, default) ---
    cbar_kw = dict(fraction=0.046, pad=0.04)
    cb0 = fig.colorbar(im0, ax=axs[0], **cbar_kw); cb0.ax.tick_params(labelsize=8)
    cb1 = fig.colorbar(im1, ax=axs[1], **cbar_kw); cb1.ax.tick_params(labelsize=8)
    cb2 = fig.colorbar(im2, ax=axs[2], **cbar_kw); cb2.ax.tick_params(labelsize=8)
    cb3 = fig.colorbar(im3, ax=axs[3], **cbar_kw); cb3.ax.tick_params(labelsize=8)

    # # Small caption on top
    # sid = "" if sample_id is None else f"sample {sample_id} | "
    # fig.suptitle(f"{sid}relL2={rel_l2:.3e}, MSE={mse:.3e}", y=1.02, fontsize=9)

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# --------- Visualize a random index from the val split ----------
@torch.no_grad()
def visualize_random(model, val_ds, device, same_scale_tp=True, save_dir=None, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, len(val_ds)))
    else:
        idx = int(np.random.randint(0, len(val_ds)))

    a_1hw, u_1hw = val_ds[idx]          # (1,H,W), (1,H,W)
    H, W = a_1hw.shape[-2], a_1hw.shape[-1]
    a_batch = a_1hw.unsqueeze(0).to(device)  # (1,1,H,W)
    u_pred = predict_full_grid(model, a_batch, H, W, device=device, chunk_pts=16384)[0]  # (H,W)

    # If your dataset returns normalized a/u, de-normalize here if needed.

    savepath = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        savepath = os.path.join(save_dir, f"viz_random_{idx:06d}.png")

    # sample_id: if val_ds was built from a subset of indices, you can store/source that id.
    sample_id = getattr(val_ds, "indices", None)
    if sample_id is not None and len(sample_id) == len(val_ds):
        sample_id = int(sample_id[idx])
    else:
        sample_id = idx

    plot_row(a_1hw, u_1hw[0], u_pred, sample_id=sample_id,
             same_scale_tp=same_scale_tp, savepath=savepath, use_extent=True)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    # ---- user settings ----
    h5_path   = "Datasets/threshold_dataset_256_1000.h5"
    ckpt_path = "deeponet_mse_resume_best.pt"
    same_scale_tp = True     # Truth & Pred share vmin/vmax (separate colorbars)
    save_dir = None          # e.g., "figs/val" to save, or None to just show
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- build val dataset (buffered) ----

    val_ds = H5FieldsDatasetBuffered(
        h5_path,
        buffer_size=1000,
        normalize_a="per_sample",
        transpose_hw=False,
        seed=999,
    )
    # ---- model (must match checkpoint) ----
    branch = DeepONetBranchCNN(
        in_ch=1,
        latent_dim=128,      # p; must match trunk p
        periodic=True,
        use_se=True,
        use_spp=True,
        add_fft_lowmodes=4,
        drop_dc=True
    )
    p = branch.proj[-1].out_features
    trunk = Trunk(in_dim=2, p=p, hidden=128)
    model = DeepONet(branch, trunk).to(device)
    # ---- load checkpoint ----
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    # ---- visualize one RANDOM sample from the val split ----
    visualize_random(model, val_ds, device, same_scale_tp=same_scale_tp, save_dir=save_dir)
