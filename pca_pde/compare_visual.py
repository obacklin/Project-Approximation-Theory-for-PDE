# compare_sample.py
import h5py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import numpy as np

# --- import your model class (has .init_pca_basis and .forward) ---
from neural_net import FullyConnectedNetwork  # <-- change if your module path is different


def _realloc_register_buffers_from_state(model: torch.nn.Module, state: dict):
    """
    For any tensor in `state` whose key points to a registered buffer in `model`,
    (and whose current buffer shape/dtype doesn't match), re-register the buffer
    on CPU with the correct shape/dtype so load_state_dict can succeed.
    """
    # Gather full buffer name set from the model (recurse=True captures nested modules)
    buffer_names = set(name for name, _ in model.named_buffers(recurse=True))
    if not buffer_names:
        return

    for k, v in state.items():
        if k not in buffer_names:
            continue

        # Split "parent.child.buffer" -> parent path and final attr
        if "." in k:
            prefix, buf_name = k.rsplit(".", 1)
            parent = model.get_submodule(prefix)  # works for top-level ("") and nested
        else:
            parent, buf_name = model, k

        # Current buffer (may be None or 0-sized depending on your init)
        current = getattr(parent, buf_name, None)

        # If there is no buffer yet or it has mismatched shape/dtype, re-register it
        needs_realloc = (
            (current is None)
            or (not isinstance(current, torch.Tensor))
            or (current.shape != v.shape)
            or (current.dtype != v.dtype)
        )

        if needs_realloc:
            # Remove any existing attribute then register a fresh buffer on CPU
            try:
                delattr(parent, buf_name)
            except Exception:
                pass
            parent.register_buffer(buf_name, torch.empty_like(v, device="cpu"))


def load_first_N(data_path, n_train):
    with h5py.File(data_path, "r") as f:
        N = min(n_train, f["data/a"].shape[0], f["data/u"].shape[0])
        X = torch.from_numpy(f["data/a"][:N]).float()  # (N,1,H,W) or (N,H,W)
        Y = torch.from_numpy(f["data/u"][:N]).float()
    # make sure we have (N,1,H,W)
    if X.ndim == 3: X = X.unsqueeze(1)
    if Y.ndim == 3: Y = Y.unsqueeze(1)
    return X, Y


def pick_random_sample(X, Y):
    idx = torch.randint(X.shape[0], (1,)).item()
    a = X[idx]  # (1,H,W)
    u = Y[idx]  # (1,H,W)
    return idx, a, u


def predict(model, a, device):
    model.eval()
    with torch.no_grad():
        a_in = a.unsqueeze(0).to(device)   # (1,1,H,W)
        u_pred = model(a_in)               # (1,1,H,W)
        u_pred = u_pred.squeeze(0).cpu()   # (1,H,W)
    return u_pred


def plot_three(a, u_true, u_pred, title="", origin="lower", savefig=None):
    A = a.squeeze(0).cpu().numpy()         # (H,W)
    U = u_true.squeeze(0).cpu().numpy()
    P = u_pred.squeeze(0).cpu().numpy()
    D = P - U                              # difference (pred - true)

    # Difference symmetric scale
    dmax = float(np.abs(D).max()) or 1e-8
    diff_kwargs = dict(vmin=-dmax, vmax=dmax)

    tp_vmin = float(min(u_true.min(), u_pred.min()))
    tp_vmax = float(max(u_true.max(), u_pred.max()))
    tp_kwargs = dict(vmin=tp_vmin, vmax=tp_vmax)

    fig_w, fig_h = 10.8, 3  # inches
    fig, axs = plt.subplots(1, 4, figsize=(fig_w, fig_h), constrained_layout=True)

    extent = (0, 1, 0, 1)
    common = dict(origin="lower", aspect="equal")
    common["extent"] = extent

    im0 = axs[0].imshow(A, **common)   # independent scale
    axs[0].set_title("Input", fontsize=9)

    im1 = axs[1].imshow(U, **common, **tp_kwargs)   # independent scale
    axs[1].set_title("Ground Truth", fontsize=9)

    im2 = axs[2].imshow(P, **common, **tp_kwargs)   # independent scale
    axs[2].set_title("Approximation", fontsize=9)
    
    im3 = axs[3].imshow(D, **common, **diff_kwargs)
    axs[3].set_title("Error", fontsize=9)
    
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

    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def main(
    ckpt_path="pca_net_best.pt",
    data_path="Datasets/logexp_dataset.h5",
    n_train=1000,
    n_basis=64,
    device=None,
    savefig=None
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1) Load data (first N)
    X, Y = load_first_N(data_path, n_train)
    H, W = X.shape[-2], X.shape[-1]

    # 2) Build model and load checkpoint
    model = FullyConnectedNetwork(n_basis=n_basis, device="cpu")  # start on CPU
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    # >>> NEW: make sure buffers (e.g., Phi_a, Phi_u, etc.) have correct shapes first
    _realloc_register_buffers_from_state(model, state)

    # Now load weights/buffers. strict=False tolerated if there are extra meta keys.
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    # If bases weren't in the checkpoint (or left empty), compute once from the dataset
    if getattr(model, "Phi_a", None) is None or model.Phi_a.numel() == 0 \
       or getattr(model, "Phi_u", None) is None or model.Phi_u.numel() == 0:
        print("PCA bases missing in checkpoint; computing from dataset...")
        ds = TensorDataset(X, Y)                 # (N,1,H,W)
        m_cpu = model.to("cpu")
        m_cpu.init_pca_basis(ds, N=len(ds), fortran_order=True)
        model = m_cpu.to(device)

    # 3) Pick a random sample and predict
    idx, a, u = pick_random_sample(X, Y)        # (1,H,W) each
    u_pred = predict(model, a, device)          # (1,H,W)

    # 4) Plot a, u, u'
    plot_three(a, u, u_pred, title=f"sample idx = {idx}", savefig=savefig)

    # optional: print MSE for this sample
    mse = torch.mean((u_pred - u) ** 2).item()
    print(f"sample {idx} MSE: {mse:.6e}")


if __name__ == "__main__":
    # tweak paths/params as needed
    ckpt_path="pca_net_treshold_best.pt"
    data_path="Datasets/threshold_dataset_256_1000.h5"
    main(
        ckpt_path=ckpt_path,
        data_path=data_path,
        n_train=1000,
        n_basis=70,
        device="cuda",
        savefig=None
    )
