import h5py, torch
from torch.utils.data import DataLoader, TensorDataset
from neural_net import FullyConnectedNetwork

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

def central_diff_grad2d(U, hx, hy):
    """
    U:  (B,1,H,W) tensor
    hx, hy: spacings (floats)
    Returns central differences on the interior: shapes (B,1,H-2,W-2)
    """
    Ux = (U[:, :, 1:-1, 2:] - U[:, :, 1:-1, :-2]) / (2.0 * hx)
    Uy = (U[:, :, 2:, 1:-1] - U[:, :, :-2, 1:-1]) / (2.0 * hy)
    return Ux, Uy

def loss_H10_grid(u_pred, u_true, hx, hy, reduction="mean"):
    """
    u_pred, u_true: (B,1,H,W) on the same grid.
    reduction: "mean" ~ average |∇e|^2,
               "sum"  ~ discrete integral ≈ ∑ |∇e|^2 * hx*hy,
               None   ~ returns per-pixel |∇e|^2 map (B,1,H-2,W-2)
    """
    Upx, Upy = central_diff_grad2d(u_pred, hx, hy)
    Utx, Uty = central_diff_grad2d(u_true, hx, hy)
    grad_err2 = (Upx - Utx)**2 + (Upy - Uty)**2  # (B,1,H-2,W-2)

    if reduction == "mean":
        return grad_err2.mean()
    if reduction == "sum":
        return grad_err2.sum() * hx * hy
    return grad_err2

def _ensure_4d(t):
    """Make tensor shape (B,1,H,W). Accepts (B,H,W) or (B,1,H,W)."""
    if t.dim() == 3:
        return t.unsqueeze(1)
    if t.dim() == 4:
        return t
    raise ValueError(f"Expected 3D or 4D tensor, got {t.shape}")


@torch.no_grad()
def _h1_norm_squared(u, hx, hy):
    """
    Returns per-sample ||u||_{H^1}^2 ≈ ∑_{interior} (|u|^2 + |∇u|^2) * hx*hy, shape (B,)
    Uses interior grid points to match central differences.
    """
    u = _ensure_4d(u)
    Ux, Uy = central_diff_grad2d(u, hx, hy)          # (B,1,H-2,W-2)
    Uint = u[:, :, 1:-1, 1:-1]                       # (B,1,H-2,W-2)
    return (Uint.pow(2).sum(dim=(1,2,3)) + (Ux.pow(2) + Uy.pow(2)).sum(dim=(1,2,3))) * (hx * hy)

@torch.no_grad()
def h10_norm(u, hx, hy, reduction="sum"):
    """
    H_0^1 seminorm of u on the grid:
    returns sqrt( ∑ |∇u|^2 * hx*hy ) if reduction='sum'
    or sqrt( mean(|∇u|^2) ) if reduction='mean'
    """
    u = _ensure_4d(u)
    Ux, Uy = central_diff_grad2d(u, hx, hy)
    g2 = Ux**2 + Uy**2
    if reduction == "sum":
        return torch.sqrt(g2.sum(dim=(1,2,3)) * hx * hy)  # (B,)
    elif reduction == "mean":
        return torch.sqrt(g2.mean(dim=(1,2,3)))           # (B,)
    else:
        raise ValueError("reduction must be 'sum' or 'mean'")

@torch.no_grad()
def evaluate_relative_H10_error(model, data_path, n_train, batch_size=64, device="cuda", last_N=300):
    """
    Loads data, takes the last `last_N` samples as the test set (default 300),
    runs the model, and returns the Monte Carlo estimate of the expected
    relative H^1 error:
        mean_i sqrt( ||e_i||_{H^1}^2 / ||u_i||_{H^1}^2 )
    Also reports standard error and a 95% CI of the mean under i.i.d. sampling.
    (Function name kept for compatibility.)
    """
    # ---- load ----
    with h5py.File(data_path, "r") as f:
        N = min(n_train, f["data/a"].shape[0], f["data/u"].shape[0])
        X = torch.from_numpy(f["data/a"][:N]).float()
        Y = torch.from_numpy(f["data/u"][:N]).float()

    # ---- use exactly the last `last_N` samples ----
    # guard & clamp
    if last_N is None:
        val_N = N
    else:
        val_N = int(last_N)
        if val_N <= 0:
            val_N = N
        val_N = min(val_N, N)

    X_val = X[-val_N:]
    Y_val = Y[-val_N:]

    # ---- infer grid + spacings from Y ----
    y0 = Y_val[0]
    if y0.dim() == 2:
        H, W = y0.shape
    elif y0.dim() == 3:
        _, H, W = y0.shape
    else:
        raise ValueError(f"Unexpected Y shape: {Y_val.shape}")
    hx = 1.0 / (H - 1)
    hy = 1.0 / (W - 1)

    # ---- data loader ----
    val_ds = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device).eval()

    rel_errors = []

    for a_batch, u_true_batch in val_loader:
        a_batch = a_batch.to(device, non_blocking=True)
        u_true_batch = u_true_batch.to(device, non_blocking=True)

        u_pred_batch = model(a_batch)

        u_true_batch = _ensure_4d(u_true_batch)
        u_pred_batch = _ensure_4d(u_pred_batch)

        # ---- Monte Carlo per-sample relative H1 error ----
        num_sq = _h1_norm_squared(u_pred_batch - u_true_batch, hx, hy)  # (B,)
        den_sq = _h1_norm_squared(u_true_batch, hx, hy)                 # (B,)
        eps = 1e-12
        rel = torch.sqrt(num_sq / den_sq)         # (B,)
        rel_errors.append(rel.detach().cpu())

    rel_errors = torch.cat(rel_errors, dim=0)  # (val_N,)
    mean_rel = rel_errors.mean()
    # sample std, standard error, and 95% CI
    std_rel = rel_errors.std(unbiased=True)
    se = std_rel / (rel_errors.numel() ** 0.5)

    summary = {
        "num_samples": int(rel_errors.numel()),
        "mean_rel_H1": float(mean_rel.item()),
        "std_rel_H1_unbiased": float(std_rel.item()),
        "se_of_mean": float(se.item())
    }
    return summary, rel_errors



if __name__ == "__main__":
    # Load model
    path_ckpt = "pca_net_threshold_best.pt"
    # Number of datapoints in training dataset
    n_train = 4000
    n_basis = 64
    path_data = "Datasets/threshold_dataset_256_4000.h5"
    last_n_samples = 3000

    # Load model from checkpoint
    model = FullyConnectedNetwork(n_basis = n_basis)
    ckpt = torch.load(path_ckpt, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt,dict) and "model_state" in ckpt else ckpt
    _realloc_register_buffers_from_state(model,state)
    model.load_state_dict(state, strict=False)
    summary, rel = evaluate_relative_H10_error(model, path_data, n_train=n_train, last_N=last_n_samples)
    print(summary)


