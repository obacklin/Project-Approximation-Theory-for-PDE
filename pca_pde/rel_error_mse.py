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

def _ensure_4d(t):
    if t.dim() == 3:
        return t.unsqueeze(1)
    if t.dim() == 4:
        return t
    raise ValueError(f"Expected 3D or 4D tensor, got {t.shape}")

@torch.no_grad()
def _per_sample_mse(u):
    """
    u: (B,1,H,W) or (B,H,W)
    returns: tensor of shape (B,) with MSE over all pixels per sample
    """
    u = _ensure_4d(u)
    # mean over channel+spatial dims → (B,)
    return u.pow(2).mean(dim=(1,2,3))

@torch.no_grad()
def evaluate_relative_MSE_error(model, data_path, n_train, batch_size=64, device="cuda", last_N=300):
    """
    Monte Carlo estimate of the expected relative error using MSE:
        mean_i sqrt( MSE(e_i) / MSE(u_i) )
    where e_i = u_pred_i - u_true_i. (This is relative RMSE.)
    Matches the data loading/batching behavior of evaluate_relative_H10_error.
    """
    # ---- load ----
    with h5py.File(data_path, "r") as f:
        N = min(n_train, f["data/a"].shape[0], f["data/u"].shape[0])
        X = torch.from_numpy(f["data/a"][:N]).float()
        Y = torch.from_numpy(f["data/u"][:N]).float()

    # ---- take last N samples as validation ----
    val_N = N if last_N is None else max(1, min(int(last_N), N))
    X_val = X[-val_N:]
    Y_val = Y[-val_N:]

    # ---- loader ----
    val_ds = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device).eval()

    rel_errors = []
    eps = 1e-12

    for a_batch, u_true_batch in val_loader:
        a_batch = a_batch.to(device, non_blocking=True)
        u_true_batch = u_true_batch.to(device, non_blocking=True)

        u_pred_batch = model(a_batch)

        u_true_batch = _ensure_4d(u_true_batch)
        u_pred_batch = _ensure_4d(u_pred_batch)

        # ---- per-sample relative RMSE ----
        mse_num = _per_sample_mse(u_pred_batch - u_true_batch)   # (B,)
        mse_den = _per_sample_mse(u_true_batch)                   # (B,)
        rel = torch.sqrt(mse_num / (mse_den + eps))               # (B,)
        rel_errors.append(rel.detach().cpu())

    rel_errors = torch.cat(rel_errors, dim=0)  # (val_N,)
    mean_rel = rel_errors.mean()
    std_rel  = rel_errors.std(unbiased=True)
    se = std_rel / (rel_errors.numel() ** 0.5)

    summary = {
        "num_samples": int(rel_errors.numel()),
        "mean_rel_RMSE": float(mean_rel.item()),
        "std_rel_RMSE_unbiased": float(std_rel.item()),
        "se_of_mean": float(se.item())
    }
    summary = f"num_samples: {int(rel_errors.numel())} | mean_rel_RMSE {float(mean_rel.item()):6e}"
    return summary, rel_errors

if __name__ == "__main__":
        # Load model
    path_ckpt = "pca_net_treshold_best.pt"
    # Number of datapoints in training dataset
    n_train = 4000
    n_basis = 70
    path_data = "Datasets/threshold_dataset_256_4000.h5"
    last_n_samples = 3000-24

    # Load model from checkpoint
    model = FullyConnectedNetwork(n_basis = 70)
    ckpt = torch.load(path_ckpt, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt,dict) and "model_state" in ckpt else ckpt
    _realloc_register_buffers_from_state(model,state)
    model.load_state_dict(state, strict=False)
    summary, rel = evaluate_relative_MSE_error(model, path_data, n_train=n_train, last_N=last_n_samples)
    print(summary)