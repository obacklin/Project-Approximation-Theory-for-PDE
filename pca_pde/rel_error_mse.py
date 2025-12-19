import h5py, torch
from torch.utils.data import DataLoader, TensorDataset
from neural_net import PCANet

def ensure_mu_from_phi(model):
    # Old checkpoints: mu buffers are empty; for non-centered PCA they should be zeros.
    if getattr(model, "mu_a", None) is not None and model.mu_a.numel() == 0 and model.Phi_a.numel() > 0:
        HW = model.Phi_a.shape[0]
        model.mu_a = torch.zeros((HW,), dtype=model.Phi_a.dtype, device=model.Phi_a.device)

    if getattr(model, "mu_u", None) is not None and model.mu_u.numel() == 0 and model.Phi_u.numel() > 0:
        HW = model.Phi_u.shape[0]
        model.mu_u = torch.zeros((HW,), dtype=model.Phi_u.dtype, device=model.Phi_u.device)


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

@torch.no_grad()
def _per_sample_mse(u):
    """
    u: (B,1,H,W) or (B,H,W)
    returns: tensor of shape (B,) with MSE over all pixels per sample
    """
    # mean over channel+spatial dims → (B,)
    return u.pow(2).mean(dim=(1,2,3))

@torch.no_grad()
def evaluate_relative_MSE_error(model, data_path, batch_size=64, device="cuda"):
    """
    Monte Carlo estimate of the expected relative error using MSE:
        mean_i sqrt( MSE(e_i) / MSE(u_i) )
    where e_i = u_pred_i - u_true_i. (This is relative RMSE.)
    Matches the data loading/batching behavior of evaluate_relative_H10_error.
    """
    # ---- load ----
    with h5py.File(data_path, "r") as f:
        N = f["data/u"].shape[0]
        X_val = torch.from_numpy(f["data/a"][:N]).float()
        Y_val = torch.from_numpy(f["data/u"][:N]).float()
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
    path_ckpt = "pca_net_lognorm_best.pt"
    # Number of datapoints in training dataset
    n_basis = 70
    path_data = "Datasets/lognormal_dataset_test_nodes257_2500.h5"
    # Load model from checkpoint
    model = PCANet(n_basis = n_basis, N=257)
    ckpt = torch.load(path_ckpt, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt,dict) and "model_state" in ckpt else ckpt
    _realloc_register_buffers_from_state(model,state)
    model.load_state_dict(state, strict=False)
    ensure_mu_from_phi(model)
    summary, rel = evaluate_relative_MSE_error(model, path_data)
    print(summary)