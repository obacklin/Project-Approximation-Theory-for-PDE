import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from DeepONet2 import (DeepONet, DeepONetBranchCNN, Trunk)
from data_reformat import H5FieldsDatasetBuffered
import time
import os, random


# ---------- Saving & Loading (from train2.py) ----------
def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, best_val, extra_meta=None):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),                 # next epoch to run from this value
        "best_val": float(best_val),
        "meta": extra_meta or {},
        # RNG states (for reproducible continuation)
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _to_cpu_bytetensor(x):
    """Coerce x into a CPU torch.ByteTensor suitable for set_rng_state."""
    import numpy as np
    if isinstance(x, torch.ByteTensor) and x.device.type == "cpu":
        return x
    if isinstance(x, torch.Tensor):
        return x.to(device="cpu", dtype=torch.uint8)
    if isinstance(x, (bytes, bytearray)):
        return torch.tensor(list(x), dtype=torch.uint8)
    if isinstance(x, np.ndarray):
        # avoid copy when already uint8
        return torch.from_numpy(x.astype(np.uint8, copy=False))
    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.uint8)
    raise TypeError(f"Cannot convert RNG state of type {type(x)} to ByteTensor")


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None,
                    map_location="cpu", strict=True):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    # --- RNG states (robust restore) ---
    rng = ckpt.get("rng_state", {}) or {}

    try:
        t_state = rng.get("torch")
        if t_state is not None:
            torch.set_rng_state(_to_cpu_bytetensor(t_state))
    except Exception as e:
        print(f"[Resume] Skipping torch RNG restore: {e}")

    try:
        cuda_state = rng.get("cuda")
        if torch.cuda.is_available() and cuda_state is not None:
            # Accept either a single tensor or a list of tensors/arrays
            if isinstance(cuda_state, (list, tuple)):
                states = [_to_cpu_bytetensor(s) for s in cuda_state]
            else:
                states = [_to_cpu_bytetensor(cuda_state)]
            torch.cuda.set_rng_state_all(states)
    except Exception as e:
        print(f"[Resume] Skipping CUDA RNG restore: {e}")

    try:
        np_state = rng.get("numpy")
        if np_state is not None:
            import numpy as np
            np.random.set_state(np_state)
    except Exception as e:
        print(f"[Resume] Skipping NumPy RNG restore: {e}")

    try:
        py_state = rng.get("python")
        if py_state is not None:
            import random
            random.setstate(py_state)
    except Exception as e:
        print(f"[Resume] Skipping Python RNG restore: {e}")

    next_epoch = int(ckpt.get("epoch", 0))
    best_val   = float(ckpt.get("best_val", float("inf")))
    meta       = ckpt.get("meta", {})
    return next_epoch, best_val, meta


# ---------- Pixel sampling & gather (from train.py; no gradients) ----------
def sample_random_pixels(B, H, W, Np, device):
    """
    Returns:
      xy : (B,Np,2) with normalized coords in [0,1]^2
           x = j/(W-1) (columns), y = i/(H-1) (rows)
      iy, ix : (B,Np) integer indices for gather
    """
    iy = torch.randint(0, H, (B, Np), device=device)
    ix = torch.randint(0, W, (B, Np), device=device)
    x = ix.to(torch.float32) / (W - 1)
    y = iy.to(torch.float32) / (H - 1)
    xy = torch.stack([x, y], dim=-1)  # (B,Np,2)
    return xy, iy, ix


def gather_u(u, iy, ix):
    """
    u:  (B,1,H,W) ; iy,ix: (B,Np)
    ->  (B,Np)
    """
    B = u.shape[0]
    bidx = torch.arange(B, device=u.device).unsqueeze(1)  # (B,1)
    return u[bidx, 0, iy, ix]


# ---------- Train / Eval (from train.py; AMP + GradScaler; pure MSE) ----------
def train_one_epoch(model, loader, opt, scaler, device, Np=4096, clip_grad=1.0):
    model.train()
    mse_sum, n_items = 0.0, 0
    for a, u in loader:  # a,u: (B,1,H,W)
        a = a.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)
        B, _, H, W = u.shape

        xy, iy, ix = sample_random_pixels(B, H, W, Np, device)
        target = gather_u(u, iy, ix)  # (B,Np)

        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            pred = model(a, xy)           # (B,Np)
            loss = F.mse_loss(pred, target)

        scaler.scale(loss).backward()

        if clip_grad is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        scaler.step(opt)
        scaler.update()
        mse_sum += loss.detach().item() * B
        n_items += B

    return mse_sum / max(n_items, 1)


@torch.no_grad()
def evaluate(model, loader, device, Np_eval=8192):
    model.eval()
    mse_sum, n_items = 0.0, 0
    for a, u in loader:
        a = a.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)
        B, _, H, W = u.shape

        xy, iy, ix = sample_random_pixels(B, H, W, Np_eval, device)
        target = gather_u(u, iy, ix)
        pred = model(a, xy)
        loss = F.mse_loss(pred, target)

        mse_sum += loss.item() * B
        n_items += B
    return mse_sum / max(n_items, 1)


# ---------- Runner (train2 features + pure MSE loop) ----------
def run_training(
    branch,
    trunk,
    train_loader, val_loader,
    epochs=50,
    Np=4096,
    lr=3e-4,
    weight_decay=1e-4,
    device="cuda",
    ckpt_path="deeponet_fftbranch_best.pt",
    resume_from=None,           # path to a checkpoint to resume from
    strict_resume=True,         # strict model key checking
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # infer latent size p from the branch (your logic)
    if hasattr(branch, "proj") and isinstance(branch.proj[-1], nn.Linear):
        p = branch.proj[-1].out_features
    else:
        p = 256  # fallback like before

    model = DeepONet(branch, trunk).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -------------------------------
    # Cosine annealing with warm restarts
    # First cycle ~20% of total epochs, then doubles: 10, 20, 40, ...
    # eta_min mimics your previous min_lr.
    # -------------------------------
    first_cycle = max(10, epochs // 5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=first_cycle, T_mult=2, eta_min=1e-6
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    start_epoch = 1

    # --- Optional resume ---
    if resume_from is not None and os.path.exists(resume_from):
        start_epoch, best_val, meta = load_checkpoint(
            resume_from,
            model=model,
            optimizer=opt,
            scaler=scaler,
            scheduler=sched,   # loads the cosine scheduler state if present
            map_location=device,
            strict=strict_resume,
        )
        print(f"[Resume] Loaded '{resume_from}' -> next_epoch={start_epoch}, best_val={best_val:.6e}")
        if "p" in meta and meta["p"] != p:
            print(f"[Resume] WARNING: trunk/branch latent dim mismatch (ckpt p={meta['p']}, current p={p}).")

        # If the checkpoint didn't have a compatible scheduler state (e.g., old ReduceLROnPlateau),
        # the loader may skip it. In that case, align the cosine scheduler position by fast-forwarding:
        if getattr(sched, "_last_lr", None) is None:
            for _ in range(start_epoch - 1):
                sched.step()

    # derive a "last" path from best path
    ckpt_last = ckpt_path.replace(".pt", "_last.pt")

    for ep in range(start_epoch, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(ep)  # dataset method, if present

        tic = time.perf_counter()
        train_mse = train_one_epoch(model, train_loader, opt, scaler, device, Np=Np, clip_grad=1.0)
        t_train = time.perf_counter() - tic

        tic = time.perf_counter()
        val_mse = evaluate(model, val_loader, device, Np_eval=min(2*Np, 32768))
        t_val = time.perf_counter() - tic

        # ---- Cosine step: no metric needed; step once per epoch
        sched.step()

        # LR readout AFTER stepping
        curr_lr = opt.param_groups[0]["lr"]

        print(f"[Epoch {ep:03d}] train MSE={train_mse:.6e} | val MSE={val_mse:.6e} | lr={curr_lr:.2e} | "
              f"time train={t_train:.2f}s | time val={t_val:.2f}s")

        # --- Save rolling 'last' checkpoint ---
        save_checkpoint(
            ckpt_last,
            model, opt, scaler, sched,
            epoch=ep+1,                          # next epoch to run
            best_val=best_val,
            extra_meta={"epoch": ep, "val_mse": float(val_mse), "p": p},
        )

        # --- Save "best" checkpoint ---
        if val_mse < best_val:
            best_val = val_mse
            save_checkpoint(
                ckpt_path,
                model, opt, scaler, sched,
                epoch=ep+1,
                best_val=best_val,
                extra_meta={"epoch": ep, "val_mse": float(val_mse), "p": p},
            )

    return model

# -------------------- MAIN --------------------
if __name__ == "__main__":
    # Adjust to your dataset path
    h5_path = "Datasets/threshold_dataset_256_4000.h5"

    with h5py.File(h5_path, "r") as f:
        N_total = f["data/a"].shape[0]

    n_val = 1000
    train_idx = np.arange(0, N_total - n_val, dtype=np.int64)
    val_idx   = np.arange(N_total - n_val, N_total, dtype=np.int64)

    # Buffered dataset
    train_ds = H5FieldsDatasetBuffered(
        h5_path, indices=train_idx,
        buffer_size=3000,
        normalize_a="per_sample",
        transpose_hw=False,
        seed=123,
    )
    val_ds = H5FieldsDatasetBuffered(
        h5_path, indices=val_idx,
        buffer_size=1000,
        normalize_a="per_sample",
        transpose_hw=False,
        seed=999,
    )

    # Keep shuffle=False to preserve contiguous reads; dataset handles block permutation
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, pin_memory=True, num_workers=0)

    # Model
    branch = DeepONetBranchCNN(
        in_ch=1,
        latent_dim=128,      # p; must match trunk p
        periodic=True,       # circular padding if torus
        use_se=True,
        use_spp=True,        # multi-scale pooling
        add_fft_lowmodes=4,  # try 4 or 8
        drop_dc=True
    )
    trunk = Trunk(in_dim=2, hidden=128, p=128)

    # ---- Train ----
    model = run_training(
        branch=branch,
        trunk=trunk,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1000,
        Np=16384,
        lr=3e-4,
        weight_decay=1e-4,
        device="cuda",
        ckpt_path="deeponet_mse_resume_best.pt",
        # To resume, set one of these:
        # resume_from="deeponet_mse_resume_best.pt",
        resume_from="deeponet_mse_resume_best_last cont.pt",
        strict_resume=True)
