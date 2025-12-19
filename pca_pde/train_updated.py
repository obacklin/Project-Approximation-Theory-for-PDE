import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from neural_net import PCANet 
import time

def loss_mse(u_pred, u_true):
    return F.mse_loss(u_pred, u_true)

def loss_relative_mse(u_pred: torch.Tensor, u_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Mean over batch of (||e||_2^2 / (||u||_2^2 + eps)).
    Shapes: (B,1,H,W) or (B,H,W) etc.
    """
    e = u_pred - u_true
    num = e.flatten(1).pow(2).sum(dim=1)                    # (B,)
    den = u_true.flatten(1).pow(2).sum(dim=1).clamp_min(eps) # (B,)
    return (num / den).mean()

@torch.no_grad()
def _evaluate_loss(model, loader, device, loss_fn):
    model.eval()
    tot, count = 0.0, 0
    for a, u in loader:

        a = a.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)

        u_pred = model(a)
        loss = loss_fn(u_pred, u)

        bs = a.size(0)
        tot += loss.item() * bs
        count += bs
    return tot / max(1, count)

def train(
    model,
    train_loader,
    val_loader,
    epochs=1250,
    lr=3e-4,                      # treat as max_lr for OneCycle
    ckpt_path="pca_net_best.pt",
    device=None,
    pca_N=None,
    weight_decay=1e-5,
    clip_grad_norm=1.0,
    pct_start=0.05,
    div_factor=25.0,
    final_div_factor=1e4,
    loss_fn=loss_relative_mse
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


    # PCA basis on CPU from training data -----
    model_cpu = model.to("cpu")
    train_ds = train_loader.dataset
    Xtr, Ytr = train_ds.tensors

    pca_ds = TensorDataset(Xtr, Ytr)
    Npca = len(pca_ds) if pca_N is None else min(int(pca_N), len(pca_ds))
    model_cpu.init_pca_basis(pca_ds, N=Npca)  # centered PCA
    model = model_cpu.to(device)

    #  Optimizer + scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        anneal_strategy="cos",
    )

    best_val = float("inf")
    best_train = float("inf")
    best_epoch = -1

    for ep in range(1, epochs + 1):
        tic = time.perf_counter()
        model.train()
        tot, count = 0.0, 0

        for a, u in train_loader:
            if a.ndim == 3: a = a.unsqueeze(1)
            if u.ndim == 3: u = u.unsqueeze(1)

            a = a.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            u_pred = model(a)
            loss = loss_fn(u_pred, u)
            loss.backward()

            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            opt.step()
            sched.step()

            bs = a.size(0)
            tot += loss.item() * bs
            count += bs

        train_mse = tot / max(1, count)
        val_mse = _evaluate_loss(model, val_loader, device, loss_fn)

        current_lr = opt.param_groups[0]["lr"]
        print(
            f"epoch {ep:03d} | lr={current_lr:.3e} | "
            f"train_mse={train_mse:.6e} | val_mse={val_mse:.6e}"
        )

        # Save best-by-validation
        if val_mse < best_val:
            best_val = val_mse
            best_train = train_mse
            best_epoch = ep
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": ep,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "lr_max": lr,
                    "weight_decay": weight_decay,
                    "clip_grad_norm": clip_grad_norm,
                },
                ckpt_path,
            )
            print(f"New best: {best_val:.4e} at epoch {ep} (saved to {ckpt_path})")

        toc = time.perf_counter()
        print(f"Time taken {toc - tic:.1f} s")

    print(
        f"best val loss: {best_val:.4e} (train at best: {best_train:.4e}) "
        f"at epoch {best_epoch} saved to {ckpt_path}"
    )

if __name__ == "__main__":
    n_basis = 70
    epochs = 500
    n_train = 1024
    data_path = "Datasets/lognormal_dataset_train_nodes257_1025.h5"
    ckpt_path = "pca_net_lognorm_best.pt"
    batch_size = 128

    with h5py.File(data_path, "r") as f:
        N = min(n_train, f["data/a"].shape[0], f["data/u"].shape[0])
        X = torch.from_numpy(f["data/a"][:N]).float()  # (N,1,H,W) expected
        Y = torch.from_numpy(f["data/u"][:N]).float()

    data = TensorDataset(X, Y)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_h5_path = "Datasets/lognormal_dataset_val_nodes257_200.h5"
    val_N = 200,
    val_batch_size = 200

    with h5py.File(val_h5_path, "r") as f:
        Nv = f["data/a"].shape[0]
        Xa = torch.from_numpy(f["data/a"][:Nv]).float()
        Yu = torch.from_numpy(f["data/u"][:Nv]).float()

    val_ds = TensorDataset(Xa.contiguous(), Yu.contiguous())
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    pca_net = PCANet(n_basis=n_basis, N=257)
    
    train(
    pca_net,
    train_loader=loader,
    val_loader=val_loader,
    epochs=epochs,
    ckpt_path=ckpt_path,
    pca_N = n_basis
    )
