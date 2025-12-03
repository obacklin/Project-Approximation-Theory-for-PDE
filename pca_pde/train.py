import h5py
import torch
from neural_net import FullyConnectedNetwork
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

def loss_mse(u_pred, u_true):
    return F.mse_loss(u_pred, u_true)

def central_diff_grad2d(U, hx, hy):
    """
    U:  (B,1,H,W) tensor
    hx, hy: spacings (floats)
    Returns central differences on the interior: shapes (B,1,H-2,W-2)
    """
    Ux = (U[:, :, 1:-1, 2:] - U[:, :, 1:-1, :-2]) / (2.0 * hx)
    Uy = (U[:, :, 2:, 1:-1] - U[:, :, :-2, 1:-1]) / (2.0 * hy)
    return Ux, Uy

def loss_H1_grid(u_pred, u_true, hx, hy, crop_to_interior = True):
    """
    Integral approximation
    u_pred, u_true: (B,1,H,W)
    crop_to_interior: if True, use interior [1:-1,1:-1] for the function term
                      so it matches gradient shapes. If False, use full grid for L2.
    """
    # gradient term on interior
    Upx, Upy = central_diff_grad2d(u_pred, hx, hy)
    Utx, Uty = central_diff_grad2d(u_true, hx, hy)
    grad_err2 = (Upx - Utx)**2 + (Upy - Uty)**2          # (B,1,H-2,W-2)

    if crop_to_interior:
        e2 = (u_pred[:, :, 1:-1, 1:-1] - u_true[:, :, 1:-1, 1:-1]) ** 2  # (B,1,H-2,W-2)
    else:
        e2 = (u_pred - u_true) ** 2  # (B,1,H,W)

    total = e2.sum() * hx * hy + grad_err2.sum() * hx * hy

    return total

def train(model, loader, epochs=20, lr=1e-3, ckpt_path="pca_net_best.pt", device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Precompute PCA bases on CPU
    model_cpu = model.to("cpu")
    # Use the same dataset the loader wraps (TensorDataset of (X, Y))
    dataset = loader.dataset
    X, Y = dataset.tensors
    if X.ndim == 3: dataset.tensors = (X.unsqueeze(1), Y.unsqueeze(1))
    N = len(dataset)
    model_cpu.init_pca_basis(dataset, N=N, fortran_order=True)  # computes Phi_a, Phi_u
    model = model_cpu.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=epochs, eta_min=max(1e-6, lr*1e-2)
)

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        tot, count = 0.0, 0
        for a, u in loader:
            # Ensure shape (B,1,H,W)
            if a.ndim == 3: a = a.unsqueeze(1)
            if u.ndim == 3: u = u.unsqueeze(1)

            a = a.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)
            _, _, H, W = u.shape
            hx = 1.0 / (W - 1)
            hy = 1.0 / (H - 1)

            opt.zero_grad(set_to_none=True)
            u_pred = model(a)
            loss = loss_mse(u_pred, u) 
            loss.backward()
            opt.step()

            batch_size = a.size(0)
            tot += loss.item() * batch_size
            count += batch_size

        avg = tot / max(1, count)
        current_lr = opt.param_groups[0]['lr']
        print(f"epoch {ep:03d} | lr={current_lr:.3e} | train_H10={avg:.6e}")
        # save best
        if avg < best:
            best = avg
            torch.save(
                {"model_state": model.state_dict(), "epoch": ep, "train_mse": best},
                ckpt_path
            )
        sched.step()
    print(f"best train loss: {best:.4e} (saved to {ckpt_path})")

if __name__ == "__main__":
    import time
    n_basis = 70
    pca_net = FullyConnectedNetwork(n_basis=n_basis)
    epochs = 2500
    # Load training dataset.
    n_train = 1024
    data_path = "Datasets/threshold_dataset_256_4000.h5"
    ckpt_path="pca_net_treshold_best.pt"
    with h5py.File(data_path, "r") as f:
        N = min(n_train, f["data/a"].shape[0], f["data/u"].shape[0])
        X = torch.from_numpy(f["data/a"][:N]).float()  # only first N
        Y = torch.from_numpy(f["data/u"][:N]).float()
    data = TensorDataset(X, Y)
    loader = DataLoader(data, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    tic = time.perf_counter()
    train(pca_net, loader, epochs, ckpt_path=ckpt_path)
    toc = time.perf_counter()
    print(f"Time taken {toc-tic:.1f} s")