import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DeepONet2 import (DeepONet, DeepONetBranchCNN, Trunk)
from data_reformat import H5FieldsDatasetBuffered

def relative_rmse(u_pred: torch.Tensor, u_true: torch.Tensor, *, eps=1e-12) -> float:
    
    num_mse = F.mse_loss(u_pred, u_true, reduction="mean")
    den_mse = u_true.pow(2).mean().clamp_min(eps)

    return float(torch.sqrt(num_mse / den_mse).item())

def avg_rel_mse(model, loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    H, W = 256, 256
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1).unsqueeze(0)  # (1, H*W, 2)

    rel_vals = []

    for a, u in loader:
        a = a.to(device)
        u = u.to(device)

        pred = model(a, XY)
        pred = pred[0].view(H,W)
        u_true = u[0,0]
        
        rel_vals.append(relative_rmse(pred, u_true))

    return float(np.mean(rel_vals))

if __name__ == "__main__":
    # Model
    ckpt_path = "deeponet_mse_resume_best.pt"
    h5_ds_path = "Datasets/threshold_dataset_256_1000.h5"

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
    model = DeepONet(branch, trunk)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    rel_err_ds = H5FieldsDatasetBuffered(
        h5_ds_path,
        buffer_size=1000,
        normalize_a="per_sample",
        transpose_hw=False,
        seed=999,
    )
    loader = DataLoader(rel_err_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    avg = avg_rel_mse(model, loader)
    print(f"Average relative RMSE over 1000 samples: {avg:.6e}")
