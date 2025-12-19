import torch
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from deeponet_cnn import(DeepONet, BranchCNN, Trunk)

import h5py
def rel_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # pred/target: (B, Np)
    err2 = (pred - target).pow(2).mean(dim=1)                 # (B,)
    ref2 = target.pow(2).mean(dim=1).clamp_min(eps)           # (B,)
    rel = err2 / ref2                                         # (B,)
    return rel.mean()    

def test(model, test_loader, data_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    ys = torch.linspace(0.0, 1.0, data_size, device=device)
    xs = torch.linspace(0.0, 1.0, data_size, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    # precompute trunk
    trunk_out = model.trunk(XY)
    total, count = 0.0, 0
    for a, u in test_loader:
        a = a.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)
        branch_out = model.branch(a)
        B = u.shape[0]
        pred = model(branch_out, trunk_out)
        u2d = u.squeeze(1)
        pred = pred.view(B, data_size, data_size)
        num = (pred - u2d).pow(2).mean(dim=(1,2))     # (B,)
        den = u2d.pow(2).mean(dim=(1,2))
        loss = (num / den).mean()                    # scalar, mean over batch
        
        total += loss*B
        count += B
    # Average 
    mean_loss = total / count
    print(f"rel L2 error: {np.sqrt(mean_loss.item()):.4e}")


if __name__ == "__main__":
    test_data_path = "Datasets/threshold_dataset_test_nodes257_2500.h5"
    p_dim = 128
    trunk_hidden_size = 128
    data_size = 257
    ckpt_path = "don_thrshold_best.pt"
    br = BranchCNN(p = p_dim)
    tr = Trunk(in_dim = 2, hidden=trunk_hidden_size, p=p_dim)
    model = DeepONet(branch=br, trunk=tr)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    A_mean_train = ckpt["norm"]["A_mean"]
    A_std_train = ckpt["norm"]["A_std"]
    model.load_state_dict(ckpt["model_state"], strict=True)

    with h5py.File(test_data_path, "r") as f:
        N = f["data/a"].shape[0]
        A_test = torch.from_numpy(f["data/a"][:,:,:256,:256])
        U_test = torch.from_numpy(f["data/u"][:])
    # Normalize data
    A_test = (A_test-A_mean_train) / A_std_train
    batch_size = 256
    test_ds = TensorDataset(A_test, U_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    test(model, test_loader, data_size)
    


