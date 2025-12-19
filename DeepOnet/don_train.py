import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import time
from deeponet_cnn import DeepONet, BranchCNN, Trunk
import os

def save_checkpoint(path, *, model, optim, sched, epoch, best_val, norm=None, extra=None):
    tmp = path + ".tmp"

    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict() if optim is not None else None,
        "sched_state": sched.state_dict() if sched is not None else None,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "extra": extra or {},
    }

    if norm is not None:
        # store on CPU to keep checkpoints portable
        ckpt["norm"] = {
            k: v.detach().float().cpu() if torch.is_tensor(v) else v
            for k, v in norm.items()
        }

    torch.save(ckpt, tmp)
    os.replace(tmp, path)

def set_trainable(module: nn.Module, trainable: bool):
    for p in module.parameters():
        p.requires_grad_(trainable)

def stop_module(module: nn.Module):
    set_trainable(module, False)

def sample_random_pixels(H, W, Np, device):
    iy = torch.randint(0, H, (Np,), device=device)
    ix = torch.randint(0, W, (Np,), device=device)
    x = ix.float() / (W - 1)
    y = iy.float() / (H - 1)
    xy = torch.stack([x, y], dim=-1)   # (Np, 2)
    return xy, iy, ix

def rel_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # pred/target: (B, Np)
    err2 = (pred - target).pow(2).mean(dim=1)                 # (B,)
    ref2 = target.pow(2).mean(dim=1).clamp_min(eps)           # (B,)
    rel = err2 / ref2                                         # (B,)
    return rel.mean()    

def train(model, n_epochs, train_loader, val_loader, ckpt_path,
          data_size, A_mean, A_std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    norm_stats = {"A_mean": A_mean, "A_std": A_std}
    best_path  = ckpt_path
    final_path = ckpt_path.replace(".pt", "_final.pt")

    # Since CNNs are generally train over fewer epochs we have to split the paramters
    # into parts because we want to make the CNN stop traning sooner and freeze paramter updates.
    optim = torch.optim.AdamW(
        [   {"params": model.branch.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
            {"params": model.trunk.parameters(), "lr": 3e-4, "weight_decay": 1e-4} 
        ],
    betas=(0.9, 0.999), 
    eps=1e-8)

    optim_steps = int(n_epochs * len(train_loader) * 0.95)
    global_steps = 0
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=[1e-4, 3e-4],
        total_steps= optim_steps,
        pct_start=0.08,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e3)

    n_points_output = 8192
    n_points_val = 2*n_points_output
    block1_stop = 40
    block2_stop = 60
    block3_stop = 70
    mlp_stop  = 100
    cnn_stop = 70
    cnn_full_stop = 145

    best_val = float("inf")
    best_epoch = -1
    tic = time.perf_counter()
    for ep in range(1, n_epochs+1):
        train_loss, train_n = 0.0, 0
        
        # CNN and MLP needs to train a very different amount of epochs
        # Stop at different intervals
        # In this way the linear layers at the end of the encoder can still keep training
        if ep == cnn_full_stop:
            stop_module(model.branch)

        for a, u in train_loader:
            a = a.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            u = u.to(device, non_blocking=True)
            # This has to be done here if sample size mod batch_size = 0, otherwise it fails on the last batch
            B = u.shape[0]
            bidx = torch.arange(B, device=u.device)[:, None]
            # Sample coords and index once per batch, enables precomputing the trunk once per batch
            xy, ix, iy = sample_random_pixels(data_size, data_size, n_points_output, device)
            branch_out = model.branch(a) # (B_size, p)
            trunk_out = model.trunk(xy) # (Np, p)
            pred = model(branch_out, trunk_out)
            target = u[bidx, 0, iy[None, :], ix[None, :]]  # (B, Np)
            
            optim.zero_grad(set_to_none=True)
            loss = rel_mse_loss(pred, target)
            loss.backward()
            optim.step()
            # Some additional steps with the lowest lr for some extra fine tuneing
            if global_steps < optim_steps:
                sched.step()
            global_steps +=1
            train_loss += loss.item()*B
            train_n += B

        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for a, u in val_loader:
                a = a.to(device, non_blocking=True)
                u = u.to(device, non_blocking=True)
                B = u.shape[0]
                bidx = torch.arange(B, device=u.device)[:, None]
                xy, ix, iy = sample_random_pixels(data_size, data_size, n_points_val, device)
                branch_out = model.branch(a) # (B_size, p)
                trunk_out = model.trunk(xy) # (Np, p)
                pred = model(branch_out, trunk_out)
                target = u[bidx, 0, iy[None, :], ix[None, :]]  # (B, Np)
                
                loss = rel_mse_loss(pred, target)
                val_sum += loss.item() * B
                val_n += B
        
        train_rel_mse = train_loss / max(train_n, 1)
        val_rel_mse = val_sum / max(val_n, 1)

        if val_rel_mse < best_val:
            best_val = val_rel_mse
            best_epoch = ep
            save_checkpoint(best_path,
                            model=model, optim=optim, sched=sched,
                            epoch= best_epoch, best_val=best_val,
                            norm=norm_stats,
                            extra={"train_rel_mse":train_rel_mse, "val_rel_mse": val_rel_mse})
            print(f"New best saved @ ep {ep} | rel.mse(validation): {val_rel_mse:.5e}")

        curr_lr = optim.param_groups[0]["lr"]
        print(f"ep: {ep} | lr={curr_lr:.3e} | rel.mse(train): {train_rel_mse:.5e} | rel.mse(validation): {val_rel_mse:.5e} | rel_L2: {np.sqrt(val_rel_mse):.5e}")
    toc = time.perf_counter()
    print(f"Total time: {toc-tic:.1f} sec")
    save_checkpoint(final_path,
                    model=model, optim=optim, sched=sched,
                    epoch=n_epochs, best_val=best_val,
                    norm=norm_stats,
                    extra= {"best_epoch": best_epoch})

if __name__ == "__main__":
    # Load traning data; input a is padded with extra circular row and col, which remove.
    torch.backends.cuda.matmul.allow_tf32 = True
    train_data_path = "Datasets/lognormal_dataset_train_nodes257_1025.h5"
    val_data_path = "Datasets/lognormal_dataset_val_nodes257_200.h5"
    p_dim = 256
    trunk_hidden_size = 256
    ckpt_path = "don_lognorm_best.pt"
    batch_size = 32
    val_batch_size = 40
    n_epochs = 3000
    # Size of the u input data, for computing random indicies for x,y in the prediction g(u)((x,y)).
    data_size = 257

    with h5py.File(train_data_path, "r") as f:
        N = f["data/a"].shape[0]
        A_train = torch.from_numpy(f["data/a"][:1024,:,:256,:256])
        print(A_train.shape)
        U_train = torch.from_numpy(f["data/u"][:1024])

    with h5py.File(val_data_path, "r") as f:
        N = f["data/a"].shape[0]
        A_val = torch.from_numpy(f["data/a"][:,:,:256,:256])
        U_val = torch.from_numpy(f["data/u"][:])
    
    # Normalize data
    A_mean = A_train.mean(dim=(0,2,3), keepdim=True)
    A_std = A_train.std(dim=(0,2,3),keepdim=True)
    A_train = (A_train-A_mean) / A_std
    A_val = (A_val-A_mean) / A_std

    train_ds = TensorDataset(A_train,U_train)
    val_ds = TensorDataset(A_val, U_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    br = BranchCNN(p = p_dim)
    tr = Trunk(in_dim = 2, hidden=trunk_hidden_size, p=p_dim)
    model = DeepONet(branch=br, trunk=tr)
    train(model, n_epochs, train_loader, val_loader, ckpt_path,
      data_size, A_mean, A_std)
    print(f"Model params: pdim={p_dim}, trunk_size = {trunk_hidden_size}, batch size = {batch_size}")