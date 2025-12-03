# compare_fd_pinn.py
import numpy as np
import torch
import matplotlib.pyplot as plt

from Darcy2DAssemble import assemble_divAgrad_dirichlet
from pinn_net import PINN, a_fun as a_fun_torch  # reuse same architecture & coefficient

def a_fun_np(X, Y):
    # numpy version, matching pinn_net.a_fun
    return 1.5*np.exp(-X**2/0.2 - Y**2/0.2)

def f_fun_np(X, Y):
    # IMPORTANT: matches pinn_net.f_fun sign/convention for -div(a∇u)=f
    return np.sin(-np.pi*X)*np.sin(1.5*np.pi*Y)

def main(
    N=128,
    ckpt_path="pinn_best.pt",
    device="cpu",          # you can set "cuda" if you want to evaluate on GPU
    show=True,
    savefig=None
):
    # -------------------------------
    # 1) Finite-difference solution
    # -------------------------------
    # Interior spacing is h = 1/(N+1), interior nodes are x,y in (0,1)
    y_interior = (np.arange(N, dtype=float) + 1.0) / (N + 1.0)

    gW = np.sin(2*np.pi * y_interior)  # g(0,y) on the west boundary
    gE = np.zeros(N)                   # east
    gS = np.zeros(N)                   # south
    gN = np.zeros(N)                   # north

    A, b, x, y, vec_to_grid = assemble_divAgrad_dirichlet(
        N=N,
        a=a_fun_np,          # same coefficient as the PINN
        f=f_fun_np,          # same forcing as the PINN
        g=(gW, gE, gS, gN)   # pass the boundary arrays
    )
    # Solve sparse linear system
    from scipy.sparse.linalg import spsolve
    u_fd_vec = spsolve(A, b)
    U_fd = vec_to_grid(u_fd_vec)   # shape (N, N), ordering k = i + N*j

    # --------------------------------
    # 2) PINN prediction on same grid
    # --------------------------------
    # Build the same (x,y) interior grid and flatten in Fortran order
    # so that k = i + N*j matches vec_to_grid's convention.
    Xg, Yg = np.meshgrid(x, y, indexing='ij')      # Xg,Yg: (N,N)
    xy_pairs = np.stack([Xg.ravel(order='F'), Yg.ravel(order='F')], axis=1)  # (N*N, 2)

    # Instantiate model and load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = PINN()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        xy_t = torch.from_numpy(xy_pairs.astype(np.float32)).to(device)
        u_pinn_vec = model(xy_t).squeeze(1).cpu().numpy()  # (N*N,)
    U_pinn = u_pinn_vec.reshape((N, N), order='F')         # (N,N), same layout as U_fd

    # ---------------------------
    # 3) Difference (FD - PINN)
    # ---------------------------
    U_diff = U_fd - U_pinn

    # Use shared color scale for the first two plots for a fair visual comparison
    vmin = min(U_fd.min(), U_pinn.min())
    vmax = max(U_fd.max(), U_pinn.max())

    # Make plots
    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3), constrained_layout=True)

    im0 = axs[0].imshow(
        U_fd.T, origin='lower', extent=(0,1,0,1), aspect='equal',
        vmin=vmin, vmax=vmax
    )
    axs[0].set_title("Ground Truth", fontsize=9)

    im1 = axs[1].imshow(
        U_pinn.T, origin='lower', extent=(0,1,0,1), aspect='equal',
        vmin=vmin, vmax=vmax
    )
    axs[1].set_title("Approximation", fontsize=9)

    # For the difference, center around zero with a symmetric range
    dmax = np.max(np.abs(U_diff))
    im2 = axs[2].imshow(
        U_diff.T, origin='lower', extent=(0,1,0,1), aspect='equal',
        vmin=-dmax, vmax=dmax
    )
    axs[2].set_title("Error", fontsize=9)

    cbar_kw = dict(fraction=0.046, pad=0.04)
    cb0 = fig.colorbar(im0, ax=axs[0], **cbar_kw); cb0.ax.tick_params(labelsize=8)
    cb1 = fig.colorbar(im1, ax=axs[1], **cbar_kw); cb1.ax.tick_params(labelsize=8)
    cb2 = fig.colorbar(im2, ax=axs[2], **cbar_kw); cb2.ax.tick_params(labelsize=8)

    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    if savefig:
        plt.savefig(savefig, dpi=300)
    if show:
        plt.show()

if __name__ == "__main__":
    # Adjust N and paths as you like
    main(N=1024, ckpt_path="pinn_best.pt", device="cuda", show=True, savefig=None)
