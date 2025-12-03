import torch
from torch import nn
import os, time, stat, shutil, errno

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        indim = 2
        self.lin_tanh_stack = nn.Sequential(
            nn.Linear(indim, 32),
            nn.Tanh(),
            nn.Linear(32,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )

    def forward(self, x):
        return self.lin_tanh_stack(x)

def diff(z, x, graph = True):
    return torch.autograd.grad(
        z, 
        x,
        grad_outputs=torch.ones_like(z), 
        create_graph=graph, 
        retain_graph=True 
        )[0]

def divergence(v,x,graph=True):
    dvx_dx = torch.autograd.grad(v[:,0:1], x, torch.ones_like(v[:, 0:1]),
                                create_graph=graph, retain_graph=True)[0][:,0:1]
    dvy_dy = torch.autograd.grad(v[:,1:2], x, torch.ones_like(v[:, 1:2]),
                                create_graph=graph, retain_graph=True)[0][:,1:2]
    return dvx_dx + dvy_dy

def a_fun(x):
    X, Y = x[:, 0:1], x[:, 1:2]
    return 1.5*torch.exp(-X**2/0.2 - Y**2/0.2)

def f_fun(x):
    X, Y = x[:, 0:1], x[:, 1:2]
    return torch.sin(-torch.pi*X)*torch.sin(1.5*torch.pi*Y)

def f_res(model: nn.Module, x: torch.Tensor, a_fun, f_fun, graph=True):
    
    u = model(x)
    grad_u = diff(u, x, graph=graph)
    a = a_fun(x)
    q = a * grad_u
    div_q = divergence(q, x, graph=graph)
    f = f_fun(x)
    res = -div_q-f
    return res
    
def interior_loss(model, x_interior, a_fun, f_fun, alpha = None):
    if not x_interior.requires_grad:
        x_interior = x_interior.detach().requires_grad_(True)
    r = f_res(model,x_interior,a_fun, f_fun)
    if alpha is not None:
        r = alpha * r
    return (r**2).mean()

def sample_boundary_with_bc(N_boundary, device):
    s = torch.rand(N_boundary, 1, device= device)
    left = torch.cat([torch.zeros_like(s), s], dim = 1)
    right = torch.cat([torch.ones_like(s), s], dim = 1)
    bottom = torch.cat([s, torch.zeros_like(s)], dim = 1)
    top = torch.cat([s, torch.ones_like(s)], dim  = 1)

    x_boundary = torch.cat([left, right, bottom, top], dim = 0)

    g_left = torch.sin(2*torch.pi * s)
    g_other = torch.zeros_like(s)
    g_bc = torch.cat([g_left, g_other, g_other, g_other], dim=0)

    return x_boundary, g_bc

def boundary_loss(model, x_boundary, g_bc, alpha=None):
    # Dirichlet
    u_bc_pred = model(x_boundary)
    res_bc = (u_bc_pred-g_bc)
    if alpha is not None:
        res_bc = alpha*res_bc
    return (res_bc**2).mean()


def residuals_interior(model, x_interior, a_fun, f_fun):
    # compute |r_i| for PDE term
    x = x_interior.detach().requires_grad_(True)
    r = f_res(model, x, a_fun, f_fun, graph=False)
    return r.detach().abs().squeeze(-1)


def residuals_boundary(model, x_boundary, g_bc):
    rb = (model(x_boundary) - g_bc)
    return rb.abs().squeeze(-1)


@torch.no_grad()
def rba_update_(alpha, abs_residuals, eta):
    if abs_residuals.numel() == 0:
        return
    m = abs_residuals.max()
    if m <= 0:
        return
    alpha.mul_(1.0 - eta).add_(eta * (abs_residuals / m).unsqueeze(-1))

def residual_abs(model, X: torch.Tensor, a_fun, f_fun, chunk: int | None = None) -> torch.Tensor:

    def _batch(xb: torch.Tensor) -> torch.Tensor:
        xb = xb.detach().requires_grad_(True)
        with torch.enable_grad():
            r = f_res(model, xb, a_fun, f_fun, graph=False)
        return r.detach().abs().squeeze(-1)

    if chunk is None:
        return _batch(X)
    outs = []
    for i in range(0, X.shape[0], chunk):
        outs.append(_batch(X[i:i+chunk]))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def rad_probability(model, a_fun, f_fun, S: torch.Tensor, a: float = 3.0, c: float = 1.0,
                    chunk: int | None = None, eps: float = 1e-12) -> torch.Tensor:

    r_abs = residual_abs(model, S, a_fun, f_fun, chunk=chunk)    
    powa  = r_abs.clamp_min(eps).pow(a)
    denom = powa.mean().clamp_min(eps)
    p = (powa / denom) + float(c)
    p = p.clamp_min(eps)
    p = p / p.sum()
    if not torch.isfinite(p).all():
        p = torch.full_like(p, 1.0 / p.numel())
    return p

# RAD resample
@torch.no_grad()
def rad_resample_uniform(model, a_fun, f_fun,
                         Nf: int, d: int, device,
                         a: float = 3.0, c: float = 1.0,
                         M: int | None = None,
                         old_alpha_f: torch.Tensor | None = None,
                         chunk: int | None = None,
                         seed: int | None = None):
    """
    Builds a uniform pool S in [0,1]^d of size M,
    samples Nf new interior points ~ Multinomial(p), and re-inits alpha_f.
    Returns:
        x_interior: (Nf, d) with requires_grad=True
        alpha_f:    (Nf, 1)
    """
    if M is None:
        M = max(4 * Nf, Nf)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    # pool S
    S = torch.rand(M, d, device=device, generator=gen)

    # probabilities p over S
    p = rad_probability(model, a_fun, f_fun, S, a=a, c=c, chunk=chunk)

    # sample Nf indices according to p
    replace = Nf > M
    idx = torch.multinomial(p, Nf, replacement=replace, generator=gen)
    x_interior = S[idx].detach().requires_grad_(True)

    # re-init alpha_f (mean of old alpha_f or ones)
    if old_alpha_f is None:
        alpha_f = torch.ones(Nf, 1, device=device)
    else:
        a0 = float(old_alpha_f.mean().clamp_min(1e-12))
        alpha_f = torch.full((Nf, 1), a0, device=device)

    return x_interior, alpha_f


def _clear_readonly(p: str):
    try:
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass

def save_if_best(model, train_loss, path="pinn_best_train.pt", meta=None):
    ## Cursed function
    prev = float('inf')
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        prev = ckpt.get("best_train_loss", float('inf'))

    if train_loss < prev:
        payload = {"model_state": model.state_dict(),
                   "best_train_loss": float(train_loss),
                   "meta": meta or {}}

        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, tmp)

        # retry replace
        for _ in range(10):
            try:
                if os.path.exists(path):
                    _clear_readonly(path)
                os.replace(tmp, path)
                print(f"Saved new best ({train_loss:.3e} < {prev:.3e}) → {path}")
                break
            except PermissionError:
                time.sleep(0.2)
        else:
            try:
                if os.path.exists(path):
                    _clear_readonly(path)
                    os.remove(path)
                shutil.move(tmp, path)
                print(f"Saved new best → {path}")
            finally:
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except OSError: pass

if __name__ == "__main__":
    ## Bunch of BS to circumvent interm saving of ckpt
    import sys
    sys.exit()
    ###
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)

    model = PINN().to(device)
    ckpt_path = "pinn_best.pt"
    epochs_adam = 50000
    resample_freq = 5000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs_adam, eta_min=1e-5)

    n_interior = 3000
    n_boudary  = 512

    x_interior = torch.rand(n_interior, 2, device=device).requires_grad_(True)
    x_boundary, g_bc = sample_boundary_with_bc(n_boudary, device=device)
    
    alpha_f  = torch.ones(x_interior.shape[0], 1, device=device)
    alpha_bc = torch.ones(x_boundary.shape[0], 1, device=device)  # all BC points pooled
    eta_rba  = 1e-4

    tic = time.perf_counter()
    best = float('inf')
    
    for ep in range(epochs_adam):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        loss_f  = interior_loss(model, x_interior, a_fun, f_fun, alpha = alpha_f)
        loss_bc = boundary_loss(model, x_boundary, g_bc, alpha= alpha_bc)
        loss = 3*loss_f + 0.5*loss_bc
        
        loss.backward()
        optimizer.step()

        if (ep + 1) % resample_freq == 0:
            # Adaptive resampling of Points.
            x_interior, alpha_f = rad_resample_uniform(model,a_fun, f_fun, Nf=n_interior, 
                                                       d=x_interior.shape[1], device = device, 
                                                       a=2.5, c=0.7, M=4*x_interior.shape[0], 
                                                       old_alpha_f=alpha_f)
        else:
            # Perform RBA update.
            r_in  = residuals_interior(model, x_interior, a_fun, f_fun)
            r_bnd = residuals_boundary(model, x_boundary, g_bc)
            rba_update_(alpha_f,  r_in,  eta_rba)
            rba_update_(alpha_bc, r_bnd, eta_rba)

        if loss.item() < best:
            best = loss.item()
            save_if_best(
                model, best, path=ckpt_path,
                meta={"phase":"adam", "epoch": ep, "loss_f": float(loss_f), "loss_bc": float(loss_bc)})
        
        scheduler.step()

        if ep % 100 == 0:
            print(f"[ADAM ep {ep}/{epochs_adam}] lr={optimizer.param_groups[0]['lr']:.2e} loss={loss.item():.3e}"
                f"(Lf={loss_f.item():.3e}, Lbc={loss_bc.item():.3e})")

    toc = time.perf_counter()
    print(f"Time taken: {toc-tic:.3f}s")