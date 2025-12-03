import torch
from torch import nn

class FullyConnectedNetwork(nn.Module):
    def __init__(self, n_basis, device="cpu", dtype=torch.float32, N = 256):
        super().__init__()
        self.n_basis = n_basis
        self.dtype = dtype
        self.device = torch.device(device)
        self.relu_stack = nn.Sequential(
            nn.Linear(self.n_basis, 500),
            nn.SELU(),
            nn.Linear(500,1000),
            nn.SELU(),
            nn.Linear(1000,2000),
            nn.SELU(),
            nn.Linear(2000, 1000),
            nn.SELU(),
            nn.Linear(1000, 500),
            nn.SELU(),
            nn.Linear(500, self.n_basis)
        )
        # placeholders filled by init_pca_basis
        self.register_buffer("Phi_a", torch.empty(0))
        self.register_buffer("Phi_u", torch.empty(0))
        self.register_buffer("eig_a", torch.empty(0))
        self.register_buffer("eig_u", torch.empty(0))
        self.hw_shape = (N, N)  # updated below

    def forward(self, a):
        z = self.encode_a(a)      # (B, r)
        z = self.relu_stack(z)    # (B, r)
        u = self.decode_u(z, like=a)
        return u
    
    @torch.no_grad()
    def init_pca_basis(
        self,
        dataset,
        N=None,                     # how many samples to use (default: all)
        indices=None,               # optional explicit subset (list/array of idx)
        oversample=10,               # q = n_basis + oversample
        niter=4,                    # power iterations for randomized PCA
        fortran_order=True          # match your "F" ordering
    ):
        """
        Compute non-centered PCA bases for inputs 'a' and outputs 'u' in a dataset
        that returns (a, u) per __getitem__. Each a/u is 2D (H,W) or (1,H,W).
        Stores:
            self.Phi_a, self.Phi_u : (H*W, n_basis)
            self.eig_a, self.eig_u : (n_basis,)  eigenvalues of non-centered covariance
        """

        # infer shapes from the first sample
        a0, u0 = dataset[0]
        if a0.ndim == 3 and a0.shape[0] == 1: a0 = a0.squeeze(0)
        if u0.ndim == 3 and u0.shape[0] == 1: u0 = u0.squeeze(0)
        assert a0.ndim == 2 and u0.ndim == 2, "Expected 2D arrays per sample."
        H, W = int(a0.shape[0]), int(a0.shape[1])
        self.hw_shape = (H, W)
        HW = H * W

        # choose subset
        total_len = len(dataset)
        if indices is None:
            if N is None: N = total_len
            N = min(N, total_len)
            idxs = range(N)
        else:
            idxs = list(indices)
            N = len(idxs)

        # --- helpers for "F" vs "C" vectorization ---
        def vecF(x2d: torch.Tensor) -> torch.Tensor:
            # Fortran-style column-major flatten ≡ transpose then C-flatten
            return x2d.t().contiguous().view(-1)

        def vecC(x2d: torch.Tensor) -> torch.Tensor:
            return x2d.contiguous().view(-1)

        vec = vecF if fortran_order else vecC

        # build data matrices X_a, X_u of shape (N, HW) on the chosen device ---
        X_a = torch.empty((N, HW), dtype=self.dtype, device=self.device)
        X_u = torch.empty((N, HW), dtype=self.dtype, device=self.device)

        for i, j in enumerate(idxs):
            a, u = dataset[j]
            # move to dtype/device and squeeze channel if present
            a = a.to(self.dtype, copy=False)
            u = u.to(self.dtype, copy=False)
            if a.ndim == 3 and a.shape[0] == 1: a = a.squeeze(0)
            if u.ndim == 3 and u.shape[0] == 1: u = u.squeeze(0)

            X_a[i] = vec(a).to(self.device, non_blocking=True)
            X_u[i] = vec(u).to(self.device, non_blocking=True)

        # randomized PCA (non-centered): V columns are eigenvectors of (1/N) X^T X ---
        r = self.n_basis
        q = min(r + oversample, min(N, HW))  # safe upper bound
        print("Starting PCA on input data...")
        # Inputs "a"
        Ua, Sa, Va = torch.pca_lowrank(X_a, q=q, center=False, niter=niter)
        Phi_a = Va[:, :r].contiguous()
        eig_a = (Sa[:r] ** 2) / N 
        print("Done...")
        print("Starting PCA on output data...")
        # Solutions u
        Uu, Su, Vu = torch.pca_lowrank(X_u, q=q, center=False, niter=niter)
        Phi_u = Vu[:, :r].contiguous()
        eig_u = (Su[:r] ** 2) / N
        print("Done...")
        # --- store as buffers so they travel with the model & land in state_dict ---
        self.Phi_a = Phi_a
        self.Phi_u = Phi_u
        self.eig_a = eig_a
        self.eig_u = eig_u

    def _vecF(self, x2d):
        return x2d.transpose(0, 1).contiguous().view(-1)

    def _devF(self, v1d):
        H, W = self.hw_shape
        return v1d.view(W, H).transpose(0, 1).contiguous()

    def _flattenF_batch(self, a):
        """
        Accepts (H,W), (B,H,W), or (B,1,H,W).
        Returns X: (B, HW), had_batch, had_channel
        """
        had_batch   = (a.ndim in (3, 4))
        had_channel = (a.ndim == 4 and a.shape[1] == 1)

        if a.ndim == 2:         # (H,W) -> (1,H,W)
            a = a.unsqueeze(0)
        elif had_channel:        # (B,1,H,W) -> (B,H,W)
            a = a[:, 0, ...]

        B, H, W = a.shape
        assert (H, W) == self.hw_shape, f"Expected {(self.hw_shape)}, got {(H,W)}"

        X = a.transpose(1, 2).contiguous().view(B, H * W)  # Fortran vec
        # move to the basis device/dtype (keeps autograd for a if it lives here)
        X = X.to(self.Phi_a.device, dtype=self.Phi_a.dtype)
        return X, had_batch, had_channel

    def _unflattenF_batch(self, Y, had_batch, had_channel):
        """
        Y: (B, HW)  -> (H,W) or (B,H,W) or (B,1,H,W) matching the input's layout.
        """
        H, W = self.hw_shape
        B    = Y.shape[0]
        u = Y.view(B, W, H).transpose(1, 2).contiguous()
        if not had_batch:
            u = u[0]
        elif had_channel:
            u = u.unsqueeze(1)
        return u

    # ---- differentiable enc/dec that handle batched & unbatched ----
    def encode_a(self, a):
        """
        a: (H,W) or (B,H,W) or (B,1,H,W)  ->  z: (B, r)  (B=1 if unbatched)
        """
        X, _, _ = self._flattenF_batch(a)
        z = X @ self.Phi_a 
        return z

    def decode_u(self, z, like=None):
        """
        z: (B, r) or (r,)  ->  u shaped like `like` (if provided), else (H,W) or (B,H,W)
        """
        if z.ndim == 1:
            z = z.unsqueeze(0)
        Y = z @ self.Phi_u.T 

        # If we got a template, mirror its batch/channel layout; otherwise, default.
        if like is None:
            # assume no channel dim
            return self._unflattenF_batch(Y, had_batch=(z.shape[0] > 1), had_channel=False)
        else:
            _, had_batch, had_channel = self._flattenF_batch(
                like if like.ndim != 4 or like.shape[1] == 1 else like[:, :1]
            )
            return self._unflattenF_batch(Y, had_batch, had_channel)

if __name__ == "__main__":
    _ = FullyConnectedNetwork()