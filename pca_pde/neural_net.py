import torch
from torch import nn

class PCANet(nn.Module):
    def __init__(self, n_basis, device="cpu", dtype=torch.float32, N=None, centered_pca: bool = False):
        super().__init__()
        self.n_basis = int(n_basis)
        self.dtype = dtype
        self.device = torch.device(device)
        self.centered_pca = bool(centered_pca)

        self.relu_stack = nn.Sequential(
            nn.Linear(self.n_basis, 500),
            nn.SELU(),
            nn.Linear(500, 1000),
            nn.SELU(),
            nn.Linear(1000, 2000),
            nn.SELU(),
            nn.Linear(2000, 1000),
            nn.SELU(),
            nn.Linear(1000, 500),
            nn.SELU(),
            nn.Linear(500, self.n_basis),
        )

        # Spatial shape used by flatten/unflatten; overwritten by init_pca_basis
        self.hw_shape = (N, N)

        # PCA buffers (always present for checkpoint consistency)
        self.register_buffer("Phi_a", torch.empty(0))
        self.register_buffer("Phi_u", torch.empty(0))
        self.register_buffer("eig_a", torch.empty(0))
        self.register_buffer("eig_u", torch.empty(0))
        self.register_buffer("mu_a",  torch.empty(0))
        self.register_buffer("mu_u",  torch.empty(0))

        # Ensure module parameters are on requested dtype/device
        self.to(device=self.device, dtype=self.dtype)

    def forward(self, a):
        z = self.encode_a(a)          # (B, r)
        z = self.relu_stack(z)        # (B, r)
        u = self.decode_u(z, like=a)  # shaped like a
        return u

    @staticmethod
    def _squeeze_channel_if_present(x: torch.Tensor) -> torch.Tensor:
        # Accept (H,W), (1,H,W), (B,H,W), (B,1,H,W)
        if x.ndim == 3 and x.shape[0] == 1:      # (1,H,W) -> (H,W)
            return x.squeeze(0)
        if x.ndim == 4 and x.shape[1] == 1:      # (B,1,H,W) -> (B,H,W)
            return x[:, 0, ...]
        return x

    def _flattenC_batch(self, x: torch.Tensor):
        """
        Accepts: (H,W), (1,H,W), (B,H,W), (B,1,H,W).
        Returns:
            X: (B, HW) (B=1 if unbatched)
            had_batch: bool
            had_channel: bool
        """
        had_batch = (x.ndim == 3 and x.shape[0] != 1) or (x.ndim == 4)
        had_channel = (x.ndim == 3 and x.shape[0] == 1) or (x.ndim == 4 and x.shape[1] == 1)

        x = self._squeeze_channel_if_present(x)

        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1,H,W)
            had_batch = False
        elif x.ndim != 3:
            raise ValueError(f"Expected (H,W), (1,H,W), (B,H,W), or (B,1,H,W). Got shape={tuple(x.shape)}")

        B, H, W = x.shape
        if (H, W) != self.hw_shape:
            raise ValueError(f"Expected spatial shape {self.hw_shape}, got {(H, W)}")

        X = x.contiguous().view(B, H * W)
        X = X.to(device=self.Phi_a.device, dtype=self.Phi_a.dtype)
        return X, had_batch, had_channel

    def _unflattenC_batch(self, Y: torch.Tensor, had_batch: bool, had_channel: bool):
        """
        Y: (B, HW) -> (H,W) or (B,H,W) or (B,1,H,W) depending on flags.
        """
        H, W = self.hw_shape
        if Y.ndim != 2 or Y.shape[1] != H * W:
            raise ValueError(f"Expected Y with shape (B,{H*W}), got {tuple(Y.shape)}")

        B = Y.shape[0]
        u = Y.contiguous().view(B, H, W)

        if not had_batch:
            u = u[0]  # (H,W)
            if had_channel:
                u = u.unsqueeze(0)  # (1,H,W)
            return u

        if had_channel:
            u = u.unsqueeze(1)  # (B,1,H,W)
        return u

    @torch.no_grad()
    def init_pca_basis(self, dataset, N=None, indices=None, centered: bool | None = None):
        """
        Deterministic PCA via economy SVD. Can be centered or non-centered.

        If centered=True:
            mu = mean(X, dim=0), Xc = X - mu
        If centered=False:
            mu = 0, Xc = X

        Then:
            Xc = U S V^T
            Phi = V[:, :r]
            eig = (S[:r]^2)/N

        Stores:
            Phi_a, Phi_u : (HW, r)
            eig_a, eig_u : (r,)
            mu_a,  mu_u  : (HW,)  (zeros if non-centered)
        """
        do_center = self.centered_pca if centered is None else bool(centered)

        # Infer shape
        a0, u0 = dataset[0]
        if not torch.is_tensor(a0): a0 = torch.as_tensor(a0)
        if not torch.is_tensor(u0): u0 = torch.as_tensor(u0)

        a0 = self._squeeze_channel_if_present(a0)
        u0 = self._squeeze_channel_if_present(u0)
        if a0.ndim != 2 or u0.ndim != 2:
            raise ValueError("Expected each sample to provide 2D fields (H,W) or channelized variants.")

        H, W = int(a0.shape[0]), int(a0.shape[1])
        self.hw_shape = (H, W)
        HW = H * W

        # Choose subset
        total_len = len(dataset)
        if indices is None:
            if N is None:
                N = total_len
            N = min(int(N), total_len)
            idxs = range(N)
        else:
            idxs = list(indices)
            N = len(idxs)

        r = int(self.n_basis)
        if r > min(N, HW):
            raise ValueError(f"n_basis={r} must be <= min(N,HW)={min(N,HW)}")

        # Build data matrices (C-order vectorization)
        X_a = torch.empty((N, HW), dtype=self.dtype, device=self.device)
        X_u = torch.empty((N, HW), dtype=self.dtype, device=self.device)

        for i, j in enumerate(idxs):
            a, u = dataset[j]
            a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
            u = torch.as_tensor(u, dtype=self.dtype, device=self.device)

            a = self._squeeze_channel_if_present(a)
            u = self._squeeze_channel_if_present(u)

            if a.ndim != 2 or u.ndim != 2:
                raise ValueError(f"Sample {j} not 2D after squeeze: a{tuple(a.shape)} u{tuple(u.shape)}")
            if (a.shape[0], a.shape[1]) != (H, W) or (u.shape[0], u.shape[1]) != (H, W):
                raise ValueError(
                    f"Inconsistent shapes at sample {j}: a{tuple(a.shape)} u{tuple(u.shape)} expected {(H,W)}"
                )

            X_a[i] = a.contiguous().view(-1)
            X_u[i] = u.contiguous().view(-1)

        def svd_pca(X: torch.Tensor, name: str):
            if do_center:
                mu = X.mean(dim=0)
                Xc = X - mu.unsqueeze(0)
                tag = "centered"
            else:
                mu = torch.zeros((HW,), dtype=X.dtype, device=X.device)
                Xc = X
                tag = "non-centered"

            print(f"Starting {tag} PCA via SVD on {name} data...")
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            Phi = Vh[:r, :].T.contiguous()           # (HW,r)
            eig = (S[:r] ** 2) / float(N)            # (r,)
            print("Done.")
            return Phi, eig.contiguous(), mu.contiguous()

        Phi_a, eig_a, mu_a = svd_pca(X_a, "input")
        Phi_u, eig_u, mu_u = svd_pca(X_u, "output")

        self.Phi_a = Phi_a
        self.Phi_u = Phi_u
        self.eig_a = eig_a
        self.eig_u = eig_u
        self.mu_a = mu_a
        self.mu_u = mu_u

    def encode_a(self, a: torch.Tensor) -> torch.Tensor:
        """
        PCA encoding (centered if mu_a != 0, otherwise non-centered).
        """
        if self.Phi_a.numel() == 0:
            raise RuntimeError("PCA basis not initialized. Call init_pca_basis first.")
        if self.mu_a.numel() == 0:
            raise RuntimeError("mu_a buffer not initialized. Recreate model with updated class or re-save checkpoint.")

        X, _, _ = self._flattenC_batch(a)  # (B,HW)
        mu = self.mu_a.to(device=X.device, dtype=X.dtype)
        z = (X - mu.unsqueeze(0)) @ self.Phi_a
        return z

    def decode_u(self, z: torch.Tensor, like: torch.Tensor = None) -> torch.Tensor:
        """
        PCA decoding (adds mu_u back if centered; mu_u is zero if non-centered).
        """
        if self.Phi_u.numel() == 0:
            raise RuntimeError("PCA basis not initialized. Call init_pca_basis first.")
        if self.mu_u.numel() == 0:
            raise RuntimeError("mu_u buffer not initialized. Recreate model with updated class or re-save checkpoint.")

        if z.ndim == 1:
            z = z.unsqueeze(0)
        elif z.ndim != 2:
            raise ValueError(f"z must be (r,) or (B,r). Got {tuple(z.shape)}")

        z = z.to(device=self.Phi_u.device, dtype=self.Phi_u.dtype)
        Y = z @ self.Phi_u.T  # (B,HW)

        mu = self.mu_u.to(device=Y.device, dtype=Y.dtype)
        Y = Y + mu.unsqueeze(0)

        if like is None:
            had_batch = (z.shape[0] > 1)
            had_channel = False
            return self._unflattenC_batch(Y, had_batch=had_batch, had_channel=had_channel)

        _, had_batch, had_channel = self._flattenC_batch(like)
        return self._unflattenC_batch(Y, had_batch=had_batch, had_channel=had_channel)


if __name__ == "__main__":
    _ = PCANet(n_basis=70, device="cpu", dtype=torch.float32, N=256)
