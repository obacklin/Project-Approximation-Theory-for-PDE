import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

def _best_gn_groups(ch: int, max_groups: int = 32) -> int:
    """Pick the largest group count ≤ max_groups that divides ch."""
    for g in [32, 16, 8, 4, 2, 1]:
        if g <= max_groups and ch % g == 0:
            return g
    return 1  # fallback (should rarely happen)

def GN(ch: int, max_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(_best_gn_groups(ch, max_groups), ch)

class SE(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, ch, r=16):
        super().__init__()
        hidden = max(1, ch // r)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ConvBlock(nn.Module):
    """Residual 3x3-3x3 block with GroupNorm (+ optional SE) and stride/dilation."""
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, se=False, padding_mode="zeros", gn_max_groups: int = 32):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=pad,
                               dilation=dilation, bias=False, padding_mode=padding_mode)
        self.gn1   = GN(out_ch, gn_max_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=pad,
                               dilation=dilation, bias=False, padding_mode=padding_mode)
        self.gn2   = GN(out_ch, gn_max_groups)
        self.act   = nn.SiLU(inplace=True)
        self.se    = SE(out_ch) if se else nn.Identity()
        self.skip  = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        if self.skip is not None:
            identity = self.skip(identity)
        return self.act(out + identity)

class DeepONetBranchCNN(nn.Module):
    """
    One-channel branch encoder: (B, 1, H, W) -> (B, latent_dim)
    - No coordinate channels, strictly single input channel.
    - No max pooling; downsampling via strided convs; optional SPP & FFT features.
    """
    def __init__(
        self,
        in_ch=1,                # must be 1
        latent_dim=256,         # p (size expected by DeepONet trunk)
        widths=(32, 64, 128, 256),
        blocks=(2, 2, 2, 2),
        periodic=False,         # 'circular' padding if True
        use_se=True,            # squeeze-excitation in blocks
        use_spp=True,           # multi-scale pooling head
        add_fft_lowmodes=0,     # 0 to disable; else small int like 4 or 8
        drop_dc=False,
        gn_max_groups: int = 32
    ):
        super().__init__()
        assert in_ch == 1, "Branch expects single-channel input."
        self.add_fft_lowmodes = add_fft_lowmodes
        self.drop_dc = drop_dc
        padmode = "circular" if periodic else "zeros"

        # Stem (downsample once: /2). Swap BN -> GN.
        c0 = 1
        self.stem = nn.Sequential(
            nn.Conv2d(c0, widths[0], 3, stride=2, padding=1, bias=False, padding_mode=padmode),  # /2
            GN(widths[0], gn_max_groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(widths[0], widths[0], 3, stride=1, padding=1, bias=False, padding_mode=padmode),
            GN(widths[0], gn_max_groups),
            nn.SiLU(inplace=True),
        )

        # Stages: first block of each later stage downsamples (stride=2)
        stages = []
        in_c = widths[0]
        for i, (w, nblk) in enumerate(zip(widths, blocks)):
            stage = []
            for b in range(nblk):
                stride = 2 if (b == 0 and i > 0) else 1
                dilation = 1 if i < len(widths) - 1 else (2 if b >= 1 else 1)  # light dilation in last stage
                stage.append(ConvBlock(in_c, w, stride=stride, dilation=dilation, se=use_se,
                                       padding_mode=padmode, gn_max_groups=gn_max_groups))
                in_c = w
            stages.append(nn.Sequential(*stage))
        self.stages = nn.ModuleList(stages)

        # Head: GAP (+ optional SPP bins) -> Linear to latent_dim
        C_last = widths[-1]
        self.spp_bins = (2, 4) if use_spp else ()  # avoid duplicating GAP by skipping b=1
        head_in = C_last * (1 + sum(b*b for b in self.spp_bins)) if self.spp_bins else C_last

        extra_fft = (2 * add_fft_lowmodes * add_fft_lowmodes) if add_fft_lowmodes > 0 else 0
        self.proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(head_in + extra_fft, latent_dim),
        )

        # init convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _fft_lowmodes_ortho(self, x, k):
        """
        x: (B,C,H',W') conv map. Returns float32 (B, 2*k*k).
        Uses rfft2(..., norm='ortho') so magnitudes are size-independent.
        """
        B, C, Hp, Wp = x.shape
        xr = x.mean(dim=1)  # (B,H',W')

        # Do FFT in fp32 with autocast disabled to avoid ComplexHalf warnings
        with autocast(device_type="cuda", enabled=False):
            X = torch.fft.rfft2(xr.float(), norm="ortho")  # complex64
            if self.drop_dc:
                X[:, 0, 0] = 0

            kx = min(k, X.shape[-2])
            ky = min(k, X.shape[-1])
            patch = X[:, :kx, :ky]  # (B,kx,ky)
            feat = torch.cat(
                [patch.real.reshape(B, -1), patch.imag.reshape(B, -1)],
                dim=1
            )  # (B, 2*kx*ky), float32
        return feat

    def forward(self, a):
        x = self.stem(a)
        for st in self.stages:
            x = st(x)

        B = x.size(0)
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        pools = [gap] + [F.adaptive_avg_pool2d(x, (b, b)).reshape(B, -1) for b in self.spp_bins]
        feat = torch.cat(pools, dim=1)

        if self.add_fft_lowmodes > 0:
            # expects you to have _fft_lowmodes_ortho defined, or remove this branch
            fft_feat = self._fft_lowmodes_ortho(x, self.add_fft_lowmodes)  # float32
            if fft_feat.dtype != feat.dtype:
                fft_feat = fft_feat.to(feat.dtype)
            feat = torch.cat([feat, fft_feat], dim=1)

        return self.proj(feat)

class Trunk(nn.Module):
    """
    Trunk phi: (x,y,...) -> psi(x) in R^p
    - out_dim: coordinate dimension (e.g., 2 for (x,y))
    - p: latent size (must match branch latent_dim)
    """
    def __init__(self, in_dim = 2, hidden=128, p=128):
        super().__init__()
        self.in_dim = in_dim
        self.p = p
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p)
        )
        # scalar bias added after dot-product
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, xy):  # xy: (B, Np, out_dim)
        return self.net(xy)  # (B, Np, p)


class DeepONet(nn.Module):
    """
    DeepONet: a to phi(a) in R^p, xy -> psi(xy) in R^{Npxp}.
    """
    def __init__(self, branch: nn.Module, trunk: Trunk):
        super().__init__()
        
        self.branch = branch   # must output (B, p)
        self.trunk  = trunk    # must output (B, Np, p)

    def forward(self, a, xy):
        
        phi = self.branch(a)        # (B, p)
        psi = self.trunk(xy)        # (B, Np, p)
        out = torch.einsum("bp,bnp->bn", phi, psi) + self.trunk.bias  # (B, Np)
        return out
