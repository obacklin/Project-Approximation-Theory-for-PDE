import torch
import torch.nn as nn

class BranchCNN(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        # Hyperparams
        ch_b1, ch_b2 = 32, 64
        g1, g2 = 8, 8
        poolsize = 4

        def gn(channels, groups): return nn.GroupNorm(groups, channels)

        self.block1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=ch_b1, kernel_size=3, stride=1, padding=1, bias=False),
                                    gn(ch_b1, g1),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(ch_b1, ch_b1, 3, stride=2, padding=1, bias = False),
                                    gn(ch_b1, g1),
                                    nn.SiLU(inplace=True)
                                    )

        self.block2 = nn.Sequential(nn.Conv2d(in_channels=ch_b1, out_channels=ch_b2, kernel_size=3, stride=1, padding=1, bias=False),
                                    gn(ch_b2, g2),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(ch_b2, ch_b2, kernel_size=3, stride=2, padding=1, bias=False),
                                    gn(ch_b2, g2),
                                    nn.SiLU(inplace=True)
                                    )
        self.pool = nn.AdaptiveAvgPool2d((poolsize, poolsize))
        self.flatten = nn.Flatten(1)
        self.mlp = nn.Sequential(
            nn.Linear(ch_b2*poolsize*poolsize, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256,p)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x

class Trunk(nn.Module):
    """
    Trunk: (x,y) -> psi(xy) in R^p
    """
    def __init__(self, in_dim: int = 2, hidden: int = 128, p: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.p = p
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SELU(),
            nn.Linear(hidden, hidden), nn.SELU(),
            nn.Linear(hidden, hidden), nn.SELU(),
            nn.Linear(hidden, hidden), nn.SELU(),
            nn.Linear(hidden, hidden), nn.SELU(),
            nn.Linear(hidden, p),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


class DeepONet(nn.Module):
    """
    DeepONet: a -> phi(a) in R^p, (x,y) -> psi(xy) in R^p, output via inner product.
    """
    def __init__(self, branch: nn.Module, trunk: Trunk):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_out, trunk_out) -> torch.Tensor:
        product = branch_out @ trunk_out.T
        return product + self.bias

if __name__ == "__main__":
    p = 128
    br = BranchCNN(p=128)
    tr = Trunk(in_dim=2, hidden=128, p=p)
    don = DeepONet(branch=br, trunk=tr)
