# starnet.py
import torch
import torch.nn as nn

# --- DropPath (with safe fallback if timm isn't available) ---
try:
    from timm.models.layers import DropPath  # timm >= 0.9
except Exception:  # pragma: no cover
    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor


# Ultralytics building blocks
from ultralytics.nn.modules.block import C2f, Conv


# --------------------------
# Basic Conv-BN (optional BN) utility
# --------------------------
class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 1,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        dilation: int | tuple = 1,
        groups: int = 1,
        with_bn: bool = True,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=int(groups),
                bias=not with_bn,
            ),
        )
        if with_bn:
            self.add_module("bn", nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 0.0)


# --------------------------
# Star-like block
# --------------------------
class StarsBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 1, drop_path: float = 0.0):
        """
        A lighter Star-style block (default mlp_ratio=1 for efficiency).
        """
        super().__init__()
        self.dwconv = ConvBN(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, kernel_size=1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, kernel_size=1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, kernel_size=1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = identity + self.drop_path(x)
        return x


# --------------------------
# Bottleneck with StarsBlock
# --------------------------
class Bottleneck_StarsBlock(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: int | tuple = (3, 3),
        e: float = 0.5,
        drop_path: float = 0.0,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.star = StarsBlock(c_, mlp_ratio=1, drop_path=drop_path)

        if isinstance(k, (list, tuple)):
            ksize = int(k[1] if len(k) > 1 else k[0])
        else:
            ksize = int(k)

        self.cv2 = Conv(c_, c2, ksize, 1, g=int(g))
        self.add = bool(shortcut) and (c1 == c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = self.star(y)
        y = self.cv2(y)
        return x + y if self.add else y


# --------------------------
# C2f with StarsBlock
# --------------------------
class C2f_StarsBlock(C2f):
    """
    A lightweight CSP bottleneck that swaps inner blocks with Bottleneck_StarsBlock.
    Designed to be *lighter* than standard C2f.
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        drop_path: float = 0.0,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            Bottleneck_StarsBlock(self.c, self.c, shortcut=shortcut, g=int(g), e=1.0, drop_path=drop_path)
            for _ in range(n)
        )


__all__ = [
    "ConvBN",
    "StarsBlock",
    "Bottleneck_StarsBlock",
    "C2f_StarsBlock",
]


# --------------------------
# Quick self-test
# --------------------------
if __name__ == "__main__":
    # StarsBlock check
    stars_block = StarsBlock(dim=32, mlp_ratio=1, drop_path=0.0)
    x = torch.randn(2, 32, 64, 64)
    y = stars_block(x)
    print("StarsBlock:", x.shape, "->", y.shape)

    # Bottleneck_StarsBlock check
    b = Bottleneck_StarsBlock(64, 64, shortcut=True, g=1, k=(3, 3), e=0.5, drop_path=0.0)
    x = torch.randn(2, 64, 64, 64)
    y = b(x)
    print("Bottleneck_StarsBlock:", x.shape, "->", y.shape)

    # C2f_StarsBlock check
    c2s = C2f_StarsBlock(64, 64, n=2, g=1, e=0.5, shortcut=True, drop_path=0.0)
    x = torch.randn(2, 64, 64, 64)
    y = c2s(x)
    print("C2f_StarsBlock:", x.shape, "->", y.shape)
