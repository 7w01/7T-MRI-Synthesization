import torch
import torch.nn as nn
from .vnet_part import InputTransition, DownTransition, UpTransition


class VNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_tr = InputTransition(4)
        self.down_tr32 = DownTransition(4, 1)
        self.down_tr64 = DownTransition(8, 1)
        self.down_tr128 = DownTransition(16, 1)
        self.up_tr128 = UpTransition(32, 32, 1)
        self.up_tr64 = UpTransition(32, 16, 1)
        self.up_tr32 = UpTransition(16, 8, 1)
        self.out_tr = nn.Sequential(
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )     

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        x = self.up_tr128(out128, out64)
        del out64
        x = self.up_tr64(x, out32)
        del out32
        x = self.up_tr32(x, out16)
        x = self.out_tr(x)
        return x