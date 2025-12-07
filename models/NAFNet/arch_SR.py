import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_norm import LayerNorm3d
from .block import NAFBlock3d
from .pixel_shuffle import PixelShuffle3d

class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM) - 3D version
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm3d(c)
        self.norm_r = LayerNorm3d(c)
        self.l_proj1 = nn.Conv3d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv3d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv3d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv3d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 4, 1)  # B, H, W, D, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 3, 1, 4) # B, H, W, c, D (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 4, 1)  # B, H, W, D, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 4, 1)  # B, H, W, D, c

        # (B, H, W, D, c) x (B, H, W, c, D) -> (B, H, W, D, D)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, D, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 2, 4, 3), dim=-1), V_l) # B, H, W, D, c

        # scale
        F_r2l = F_r2l.permute(0, 4, 1, 2, 3) * self.beta
        F_l2r = F_l2r.permute(0, 4, 1, 2, 3) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats

class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution - 3D version
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock3d(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats

class NAFNetSR(nn.Module):
    '''
    NAFNet for Super-Resolution - 3D version
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=1, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv3d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate, 
                NAFBlockSR(width)) for i in range(num_blks)]
        )

        self.up = nn.Sequential(
            nn.Conv3d(in_channels=width, out_channels=img_channel * up_scale**3, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            PixelShuffle3d(up_scale)
        )
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='trilinear')

        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out

