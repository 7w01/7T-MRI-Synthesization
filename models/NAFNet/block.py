# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn

from layer_norm import LayerNorm3d


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock3d(nn.Module):
    def __init__(self, in_channels, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channels = in_channels * DW_Expand
        self.conv1 = nn.Conv3d(in_channels, dw_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv3d(dw_channels, dw_channels, kernel_size=3, padding=1, stride=1, groups=dw_channels, bias=True)
        self.conv3 = nn.Conv3d(dw_channels // 2, in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dw_channels // 2, dw_channels // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channels = FFN_Expand * in_channels
        self.conv4 = nn.Conv3d(in_channels, ffn_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv3d(ffn_channels // 2, in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm3d(in_channels)
        self.norm2 = LayerNorm3d(in_channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma