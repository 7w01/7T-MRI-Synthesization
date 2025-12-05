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
import torch.nn.functional as F

from .block import NAFBlock3d
from utils import PixelShuffle3d


class NAFNet3d(nn.Module):

    def __init__(self, in_channels=1, embed_channels=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv3d(in_channels, embed_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv3d(embed_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = embed_channels
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock3d(channels) for _ in range(num)]))

            self.downs.append(nn.Conv3d(channels, 2*channels, 2, 2))
            channels = channels * 2

        self.middle_blks = nn.Sequential(*[NAFBlock3d(channels) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv3d(channels, channels * 2, 1, bias=False), PixelShuffle3d(2)))
            channels = channels // 2

            self.decoders.append(nn.Sequential(*[NAFBlock3d(channels) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W, D = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W, :D]

    def check_image_size(self, x):
        _, _, h, w, d = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        mod_pad_d = (self.padder_size - d % self.padder_size) % self.padder_size

        x = F.pad(x, (0, mod_pad_d, 0, mod_pad_w, 0, mod_pad_h))
        return x