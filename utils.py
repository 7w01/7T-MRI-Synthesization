import torch
import numpy as np
import matplotlib.pyplot as plt


def patchify(img, patch_size, grid_size):
    """
    Convert 3D images into patches.
    """
    B, C, H, W, D = img.shape
    ph, pw, pd = patch_size
    gh, gw, gd = grid_size

    patches = (img
        .reshape(B, C, gh, ph, gw, pw, gd, pd)
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(B, -1, C * ph * pw * pd)
    )
    return patches


def unpatchify(patches, patch_size, grid_size):
    """
    Reconstruct 3D images from patches.
    """
    B, N, _ = patches.shape
    ph, pw, pd = patch_size
    gh, gw, gd = grid_size

    img = (patches
        .reshape(B, gh, gw, gd, -1, ph, pw, pd)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(B, -1, gh * ph, gw * pw, gd * pd)
    )
    return img


class PixelShuffle3d(torch.nn.Module):
    """
    3D version of PixelShuffle operation.
    Rearranges elements in a tensor of shape (*, C*r^3, H, W, D) to a tensor of shape (*, C, H*r, W*r, D*r).
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle3d, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        B, C, H, W, D = inputs.size()
        upscale_factor = self.upscale_factor
        
        out_height = H * upscale_factor
        out_width = W * upscale_factor
        out_depth = D * upscale_factor
        
        orig_channels = C // (upscale_factor ** 3)
        
        if C % (upscale_factor ** 3) != 0:
            raise RuntimeError("Input C must be divisible by (upscale_factor ** 3)")
            
        view = inputs.contiguous().view(
            B, orig_channels,
            upscale_factor, upscale_factor, upscale_factor,
            H, W, D 
        )
        
        permute = view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        
        outputs = permute.view(B, orig_channels, out_height, out_width, out_depth)
        
        return outputs