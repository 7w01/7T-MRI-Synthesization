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