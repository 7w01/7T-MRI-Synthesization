import torch
import numpy as np
import torchio as tio


def patchify(img, patch_size, grid_size):
    """
    Convert 3D images into patches.
    """
    C, H, W, D = img.shape
    ph, pw, pd = patch_size
    gh, gw, gd = grid_size

    # Calculate required size
    required_h = gh * ph
    required_w = gw * pw
    required_d = gd * pd
   
    # Pad image if necessary
    pad_h = max(0, required_h - H)
    pad_w = max(0, required_w - W)
    pad_d = max(0, required_d - D)

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        # Pad with zeros
        img = np.pad(img, ((0, 0), 
                            (pad_h//2, pad_h - pad_h//2), 
                            (pad_w//2, pad_w - pad_w//2), 
                            (pad_d//2, pad_d - pad_d//2)), 
                            mode='constant', constant_values=0)

    patches = (img
        .reshape(C, gh, ph, gw, pw, gd, pd)
        .transpose(1, 3, 5, 0, 2, 4, 6)
        .reshape(gh * gw * gd, C, ph, pw, pd)
    )
    return patches


def unpatchify(patches, grid_size):
    """
    Reconstruct 3D images from patches.
    """
    N, C, ph, pw, pd = patches.shape
    gh, gw, gd = grid_size

    img = (patches
        .reshape(gh, gw, gd, C, ph, pw, pd)
        .permute(3, 0, 4, 1, 5, 2, 6)
        .reshape(C, gh * ph, gw * pw, gd * pd)
    )
    return img


def patchify_with_overlap(img, patch_size, grid_size, overlap):
    """
    Convert 3D images into overlapping patches.
    
    Args:
        img: 3D image tensor of shape (C, H, W, D)
        patch_size: tuple of patch dimensions (ph, pw, pd)
        grid_size: tuple of grid dimensions (gh, gw, gd)
        overlap: tuple of overlap dimensions (oh, ow, od)
        
    Returns:
        patches: tensor of shape (num_patches, C, ph, pw, pd)
    """
    C, H, W, D = img.shape
    ph, pw, pd = patch_size
    gh, gw, gd = grid_size
    oh, ow, od = overlap

    # Calculate step size
    assert (oh > 1) and (ow > 1) and (od > 1), "overlap must > 1 !"
    sh = ph - oh
    sw = pw - ow
    sd = pd - od

    # Calculate required size with overlap
    required_h = (gh - 1) * sh + ph
    required_w = (gw - 1) * sw + pw
    required_d = (gd - 1) * sd + pd
   
    # Pad image if necessary
    pad_h = max(0, required_h - H)
    pad_w = max(0, required_w - W)
    pad_d = max(0, required_d - D)

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        # Pad with zeros
        img = np.pad(img, ((0, 0), 
                            (pad_h//2, pad_h - pad_h//2), 
                            (pad_w//2, pad_w - pad_w//2), 
                            (pad_d//2, pad_d - pad_d//2)), 
                            mode='constant', constant_values=0)

    # Extract overlapping patches
    patches = []
    for h in range(gh):
        for w in range(gw):
            for d in range(gd):
                start_h = h * sh
                start_w = w * sw
                start_d = d * sd
                
                patch = img[:, start_h:start_h+ph, start_w:start_w+pw, start_d:start_d+pd]
                patches.append(patch)
                
    patches = np.stack(patches, axis=0)
    return patches


def unpatchify_with_overlap(patches, grid_size, overlap, crop_border=1):
    """
    Reconstruct 3D images from overlapping patches using averaging for overlapping regions.
    Crop borders to reduce boundary effects from convolution operations.
    
    Args:
        patches: tensor of shape (num_patches, C, ph, pw, pd)
        grid_size: tuple of grid dimensions (gh, gw, gd)
        overlap: tuple of overlap dimensions (oh, ow, od)
        crop_border: number of pixels to crop from each border (default: 1)
        
    Returns:
        img: reconstructed 3D image tensor of shape (C, H, W, D)
    """
    N, C, ph, pw, pd = patches.shape
    gh, gw, gd = grid_size
    oh, ow, od = overlap

    # Calculate step size
    if crop_border > 0:
        patches = patches[:, :, crop_border:-crop_border, crop_border:-crop_border, crop_border:-crop_border]

        ph = ph - 2 * crop_border
        pw = pw - 2 * crop_border
        pd = pd - 2 * crop_border
        
        sh = ph - (oh - 2 * crop_border)
        sw = pw - (ow - 2 * crop_border)
        sd = pd - (od - 2 * crop_border)
    else:
        sh = ph - oh
        sw = pw - ow
        sd = pd - od
        
    H = (gh - 1) * sh + ph
    W = (gw - 1) * sw + pw
    D = (gd - 1) * sd + pd

    # Initialize output image and count arrays
    img = np.zeros((C, H, W, D), dtype=np.float32)
    count = np.zeros((C, H, W, D), dtype=np.float32)

    # Place patches back into image with averaging for overlaps
    patch_idx = 0
    for h in range(gh):
        for w in range(gw):
            for d in range(gd):
                start_h = h * sh
                start_w = w * sw
                start_d = d * sd
                
                img[:, start_h:start_h+ph, start_w:start_w+pw, start_d:start_d+pd] += patches[patch_idx]
                count[:, start_h:start_h+ph, start_w:start_w+pw, start_d:start_d+pd] += 1
                patch_idx += 1

    # Average overlapping regions
    img = img / count
    return img
