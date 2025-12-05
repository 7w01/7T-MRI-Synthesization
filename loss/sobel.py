import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from dataloaders.hippo_subfield import HippoSubfieldSeg
import argparse
import matplotlib.pyplot as plt
import os


def sobel_operator_3d(image):
    device = image.device
    
    B, C, H, W, D = image.shape

    sobel_x_kernel = torch.tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ], dtype=torch.float32, device=device).reshape(1, 1, 3, 3, 3)
    
    sobel_y_kernel = torch.tensor([
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ], dtype=torch.float32, device=device).reshape(1, 1, 3, 3, 3)
    
    sobel_z_kernel = torch.tensor([
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    ], dtype=torch.float32, device=device).reshape(1, 1, 3, 3, 3)
    
    grad_x = F.conv3d(image, sobel_x_kernel, padding=1)
    grad_y = F.conv3d(image, sobel_y_kernel, padding=1)
    grad_z = F.conv3d(image, sobel_z_kernel, padding=1)
    
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    
    return magnitude


def visualize_sobel_results(image, magnitude, save_path='sobel_visualization.png'):

    image_np = image.detach().cpu().numpy()
    magnitude_np = magnitude.detach().cpu().numpy()
    
    slice_idx = image_np.shape[-1] // 2
    
    image_slice = image_np[0, 0, :, :, slice_idx]
    magnitude_slice = magnitude_np[0, 0, :, :, slice_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    im1 = axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    im2 = axes[1].imshow(magnitude_slice, cmap='gray')
    axes[1].set_title('Sobel Edge Detection')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--data_seed', type=int, default=32)
    parser.add_argument('--num_fold', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size_3t', type=int, default=[144,176, 160])
    parser.add_argument('--image_size_7t', type=int, default=[288, 352, 320])
    args = parser.parse_args()

    train_dataloader, val_dataloader = HippoSubfieldSeg(args)

    for data in train_dataloader:
        image =  data['label'].to(args.device)

        magnitude = sobel_operator_3d(image)
        
        visualize_sobel_results(image, magnitude, 'visualize/sobel_result.png')