import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from dataloaders.hippo_subfield import HippoSubfieldSeg
from models.VNet.arch import *
import matplotlib.pyplot as plt
import os

from torch.amp import autocast, GradScaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_seed', type=int, default=32)
    parser.add_argument('--num_fold', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size_3t', type=int, default=[144,176, 160])
    parser.add_argument('--image_size_7t', type=int, default=[288, 352, 320])

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.0001)
    
    # Model
    parser.add_argument('--embed_channels', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=[1, 1, 1])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--window_sizes', type=int, default=[[2, 2, 2]])

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize results during training')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use Automatic Mixed Precision')

    parser.add_argument('--model_root', type=str, default='models')
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Using AMP: {args.use_amp}")

    # model = FNO_Unet3D(
    #     embed_dim=args.embed_channels,
    #     num_layers=args.num_layers,
    # ).to(args.device)

    model = VNet().to(args.device)

    # load dataloaders
    train_dataloader, val_dataloader = HippoSubfieldSeg(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    scaler = GradScaler() if args.use_amp else None

    for epoch in range(args.num_epochs):
        train_loss = []

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                            desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for i, data in progress_bar:
            image, label = data['image'].to(args.device), data['label'].to(args.device)

            image = F.interpolate(image, size=args.image_size_7t, mode='trilinear', align_corners=False)

            if args.use_amp and scaler is not None:
                with autocast(device_type='cuda'):
                    systhesis = model(image)
                    loss = F.mse_loss(systhesis, label)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                systhesis = model(image)
                loss = F.mse_loss(systhesis, label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Visualize
            if args.visualize:
                with torch.no_grad():
                    if args.use_amp and scaler is not None:
                        image_vis = image.float()
                        label_vis = label.float()
                        systhesis_vis = systhesis.float()
                    else:
                        image_vis = image
                        label_vis = label
                        systhesis_vis = systhesis
                    
                    original_img = image_vis[0, 0].cpu().numpy()
                    true_label = label_vis[0, 0].cpu().numpy()
                    pred_label = systhesis_vis[0, 0].cpu().numpy()
                    
                    orig_slice_idx = original_img.shape[2] // 2
                    label_slice_idx = true_label.shape[2] // 2
                    pred_slice_idx = pred_label.shape[2] // 2
                    
                    orig_slice = original_img[:, :, orig_slice_idx]
                    true_slice = true_label[:, :, label_slice_idx]
                    pred_slice = pred_label[:, :, pred_slice_idx]
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(orig_slice, cmap='gray')
                    axes[0].set_title('Original Image (Middle Slice)')
                    axes[0].axis('off')
                    
                    axes[1].imshow(true_slice, cmap='gray')
                    axes[1].set_title('True Label (Middle Slice)')
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred_slice, cmap='gray')
                    axes[2].set_title('Predicted Label (Middle Slice)')
                    axes[2].axis('off')
                    
                    plt.suptitle(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}')
                    plt.tight_layout()
                    
                    # Save the figure
                    vis_dir = 'visualize'
                    os.makedirs(vis_dir, exist_ok=True)
                    plt.savefig(os.path.join(vis_dir, f'VNet.png'))
                    plt.close()

            progress_bar.set_postfix({"Loss": loss.item()})     
            train_loss.append(loss.item())

        print(f"Loss: {sum(train_loss) / len(train_loss)}")