import os
import torch
import argparse
import matplotlib.pyplot as plt

from patch_transformation import unpatchify_with_overlap


def args_config():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_seed', type=int, default=32)
    parser.add_argument('--num_fold', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--img_size_3t', type=int, default=[144,176, 160])
    parser.add_argument('--img_size_7t', type=int, default=[288, 352, 320])

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.0001)
    
    # Model
    parser.add_argument('--model', type=str)
    parser.add_argument('--embed_channels', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=[64, 64, 64])
    parser.add_argument('--grid_size', type=int, default=[5, 6, 5])
    parser.add_argument('--overlap', type=int, default=[8, 8, 8])
    parser.add_argument('--num_layers', type=int, default=3)

    # Training
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize results during training')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use Automatic Mixed Precision')

    parser.add_argument('--model_root', type=str, default='models')
    parser.add_argument('--device', type=str, default='cuda:7' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


def visualize(args, whole_img_3t, whole_img_7t, whole_pred_7t, epoch, loss):
    img_3t_vis = whole_img_3t.as_tensor()[:args.grid_length]
    img_7t_vis = whole_img_7t.as_tensor()[:args.grid_length]
    pred_7t_vis = whole_pred_7t.as_tensor()[:args.grid_length]

    whole_img_3t = whole_img_3t[args.grid_length:]
    whole_img_7t = whole_img_7t[args.grid_length:]
    whole_pred_7t = whole_pred_7t[args.grid_length:]

    img_3t_vis = img_3t_vis.cpu().numpy()
    img_7t_vis = img_7t_vis.cpu().numpy()
    pred_7t_vis = pred_7t_vis.cpu().numpy()

    img_3t_vis = unpatchify_with_overlap(img_3t_vis, args.grid_size, args.overlap)
    img_7t_vis = unpatchify_with_overlap(img_7t_vis, args.grid_size, args.overlap)
    pred_7t_vis = unpatchify_with_overlap(pred_7t_vis, args.grid_size, args.overlap)
    
    img_3t_slice_idx = img_3t_vis.shape[3] // 2
    img_7t_slice_idx = img_7t_vis.shape[3] // 2
    pred_7t_slice_idx = pred_7t_vis.shape[3] // 2
    
    img_3t_slice = img_3t_vis[0, :, :, img_3t_slice_idx]
    img_7t_slice = img_7t_vis[0, :, :, img_7t_slice_idx]
    pred_7t_slice = pred_7t_vis[0, :, :, pred_7t_slice_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_3t_slice, cmap='gray')
    axes[0].set_title('3T (Middle Slice)')
    axes[0].axis('off')
    
    axes[1].imshow(img_7t_slice, cmap='gray')
    axes[1].set_title('GT 7T (Middle Slice)')
    axes[1].axis('off')
    
    axes[2].imshow(pred_7t_slice, cmap='gray')
    axes[2].set_title('Pred 7T (Middle Slice)')
    axes[2].axis('off')
    
    plt.suptitle(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    plt.tight_layout()
    
    # Save the figure
    vis_dir = 'result_show'
    plt.savefig(os.path.join(vis_dir, f'{args.model}.png'))
    plt.close()

    return whole_img_3t, whole_img_7t, whole_pred_7t