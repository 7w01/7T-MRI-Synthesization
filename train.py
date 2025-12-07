import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils import args_config, visualize

from dataloaders.hippo_subfield import HippoSubfieldSeg
from models.VNet.arch import *


if __name__ == '__main__':

    args = args_config()
    
    print(f"Using device: {args.device}")
    print(f"\nModel: {args.model}")

    # initialize model
    model = VNet().to(args.device)

    # load dataloaders
    train_dataloader, val_dataloader = HippoSubfieldSeg(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    for epoch in range(args.num_epochs):
        train_loss = []
                
        whole_img_3t = torch.tensor([]).to(args.device)
        whole_img_7t = torch.tensor([]).to(args.device)
        whole_pred_7t = torch.tensor([]).to(args.device)

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                            desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")

        for i, data in progress_bar:
            img_3t, img_7t = data['3t'].to(args.device), data['7t'].to(args.device)

            pred_7t = model(img_3t)
            loss = F.mse_loss(pred_7t, img_7t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # push patches for merge to a whole image
            if whole_img_3t.numel() == 0:
                whole_img_3t = img_3t
                whole_img_7t = img_7t
                whole_pred_7t = pred_7t
            else:
                whole_img_3t = torch.concat((whole_img_3t, img_3t), dim=0)
                whole_img_7t = torch.concat((whole_img_7t, img_7t), dim=0)
                whole_pred_7t = torch.concat((whole_pred_7t, pred_7t), dim=0)

            progress_bar.set_postfix({"Loss": loss.item()})     
            train_loss.append(loss.item())

            # Visualize
            if args.visualize and (whole_img_3t.shape[0] >= args.grid_length):
                with torch.no_grad():
                    whole_img_3t, whole_img_7t, whole_pred_7t = \
                        visualize(args, whole_img_3t, whole_img_7t, whole_pred_7t, epoch, loss)
            
        print(f"Loss: {sum(train_loss) / len(train_loss)}")