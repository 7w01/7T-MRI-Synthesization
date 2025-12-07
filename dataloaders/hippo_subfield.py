import os
import torch
import numpy as np

from monai import transforms
from monai.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from patch_transformation import patchify_with_overlap

def HippoSubfieldSeg(args):
    print("=> creating dataloader")
    
    data_path = '/sharedData/datasets/HippoSubfieldSeg'

    img_folder_7T = os.path.join(data_path, 'hss_7T_preproc')
    img_iter_7T = sorted(os.listdir(img_folder_7T))

    img_folder_3T = os.path.join(data_path, 'hss_3T_preproc')
    img_iter_3T = sorted(os.listdir(img_folder_3T))
    
    data_dict = []

    for img_7T, img_3T in zip(img_iter_7T, img_iter_3T):
        img_path_7T = os.path.join(img_folder_7T, img_7T)
        img_path_3T = os.path.join(img_folder_3T, img_3T)
        data_dict.append({"3t": img_path_3T, "7t": img_path_7T})             
    
    args.grid_length = np.prod(args.grid_size)

    # preprocessing
    preproc_data_dict = pre_transforms(args, data_dict)

    # augmentations
    # augmentations = augmentation_transforms()

    # construct dataloader
    train_dataloaders = []
    val_dataloaders = []

    kf = KFold(n_splits=args.num_fold, shuffle=True, random_state=args.data_seed)

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(preproc_data_dict)):

        train_3t = torch.concat([preproc_data_dict[i]["3t"] for i in train_indices], dim=0)
        train_7t = torch.concat([preproc_data_dict[i]["7t"] for i in train_indices], dim=0)

        val_3t = torch.concat([preproc_data_dict[i]["3t"] for i in val_indices], dim=0)
        val_7t = torch.concat([preproc_data_dict[i]["7t"] for i in val_indices], dim=0)

        train_data = [{"3t": img_3t, "7t": img_7t} for img_3t, img_7t in zip(train_3t, train_7t)]
        val_data = [{"3t": img_3t, "7t": img_7t} for img_3t, img_7t in zip(val_3t, val_7t)]
      
        train_dataset = Dataset(data=train_data, transform=None)
        val_dataset = Dataset(data=val_data, transform=None)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False
        )

        train_dataloaders.append(train_dataloader)
        val_dataloaders.append(val_dataloader)

    print("=> finish creating dataloader")

    return train_dataloaders[0], val_dataloaders[0]


class AddChanneld(transforms.MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key][np.newaxis, :]
        return d

class MinMaxNormalizationd(transforms.MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            # Handle negative values
            mask = (img != 0)
            min_val = img.min()
            img = np.where(mask, img + np.abs(min_val), img)

            # min-max normalization
            img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)

            d[key] = img_normalized
        return d

class GridPatch(transforms.MapTransform):
    def __init__(self, keys, patch_size, grid_size, overlap):
        transforms.MapTransform.__init__(self, keys)
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.overlap = overlap

    def __call__(self, data, ):
        d = dict(data)
        for key in self.keys:
            d[key] = patchify_with_overlap(d[key], self.patch_size, self.grid_size, self.overlap)
        return d

def pre_transforms(args, data_dict):
    transform = transforms.Compose(
        [transforms.LoadImaged(keys=["3t", "7t"]),
        AddChanneld(keys=["3t", "7t"]),
        transforms.Orientationd(keys=["3t", "7t"], axcodes="RAS", labels=None),
        transforms.CropForegroundd(
            keys="3t",
            source_key="3t",
            select_fn=lambda x: x > 0,
            margin=0,
            allow_missing_keys=False
        ),
        transforms.CropForegroundd(
            keys="7t",
            source_key="7t",
            select_fn=lambda x: x > 0,
            margin=0,
            allow_missing_keys=False
        ),
        transforms.Resized(
            keys=["3t", "7t"],
            spatial_size=args.img_size_7t,
            # spatial_size=[144, 176, 160], 
            mode=["trilinear", "trilinear"]),
        MinMaxNormalizationd(keys=["3t", "7t"]),
        GridPatch(
            keys=["3t", "7t"],
            patch_size=args.patch_size,
            grid_size=args.grid_size,
            overlap=args.overlap
        ),
        transforms.ToTensord(keys=["3t", "7t"])
        ]
    )

    return transform(data_dict)

# def augmentation_transforms():
#     transform = transforms.Compose(
#         [transforms.LoadImaged(keys=["3t", "7t"]),
#         AddChanneld(keys=["3t", "7t"]),
#         transforms.Orientationd(keys=["3t", "7t"], axcodes="RAS", labels=None),
#         transforms.CropForegroundd(
#             keys="3t",
#             source_key="3t",
#             select_fn=lambda x: x > 0,
#             margin=0,
#             allow_missing_keys=False
#         ),
#         transforms.CropForegroundd(
#             keys="7t",
#             source_key="7t",
#             select_fn=lambda x: x > 0,
#             margin=0,
#             allow_missing_keys=False
#         ),
#         transforms.Resized(
#             keys="3t",
#             spatial_size=[144, 176, 160], 
#             mode="trilinear"),
#         transforms.Resized(
#             keys="7t",
#             spatial_size=[288, 352, 320],
#             mode=("trilinear")),
#         MinMaxNormalizationd(keys=["3t", "7t"]),
#         transforms.ToTensord(keys=["3t", "7t"])
#         ]
#     )
#     return transform