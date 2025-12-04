import os
import torch
import numpy as np
import pandas as pd

from monai import transforms
from monai.data import Dataset, DataLoader

from sklearn.model_selection import KFold


def HippoSubfieldSeg(args):
    print("=> creating dataloader")
    
    data_path = '/sharedData/datasets/HippoSubfieldSeg'


    transform = get_transforms()

    img_folder_7T = os.path.join(data_path, 'hss_7T_preproc')
    img_iter_7T = sorted(os.listdir(img_folder_7T))

    img_folder_3T = os.path.join(data_path, 'hss_3T_preproc')
    img_iter_3T = sorted(os.listdir(img_folder_3T))
    
    data_dict = []

    for img_7T, img_3T in zip(img_iter_7T, img_iter_3T):
        img_path_7T = os.path.join(img_folder_7T, img_7T)
        img_path_3T = os.path.join(img_folder_3T, img_3T)
        data_dict.append({"image": img_path_3T, "label": img_path_7T})             

    # construct dataloader
    train_dataloaders = []
    val_dataloaders = []

    kf = KFold(n_splits=args.num_fold, shuffle=True, random_state=args.data_seed)

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(data_dict)):

        train_data = [data_dict[i] for i in train_indices]
        val_data = [data_dict[i] for i in val_indices]

        train_dataset = Dataset(data=train_data, transform=transform)
        val_dataset = Dataset(data=val_data, transform=transform)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=True
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


class LabelCombination(transforms.MapTransform):
    """
    Combine Label_L and Label_R
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            value = d[key]
            value[value > 0] = 1 
            d[key] = value
        return d


class ExtendIntensityd(transforms.MapTransform):
    """
    Combine Label_L and Label_R
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            value = d[key]
            value[value<-0.5] = value[value<-0.5] * 2 + 0.5
            
            d[key] = value
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

            # value = d[key]
            # value[value<-3] = -3
            # value[value>3] = 3
        
            # d[key] = value

def get_transforms():
    transform = transforms.Compose(
        [transforms.LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        transforms.CropForegroundd(
            keys="image",
            source_key="image",
            select_fn=lambda x: x > 0,
            margin=0,
            allow_missing_keys=False
        ),
        transforms.CropForegroundd(
            keys="label",
            source_key="label",
            select_fn=lambda x: x > 0,
            margin=0,
            allow_missing_keys=False
        ),
        transforms.Resized(
            keys="image",
            spatial_size=[144, 176, 160], 
            mode="trilinear"),
        transforms.Resized(
            keys="label",
            spatial_size=[288, 352, 320],
            mode=("trilinear")),
        MinMaxNormalizationd(keys=["image", "label"]),
        transforms.ToTensord(keys=["image", "label"])
        ]
    )
    return transform
