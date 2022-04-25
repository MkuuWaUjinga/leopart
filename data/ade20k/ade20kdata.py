import numpy as np
import os
import torch
import pickle
import pytorch_lightning as pl

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import data.ade20k.utils_ade20k as utils

from typing import Optional

from PIL import Image
from PIL.Image import NEAREST


class Ade20kDataModule(pl.LightningDataModule):

    def __init__(self,
                 root,
                 train_transforms,
                 val_transforms,
                 shuffle,
                 num_workers,
                 batch_size,
                 val_target_transforms):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.val_target_transforms = val_target_transforms

    def setup(self, stage: Optional[str] = None):
        # Split test set in val an test
        if stage == 'fit' or stage is None:
            train_len = 25258
            train_start_index = 0
            val_len = 2000
            self.val = Ade20KPartsDataset(self.root, train_len + val_len, train_start_index,
                                          img_transform=self.val_transforms,
                                          mask_transform=self.val_target_transforms)
            self.train = self.val
            print(f"Val size {len(self.val)}")
        else:
            raise NotImplementedError("Unlabelled NEON doesn't have a dedicated val/test set.")
        print(f"Data Module setup at stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)


class Ade20KPartsDataset(Dataset):
    def __init__(self, root, val_len, val_start_index, mask_transform=None, img_transform=None):
        self.root = root
        with open(os.path.join(self.root, "ADE20K_2021_17_01/index_ade20k.pkl"), 'rb') as f:
            index_ade20k = pickle.load(f)

        # get val idx  and img idst
        self.idx = self.construct_index(index_ade20k, val_start_index, val_len)
        # select images that have object part annotations and are street scenes
        part_img_ids = np.sum(self.idx["objectIsPart"], axis=0).nonzero()[0]
        street_img_idx = (np.array(self.idx["scene"]) == "/street").nonzero()[0]
        print(f"Found {len(street_img_idx)} street images")
        self.image_ids = list(set(part_img_ids).intersection(set(street_img_idx)))
        print(f"Found {len(self.image_ids)} street images with parts annotations")
        # construct part id map
        # all part ids found during iteration through whole dataset
        all_part_ids = [50, 51, 54, 83, 101, 112, 135, 136, 144, 175, 184, 211, 213, 277, 320, 495, 543, 544, 580, 582,
                        626, 665, 772, 774, 776, 777, 781, 783, 785, 840, 859, 860, 876, 890, 904, 909, 934, 938, 1062,
                        1063, 1072, 1081, 1140, 1145, 1156, 1180, 1206, 1212, 1213, 1249, 1259, 1277, 1279, 1280, 1395,
                        1397, 1412, 1428, 1429, 1430, 1431, 1439, 1470, 1540, 1541, 1564, 1882, 1883, 1936, 1951, 1957,
                        2052, 2067, 2103, 2117, 2118, 2119, 2120, 2122, 2130, 2155, 2156, 2164, 2190, 2346, 2370, 2371,
                        2376, 2379, 2421, 2529, 2564, 2567, 2570, 2700, 2742, 2820, 2828, 2855, 2884, 2940, 2978, 3035,
                        3050, 3054, 3056, 3057, 3063, 3137, 3153, 3154]
        print(f"Found {len(all_part_ids)} parts")
        self.part_id_map = {part_id: i for i, part_id in enumerate(all_part_ids)}
        self.part_id_map[0] = 255 # map no parts pixels to ignore index
        self.mask_transform = mask_transform
        self.img_transform = img_transform

    def construct_index(self, index_ade20k, start_idx, length):
        index_ade20k["filename"] = index_ade20k["filename"][start_idx:start_idx+length]
        index_ade20k["folder"] = index_ade20k["folder"][start_idx:start_idx+length]
        index_ade20k["objectIsPart"] = index_ade20k["objectIsPart"][:, start_idx: start_idx + length]
        index_ade20k["objectPresence"] = index_ade20k["objectPresence"][:, start_idx: start_idx + length]
        index_ade20k["scene"] = index_ade20k["scene"][start_idx:start_idx + length]
        return index_ade20k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        i = self.image_ids[idx]
        full_file_path = os.path.join(self.idx['folder'][i], self.idx['filename'][i])
        info = utils.loadAde20K(os.path.join(self.root, full_file_path))
        img = Image.open(info['img_name'])
        img = self.img_transform(img) # normalize and resize image
        parts_mask = torch.from_numpy(info['partclass_mask'][0]).unsqueeze(0).float() # only keep parts not parts of parts
        parts_mask = self.mask_transform(parts_mask) # resize parts mask

        # zero index and linearize class part classes. Assign non parts ignore index
        linearized_mask = parts_mask.clone()
        for part_id in torch.unique(parts_mask):
            linearized_mask[parts_mask == part_id] = self.part_id_map[part_id.item()]
        linearized_mask /= 255

        return img, linearized_mask
