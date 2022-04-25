import os
import pytorch_lightning as pl

from typing import Optional, Callable
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing import Tuple, Any


class VOCDataModule(pl.LightningDataModule):

    CLASS_IDX_TO_NAME = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                         'train', 'tvmonitor']

    def __init__(self,
                 data_dir: str,
                 train_split: str,
                 val_split: str,
                 train_image_transform: Optional[Callable],
                 val_image_transform: Optional[Callable],
                 val_target_transform: Optional[Callable],
                 batch_size: int,
                 num_workers: int,
                 shuffle: bool = True,
                 return_masks: bool = False,
                 drop_last: bool = True):
        """
        Data module for PVOC data. "trainaug" and "train" are valid train_splits.
        If return_masks is set train_image_transform should be callable with imgs and masks or None.
        """
        super().__init__()
        self.root = os.path.join(data_dir, "VOCSegmentation")
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.return_masks = return_masks

        # Set up datasets in __init__ as we need to know the number of samples to init cosine lr schedules
        assert train_split == "trainaug" or train_split == "train"
        self.voc_train = VOCDataset(root=self.root, image_set=train_split, transforms=self.train_image_transform,
                                    return_masks=self.return_masks)
        self.voc_val = VOCDataset(root=self.root, image_set=val_split, transform=self.val_image_transform,
                                  target_transform=self.val_target_transform)

    def __len__(self):
        return len(self.voc_train)

    def class_id_to_name(self, i: int):
        return self.CLASS_IDX_TO_NAME[i]

    def setup(self, stage: Optional[str] = None):
        print(f"Train size {len(self.voc_train)}")
        print(f"Val size {len(self.voc_val)}")

    def train_dataloader(self):
        return DataLoader(self.voc_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.voc_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)


class TrainXVOCValDataModule(pl.LightningDataModule):
    # wrapper class to allow for training on a different data set

    def __init__(self, train_datamodule: pl.LightningDataModule, val_datamodule: VOCDataModule):
        super().__init__()
        self.train_datamodule = train_datamodule
        self.val_datamodule = val_datamodule

    def setup(self, stage: str = None):
        self.train_datamodule.setup(stage)
        self.val_datamodule.setup(stage)

    def class_id_to_name(self, i: int):
        return self.val_datamodule.class_id_to_name(i)

    def __len__(self):
        return len(self.train_datamodule)

    def train_dataloader(self):
        return self.train_datamodule.train_dataloader()

    def val_dataloader(self):
        return self.val_datamodule.val_dataloader()


class VOCDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            image_set: str = "trainaug",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = False
    ):
        super(VOCDataset, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        if self.image_set == "trainaug" or self.image_set == "train":
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, 'images')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(root, 'sets')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]
        self.return_masks = return_masks

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.image_set == "val":
            mask = Image.open(self.masks[index])
            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask
        elif "train" in self.image_set:
            if self.transforms:
                if self.return_masks:
                    mask = Image.open(self.masks[index])
                    res = self.transforms(img, mask)
                else:
                    res = self.transforms(img)
                return res
            return img

    def __len__(self) -> int:
        return len(self.images)
