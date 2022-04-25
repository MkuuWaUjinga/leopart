import json
import pytorch_lightning as pl
import os
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from typing import List, Optional, Callable, Tuple, Any


class CocoDataModule(pl.LightningDataModule):

    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 data_dir: str,
                 train_transforms,
                 val_transforms,
                 file_list: List[str],
                 mask_type: str = None,
                 file_list_val: List[str] = None,
                 val_target_transforms=None,
                 shuffle: bool = True,
                 size_val_set: int = 10):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size_val_set = size_val_set
        self.file_list = file_list
        self.file_list_val = file_list_val
        self.data_dir = data_dir
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.file_list_val = file_list_val
        self.val_target_transforms = val_target_transforms
        self.mask_type = mask_type
        self.coco_train = None
        self.coco_val = None

    def __len__(self):
        return len(self.file_list)

    def setup(self, stage: Optional[str] = None):
        # Split test set in val an test
        if self.mask_type is None:
            self.coco_train = UnlabelledCoco(self.file_list,
                                             self.train_transforms,
                                             os.path.join(self.data_dir, "train2017"))
            self.coco_val = UnlabelledCoco(self.file_list[:self.size_val_set * self.batch_size],
                                           self.val_transforms,
                                           os.path.join(self.data_dir, "val2017"))
        else:
            self.coco_train = COCOSegmentation(self.data_dir,
                                               self.file_list,
                                               self.mask_type,
                                               image_set="train",
                                               transforms=self.train_transforms)
            self.coco_val = COCOSegmentation(self.data_dir,
                                             self.file_list_val,
                                             self.mask_type,
                                             image_set="val",
                                             transform=self.val_transforms,
                                             target_transform=self.val_target_transforms)

        print(f"Train size {len(self.coco_train)}")
        print(f"Val size {len(self.coco_val)}")
        print(f"Data Module setup at stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.coco_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.coco_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)


class COCOSegmentation(VisionDataset):

    def __init__(
            self,
            root: str,
            file_names: List[str],
            mask_type: str,
            image_set: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(COCOSegmentation, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        self.file_names = file_names
        self.mask_type = mask_type
        assert self.image_set in ["train", "val"]
        assert mask_type in ["stuff", "thing"]

        # Set mask folder depending on mask_type
        if mask_type == "thing":
            seg_folder = "annotations/panoptic_annotations/semantic_segmentation_{}2017/"
            json_file = "annotations/panoptic_annotations/panoptic_val2017.json"
        elif mask_type == "stuff":
            seg_folder = "annotations/annotations/stuff_annotations/stuff_{}2017_pixelmaps/"
            json_file = "annotations/annotations/stuff_annotations/stuff_val2017.json"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_folder = seg_folder.format(image_set)

        # Load categories to category to id map for merging to coarse categories
        with open(os.path.join(root, json_file)) as f:
            an_json = json.load(f)
            all_cat = an_json['categories']
            if mask_type == "thing":
                all_thing_cat_sup = set(cat_dict["supercategory"] for cat_dict in all_cat if cat_dict["isthing"] == 1)
                super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(all_thing_cat_sup))}
                self.cat_id_map = {}
                for cat_dict in all_cat:
                    if cat_dict["isthing"] == 1:
                        self.cat_id_map[cat_dict["id"]] = super_cat_to_id[cat_dict["supercategory"]]
                    elif cat_dict["isthing"] == 0:
                        self.cat_id_map[cat_dict["id"]] = 255
            else:
                super_cats = set([cat_dict['supercategory'] for cat_dict in all_cat])
                super_cats.remove("other")  # remove others from prediction targets as this is not semantic
                super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(super_cats))}
                super_cat_to_id["other"] = 255  # ignore_index for CE
                self.cat_id_map = {cat_dict['id']: super_cat_to_id[cat_dict['supercategory']] for cat_dict in all_cat}

        # Get images and masks fnames
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, "coco", "images", f"{image_set}2017")
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir):
            print(seg_dir)
            print(image_dir)
            raise RuntimeError('Dataset not found or corrupted.')
        self.images = [os.path.join(image_dir, x) for x in self.file_names]
        self.masks = [os.path.join(seg_dir, x.replace("jpg", "png")) for x in self.file_names]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        if self.transforms:
            img, mask = self.transforms(img, mask)

        if self.mask_type == "stuff":
            # move stuff labels from {0} U [92, 183] to [0,15] and [255] with 255 == {0, 183}
            # (183 is 'other' and 0 is things)
            mask *= 255
            assert torch.max(mask).item() <= 183
            mask[mask == 0] = 183  # [92, 183]
            assert torch.min(mask).item() >= 92
            for cat_id in torch.unique(mask):
                mask[mask == cat_id] = self.cat_id_map[cat_id.item()]

            assert torch.max(mask).item() <= 255
            assert torch.min(mask).item() >= 0
            mask /= 255
            return img, mask
        elif self.mask_type == "thing":
            mask *= 255
            assert torch.max(mask).item() <= 200
            mask[mask == 0] = 200  # map unlabelled to stuff
            merged_mask = mask.clone()
            for cat_id in torch.unique(mask):
                merged_mask[mask == cat_id] = self.cat_id_map[int(cat_id.item())]  # [0, 11] + {255}

            assert torch.max(merged_mask).item() <= 255
            assert torch.min(merged_mask).item() >= 0
            merged_mask /= 255
            return img, merged_mask
        return img, mask


class UnlabelledCoco(Dataset):

    def __init__(self, file_list, transforms, data_dir):
        self.file_names = file_list
        self.transform = transforms
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = self.file_names[idx]
        image = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
