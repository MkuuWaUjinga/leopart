import click
import numpy as np
import os
import torch
import pandas as pd
import pytorch_lightning as pl
import random
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import torchvision.transforms as T
import sacred

from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from torchvision.transforms.functional import InterpolationMode
from typing import List, Any, Tuple

from data.ade20k.ade20kdata import Ade20kDataModule
from data.VOCdevkit.vocdata import VOCDataModule
from data.coco.coco_data_module import CocoDataModule
from experiments.utils import PredsmIoU, get_backbone_weights, normalize_and_transform, cluster
from src.vit import vit_small, vit_base
from src.resnet import ResnetDilated

ex = sacred.experiment.Experiment()
api_key = '<your neptune key>'


@click.command()
@click.option("--config_path", type=str)
def entry(config_path):
    if config_path is not None:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), config_path))
    else:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), "overcluster_dev.yml"))
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    ex_name = f"overclustering-{time}"
    checkpoint_dir = os.path.join(ex.configurations[0]._conf["val"]["ckpt_dir"], ex_name)
    ex.observers.append(sacred.observers.FileStorageObserver(checkpoint_dir))
    ex.run(options={'--name': ex_name})


@ex.main
@ex.capture
def overcluster(_config, _run):
    # Init logger
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project_name='<your project name>',
        experiment_name=_run.experiment_info["name"],
        params=pd.json_normalize(_config).to_dict(orient='records')[0],
        tags=_config["tags"].split(','),
    )
    print("Config:")
    print(_config)
    data_config = _config["data"]
    val_config = _config["val"]
    input_size = data_config["size_crops"]
    num_seeds = val_config["num_seeds"]

    # Init data and transforms
    train_transforms = None
    val_image_transforms = T.Compose([T.Resize((input_size, input_size)),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])

    data_dir = data_config["data_dir"]
    dataset_name = data_config["dataset_name"]
    if dataset_name == "voc":
        ignore_index = 255
        num_classes = 21
        data_module = VOCDataModule(batch_size=val_config["batch_size"],
                                    return_masks=True,
                                    num_workers=_config["num_workers"],
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_dir,
                                    train_image_transform=train_transforms,
                                    drop_last=True,
                                    val_image_transform=val_image_transforms,
                                    val_target_transform=val_target_transforms)
    elif "coco" in dataset_name:
        assert len(dataset_name.split("-")) == 2
        mask_type = dataset_name.split("-")[-1]
        assert mask_type in ["stuff", "thing"]
        if mask_type == "thing":
            num_classes = 12
        else:
            num_classes = 15
        ignore_index = 255
        file_list = os.listdir(os.path.join(data_dir, "coco", "images", "train2017"))
        file_list_val = os.listdir(os.path.join(data_dir, "coco", "images", "val2017"))
        random.shuffle(file_list_val)

        data_module = CocoDataModule(batch_size=val_config["batch_size"],
                                     num_workers=_config["num_workers"],
                                     file_list=file_list,
                                     data_dir=data_dir,
                                     file_list_val=file_list_val,
                                     mask_type=mask_type,
                                     train_transforms=train_transforms,
                                     val_transforms=val_image_transforms,
                                     val_target_transforms=val_target_transforms)
    elif dataset_name == "ade20k":
        num_classes = 111
        ignore_index = 255
        val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST)])
        data_module = Ade20kDataModule(data_dir, train_transforms=None,
                                       val_target_transforms=val_target_transforms, val_transforms=val_image_transforms,
                                       shuffle=False, batch_size=val_config["batch_size"],
                                       num_workers=_config["num_workers"])
    else:
        raise ValueError(f"{dataset_name} not supported")

    # Init method
    arch = val_config["arch"]
    patch_size = val_config["patch_size"]
    restart = val_config["restart"]
    method = val_config["method"]
    spatial_res = input_size / patch_size
    assert spatial_res.is_integer()
    model = Overcluster(
        patch_size=patch_size,
        arch=arch,
        pca_dim=val_config["pca_dim"],
        k=val_config["K"],
        num_classes=num_classes,
        spatial_res=int(spatial_res),
        num_seeds=num_seeds,
        ignore_index=ignore_index,
        mask_eval_size=100
    )

    # Optionally load weights
    if not restart and val_config["method"] != "random":
        weights = get_backbone_weights(arch, method, patch_size=patch_size, ckpt_path=val_config.get("ckpt_path"))
        msg = model.load_state_dict(weights, strict=False)
        print(msg)

    # Only do a validation loop to get embeddings
    trainer = Trainer(
        logger=neptune_logger,
        gpus=_config["gpus"],
        accelerator='ddp' if _config["gpus"] > 1 else None,
        fast_dev_run=val_config["fast_dev_run"],
        terminate_on_nan=True,
    )
    trainer.validate(model, datamodule=data_module)


class Overcluster(pl.LightningModule):

    def __init__(self, patch_size: int, num_classes: int, k: int, pca_dim: int, arch: str, spatial_res: int,
                 num_seeds: int, mask_eval_size: int = 100, ignore_index: int = 255):
        super().__init__()
        self.save_hyperparameters()

        # Init Model
        if arch == 'vit-small':
            self.model = vit_small(patch_size=patch_size)
        elif arch == 'resnet50':
            backbone = resnet.__dict__[arch](pretrained=False)
            self.model = ResnetDilated(backbone)
        elif arch == 'vit-base':
            self.model = vit_base(patch_size=patch_size)

        self.arch = arch
        self.num_seeds = num_seeds
        self.spatial_res = spatial_res
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.pca_dim = pca_dim
        self.k = k
        self.masks_eval_size = mask_eval_size

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.model.eval()
        with torch.no_grad():
            imgs, masks = batch
            bs = imgs.size(0)
            tokens = self.model.forward_backbone(imgs)
            if self.arch == 'resnet50':
                tokens = tokens.permute(0, 2, 3, 1).reshape(bs * self.spatial_res ** 2,  self.model.embed_dim)
            elif 'vit' in self.arch:
                tokens = tokens[:, 1:]
                tokens = tokens.reshape(bs * self.spatial_res ** 2, self.model.embed_dim)

            # Downsample masks to self.masks_eval_size
            masks *= 255
            if masks.size(3) != self.masks_eval_size:
                masks = F.interpolate(masks, size=(self.masks_eval_size, self.masks_eval_size), mode='nearest')
            return tokens.cpu(), masks.cpu()

    def validation_epoch_end(self, outputs: List[Any]):
        print("collected data")
        tokens = torch.cat([out[0].cpu() for out in outputs])
        masks = torch.cat([out[1].cpu() for out in outputs])
        print(f"Start normalization")
        normalized_feats = normalize_and_transform(tokens, self.pca_dim)
        clusterings = []
        for i in range(self.num_seeds):
            clusterings.append(cluster(self.pca_dim, normalized_feats.numpy(), self.spatial_res, self.k, seed=i))

        results = []
        print(f"Number of pixels ignored is: {torch.sum(masks == self.ignore_index)}")
        for clustering in clusterings:
            clustering = F.interpolate(clustering.float(), size=(self.masks_eval_size, self.masks_eval_size),
                                       mode='nearest')
            metric = PredsmIoU(self.k, self.num_classes)
            metric.update(masks[masks != self.ignore_index], clustering[masks != self.ignore_index])
            if self.k == self.num_classes:
                results.append(metric.compute(True, many_to_one=False, precision_based=False)[0])
            else:
                results.append(metric.compute(True, many_to_one=True, precision_based=True)[0])
        print(results)
        print(np.mean(results))


if __name__ == "__main__":
    entry()
