import click
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
import torchvision.models.resnet as resnet
import torchvision.transforms as T
import sacred

from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms.functional import InterpolationMode
from typing import List, Any, Tuple

from data.VOCdevkit.vocdata import VOCDataModule
from data.coco.coco_data_module import CocoDataModule
from experiments.utils import PredsmIoU, get_backbone_weights
from experiments.linear_probing.fcn.fcn_head import FCNHead
from src.resnet import ResnetDilated
from src.vit import vit_small, vit_base
from src.linear_finetuning_transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor

ex = sacred.experiment.Experiment()
api_key = "<your api token>"

@click.command()
@click.option("--config_path", type=str)
def entry(config_path):
    if config_path is not None:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), config_path))
    else:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), "finetune_dev.yml"))
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    ex_name = f"linear-finetune-{time}"
    checkpoint_dir = os.path.join(ex.configurations[0]._conf["train"]["ckpt_dir"], ex_name)
    ex.observers.append(sacred.observers.FileStorageObserver(checkpoint_dir))
    ex.run(config_updates={'seed': 400}, options={'--name': ex_name})

@ex.main
@ex.capture
def linear_finetune(_config, _run):
    # Init logger
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project_name="<your project name>",
        experiment_name=_run.experiment_info["name"],
        params=pd.json_normalize(_config).to_dict(orient='records')[0],
        tags=_config["tags"].split(','),
    )
    print("Config:")
    print(_config)
    data_config = _config["data"]
    train_config = _config["train"]
    seed_everything(_config["seed"])
    input_size = data_config["size_crops"]

    # Init transforms and train data
    train_transforms = Compose([
        RandomResizedCrop(size=input_size, scale=(0.8, 1.)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_image_transforms = T.Compose([T.Resize((input_size, input_size)),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])

    data_dir = data_config["data_dir"]
    dataset_name = data_config["dataset_name"]
    if dataset_name == "voc":
        num_classes = 21
        ignore_index = 255
        data_module = VOCDataModule(batch_size=train_config["batch_size"],
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
        assert mask_type in ["thing", "stuff"]
        if mask_type == "thing":
            num_classes = 12
        else:
            num_classes = 15
        ignore_index = 255
        file_list = os.listdir(os.path.join(data_dir, "coco", "images", "train2017"))
        file_list_val = os.listdir(os.path.join(data_dir, "coco", "images", "val2017"))
        random.shuffle(file_list_val)
        # sample 10% of train images
        random.shuffle(file_list)
        file_list = file_list[:int(len(file_list)*0.1)]
        print(f"sampled {len(file_list)} COCO images for training")

        data_module = CocoDataModule(batch_size=train_config["batch_size"],
                                     num_workers=_config["num_workers"],
                                     file_list=file_list,
                                     data_dir=data_dir,
                                     file_list_val=file_list_val,
                                     mask_type=mask_type,
                                     train_transforms=train_transforms,
                                     val_transforms=val_image_transforms,
                                     val_target_transforms=val_target_transforms)
    else:
        raise ValueError(f"{dataset_name} not supported")

    # Init Method
    arch = train_config["arch"]
    patch_size = train_config["patch_size"]
    restart = train_config["restart"]
    val_iters = train_config["val_iters"]
    method = train_config["method"]
    spatial_res = input_size / patch_size
    decay_rate = train_config.get("decay_rate")
    assert spatial_res.is_integer()
    model = LinearFinetune(
        patch_size=patch_size,
        head_type=train_config.get("head_type"),
        arch=arch,
        num_classes=num_classes,
        lr=train_config["lr"],
        input_size=input_size,
        spatial_res=int(spatial_res),
        val_iters=val_iters,
        decay_rate=decay_rate if decay_rate is not None else 0.1,
        drop_at=train_config["drop_at"],
        ignore_index=ignore_index
    )

    # Optionally load weights
    if not restart:
        weights = get_backbone_weights(arch, method, patch_size=patch_size, ckpt_path=train_config.get("ckpt_path"))
        msg = model.load_state_dict(weights, strict=False)
        print(msg)

    # Init checkpoint callback storing top 3 heads
    checkpoint_dir = os.path.join(train_config["ckpt_dir"], _run.experiment_info["name"])
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='miou_val',
        filename='ckp-{epoch:02d}-{miou_val:.4f}',
        save_top_k=3,
        mode='max',
        verbose=True,
    )

    # Init trainer and start training head
    trainer = Trainer(
        num_sanity_val_steps=val_iters,
        logger=neptune_logger,
        max_epochs=train_config["max_epochs"],
        gpus=_config["gpus"],
        accelerator='ddp' if _config["gpus"] > 1 else None,
        fast_dev_run=train_config["fast_dev_run"],
        log_every_n_steps=50,
        benchmark=True,
        deterministic=False,
        resume_from_checkpoint=train_config["ckpt_path"] if restart else None,
        amp_backend='native',
        terminate_on_nan=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, datamodule=data_module)


class LinearFinetune(pl.LightningModule):

    def __init__(self, patch_size: int, num_classes: int, lr: float, input_size: int, spatial_res: int, val_iters: int,
                 drop_at: int, arch: str, head_type: str = None, decay_rate: float = 0.1, ignore_index: int = 255):
        super().__init__()
        self.save_hyperparameters()

        # Init Model
        if arch=='vit-small':
            self.model = vit_small(patch_size=patch_size)
        elif arch == 'vit-base':
            self.model = vit_base(patch_size=patch_size)
        elif arch=='resnet50':
            backbone = resnet.__dict__[arch](pretrained=False)
            self.model = ResnetDilated(backbone)
        if head_type == "fcn":
            self.finetune_head = FCNHead(
                in_channels=self.model.embed_dim,
                channels=512,
                num_convs=2,
                concat_input=True,
                dropout_ratio=0.1,
                num_classes=num_classes,
            )
        else:
            self.finetune_head = nn.Conv2d(self.model.embed_dim, num_classes, 1)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.miou_metric = PredsmIoU(num_classes, num_classes)
        self.num_classes = num_classes
        self.lr = lr
        self.val_iters = val_iters
        self.input_size = input_size
        self.spatial_res = spatial_res
        self.drop_at = drop_at
        self.arch = arch
        self.ignore_index = ignore_index
        self.decay_rate = decay_rate
        self.train_mask_size = 100
        self.val_mask_size = 100

    def on_after_backward(self):
        # Freeze all layers of backbone
        for param in self.model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.finetune_head.parameters(), weight_decay=0.0001,
                                    momentum=0.9, lr=self.lr)
        scheduler = StepLR(optimizer, gamma=self.decay_rate, step_size=self.drop_at)
        return [optimizer], [scheduler]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        imgs, masks = batch
        bs = imgs.size(0)
        res = imgs.size(3)
        assert res == self.input_size
        self.model.eval()

        with torch.no_grad():
            tokens = self.model.forward_backbone(imgs)
            if 'vit' in self.arch:
                tokens = tokens[:, 1:].reshape(bs, self.spatial_res, self.spatial_res, self.model.embed_dim).\
                    permute(0, 3, 1, 2)
            tokens = nn.functional.interpolate(tokens, size=(self.train_mask_size, self.train_mask_size),
                                               mode='bilinear')
        mask_preds = self.finetune_head(tokens)

        masks *= 255
        if self.train_mask_size != self.input_size:
            with torch.no_grad():
                masks = nn.functional.interpolate(masks, size=(self.train_mask_size, self.train_mask_size),
                                                  mode='nearest')

        loss = self.criterion(mask_preds, masks.long().squeeze())

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if batch_idx < self.val_iters:
            with torch.no_grad():
                imgs, masks = batch
                bs = imgs.size(0)
                tokens = self.model.forward_backbone(imgs)
                if 'vit' in self.arch:
                    tokens = tokens[:, 1:].reshape(bs, self.spatial_res, self.spatial_res, self.model.embed_dim).\
                        permute(0, 3, 1, 2)
                tokens = nn.functional.interpolate(tokens, size=(self.val_mask_size, self.val_mask_size),
                                                   mode='bilinear')
                mask_preds = self.finetune_head(tokens)

                # downsample masks and preds
                gt = masks * 255
                gt = nn.functional.interpolate(gt, size=(self.val_mask_size, self.val_mask_size), mode='nearest')
                valid = (gt != self.ignore_index) # mask to remove object boundary class
                mask_preds = torch.argmax(mask_preds, dim=1).unsqueeze(1)

                # update metric
                self.miou_metric.update(gt[valid], mask_preds[valid])

    def validation_epoch_end(self, outputs: List[Any]):
        miou = self.miou_metric.compute(True, many_to_one=False, linear_probe=True)[0]
        self.miou_metric.reset()
        print(miou)
        self.log('miou_val', round(miou, 6))


if __name__ == "__main__":
    entry()
