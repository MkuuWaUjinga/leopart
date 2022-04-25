import click
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models.resnet as resnet

from torchvision.transforms.functional import InterpolationMode

from data.VOCdevkit.vocdata import VOCDataModule
from data.coco.coco_data_module import CocoDataModule
from experiments.utils import PredsmIoU, get_backbone_weights
from experiments.linear_probing.fcn.fcn_head import FCNHead
from src.vit import vit_small, vit_base
from src.resnet import ResnetDilated


@click.command()
@click.option("--ckpt_path_backbone", type=str, required=True)
@click.option("--ckpt_path_head", type=str, required=True)
@click.option("--patch_size", type=int, default=16)
@click.option("--arch", type=str, default="vit-small")
@click.option("--num_classes", type=int, default=21)
@click.option("--head_type", type=str, default="linear")
@click.option("--dataset_name", type=str, default="voc")
@click.option("--data_dir", type=str, required=True)
@click.option("--batch_size", type=int, default=15)
@click.option("--input_size", type=int, default=448)
@click.option("--mask_eval_size", type=int, default=448)
def eval_bulk(ckpt_path_backbone: str, ckpt_path_head: str, patch_size: int, arch: str, num_classes: int,
              head_type: str, dataset_name: str, data_dir: str, batch_size: int, input_size: int, mask_eval_size: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    miou_metric = PredsmIoU(num_classes, num_classes)

    # Init model
    if arch == "vit-small":
        model = vit_small(patch_size=patch_size)
    elif arch == "vit-base":
        model = vit_base(patch_size=patch_size)
    elif arch == "resnet":
        backbone = resnet.__dict__[arch](pretrained=False)
        model = ResnetDilated(backbone)
    else:
        raise ValueError(f"{arch} not supported as model")
    if head_type == 'fcn':
        finetune_head = FCNHead(
            in_channels=model.embed_dim,
            channels=512,
            num_convs=2,
            concat_input=True,
            dropout_ratio=0,
            num_classes=num_classes,
        )
        print(finetune_head)
    else:
        finetune_head = nn.Conv2d(model.embed_dim, num_classes, 1)

    # load backbone
    weights = get_backbone_weights(arch, "ours", patch_size, weight_prefix="", ckpt_path=ckpt_path_backbone)
    msg = model.load_state_dict(weights, strict=False)
    print(msg)
    model.eval()
    model.to(device)

    # load linear head
    state_dict = torch.load(ckpt_path_head)
    weights = {k.replace("finetune_head.", ""): v for k, v in state_dict.items()}
    msg = finetune_head.load_state_dict(weights, strict=False)
    print(msg)
    assert len(msg[0]) == 0
    finetune_head.eval()
    finetune_head.to(device)

    # Init transforms and data
    val_image_transforms = T.Compose([T.Resize((input_size, input_size)),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])

    if dataset_name == "voc":
        data_module = VOCDataModule(batch_size=batch_size,
                                    num_workers=5,
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_dir,
                                    train_image_transform=val_image_transforms,
                                    drop_last=False,
                                    val_image_transform=val_image_transforms,
                                    val_target_transform=val_target_transforms)
    elif "coco" in dataset_name:
        assert len(dataset_name.split("-")) == 2
        mask_type = dataset_name.split("-")[-1]
        file_list = os.listdir(os.path.join(data_dir, "coco", "images", "train2017"))
        file_list_val = os.listdir(os.path.join(data_dir, "coco", "images", "val2017"))
        data_module = CocoDataModule(batch_size=batch_size,
                                     num_workers=5,
                                     file_list=file_list,
                                     data_dir=data_dir,
                                     file_list_val=file_list_val,
                                     mask_type=mask_type,
                                     train_transforms=val_image_transforms,
                                     val_transforms=val_image_transforms,
                                     val_target_transforms=val_target_transforms)
    else:
        raise ValueError(f"{dataset_name} not supported as dataset")

    data_module.setup()
    spatial_res = input_size / patch_size # for resnets we use dilated convs in the last bottlenceck to match the vits res
    assert spatial_res.is_integer()
    spatial_res = int(spatial_res)

    # Get head predictions
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(data_module.val_dataloader()):
            bs = imgs.size(0)
            tokens = model.forward_backbone(imgs.to(device))
            if "vit" in arch:
               tokens = tokens[:, 1:].reshape(bs, spatial_res, spatial_res, model.embed_dim).permute(0, 3, 1, 2)
            tokens = nn.functional.interpolate(tokens, size=(mask_eval_size, mask_eval_size), mode='bilinear')
            mask_preds = finetune_head(tokens)
            mask_preds = torch.argmax(mask_preds, dim=1).unsqueeze(1)

            # downsample masks and preds
            gt = masks * 255
            gt = nn.functional.interpolate(gt, size=(mask_eval_size, mask_eval_size), mode='nearest')
            valid = (gt != 255) # remove object boundary class

            # update metric
            miou_metric.update(gt[valid].cpu(), mask_preds[valid].cpu())

            if (i + 1) % 50 == 0:
                print(f"{(i+1) * bs} done")

    # Calculate mIoU
    miou = miou_metric.compute(True, linear_probe=True)[0]
    miou_metric.reset()
    print(miou)


if __name__ == "__main__":
    eval_bulk()
