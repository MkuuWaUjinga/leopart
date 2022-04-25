import numpy as np
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from data.VOCdevkit.vocdata import VOCDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, GaussianBlur
from torchvision.transforms.functional import InterpolationMode
from skimage.measure import label


class EvaluateAttnMaps(pl.callbacks.Callback):

    def __init__(self,
                 voc_root: str,
                 train_input_height: int,
                 attn_batch_size: int,
                 num_workers: int,
                 threshold: float = 0.6):
        # Setup transforms and dataloader pvoc
        image_transforms = Compose([Resize((train_input_height, train_input_height)),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        target_transforms = Compose([Resize((train_input_height, train_input_height),
                                            interpolation=InterpolationMode.NEAREST),
                                     ToTensor()])
        self.dataset = VOCDataset(root=os.path.join(voc_root, "VOCSegmentation"), image_set="val",
                                  transform=image_transforms, target_transform=target_transforms)
        self.loader = DataLoader(self.dataset, batch_size=attn_batch_size, shuffle=False, num_workers=num_workers,
                                 drop_last=True, pin_memory=True)
        self.threshold = threshold

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Evaluate attention maps.
        if pl_module.global_rank == 0 and pl_module.local_rank == 0:
            print("\n" + "#" * 20 + "Evaluating attention maps on VOC2012 with threshold: " +
                  str(self.threshold) + "#" * 20)
            jacs_merged_attn = 0
            jacs_all_heads = 0
            # If teacher is present use teacher attention as it is also used during training
            if hasattr(pl_module, 'teacher'):
                patch_size = pl_module.teacher.patch_size
                model = pl_module.teacher
            else:
                patch_size = pl_module.model.patch_size
                model = pl_module.model

            model.eval()
            for i, (imgs, maps) in enumerate(self.loader):
                w_featmap = imgs.shape[-2] // patch_size
                h_featmap = imgs.shape[-1] // patch_size

                with torch.no_grad():
                    attentions = model.get_last_selfattention(imgs.to(pl_module.device))
                bs = attentions.shape[0]
                attentions = attentions[..., 0, 1:]
                # Evaluate two different protocols: merged attention and best head
                jacs_merged_attn += self.evaluate_merged_attentions(attentions, bs, w_featmap, h_featmap, patch_size,
                                                                    maps)
                jacs_all_heads += self.evaluate_best_head(attentions, bs, w_featmap, h_featmap, patch_size, maps)

            jacs_merged_attn /= len(self.dataset)
            jacs_all_heads /= len(self.dataset)
            print(f"Merged Jaccard on VOC12: {jacs_merged_attn.item()}")
            print(f"All heads Jaccard on VOC12: {jacs_all_heads.item()}")
            pl_module.logger.experiment.log_metric('attn_jacs_voc', jacs_merged_attn.item())
            pl_module.logger.experiment.log_metric('all_heads_jacs_voc', jacs_all_heads.item())

    def evaluate_best_head(self, attentions: torch.Tensor, bs: int, w_featmap: int, h_featmap: int, patch_size: int,
                           maps: torch.Tensor) -> torch.Tensor:
        jacs = 0
        nh = attentions.shape[1] # number of heads

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=-1, keepdim=True)
        cumval = torch.cumsum(val, dim=-1)
        th_attn = cumval > (1 - self.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[:, head] = torch.gather(th_attn[:, head], dim=1, index=idx2[:, head])
        th_attn = th_attn.reshape(bs, nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn, scale_factor=patch_size, mode="nearest").cpu().numpy()

        # Calculate IoU for each image
        for k, map in enumerate(maps):
            jac = 0
            objects = np.unique(map)
            objects = np.delete(objects, [0, -1])
            for o in objects:
                masko = map == o
                intersection = masko * th_attn[k]
                intersection = torch.sum(torch.sum(intersection, dim=-1), dim=-1)
                union = (masko + th_attn[k]) > 0
                union = torch.sum(torch.sum(union, dim=-1), dim=-1)
                jaco = intersection / union
                jac += max(jaco)
            if len(objects) != 0:
                jac /= len(objects)
            jacs += jac
        return jacs

    def evaluate_merged_attentions(self, attentions: torch.Tensor, bs: int, w_featmap: int, h_featmap: int,
                                   patch_size: int, maps: torch.Tensor) -> torch.Tensor:
        jacs = 0
        # Average attentions
        attentions = sum(attentions[:, i] * 1 / attentions.size(1) for i in range(attentions.size(1)))
        nh = 1  # number of heads is one as we merged all heads

        # Gaussian blurring
        attentions = GaussianBlur(7, sigma=(.6))(attentions.reshape(bs * nh, 1, w_featmap, h_featmap))\
            .reshape(bs, nh, -1)

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=-1, keepdim=True)
        cumval = torch.cumsum(val, dim=-1)
        th_attn = cumval > (1 - self.threshold)
        idx2 = torch.argsort(idx)
        th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
        th_attn = th_attn.reshape(bs, nh, w_featmap, h_featmap).float()

        # remove components that are less then 3 pixels
        for j, th_att in enumerate(th_attn):
            labelled = label(th_att.cpu().numpy())
            for k in range(1, np.max(labelled) + 1):
                mask = labelled == k
                if np.sum(mask) <= 2:
                    th_attn[j, 0][mask] = 0

        # interpolate
        th_attn = nn.functional.interpolate(th_attn, scale_factor=patch_size, mode="nearest").cpu().numpy()

        # Calculate IoU for each image
        for k, map in enumerate(maps):
            gt_fg_mask = (map != 0.).float()
            intersection = gt_fg_mask * th_attn[k]
            intersection = torch.sum(torch.sum(intersection, dim=-1), dim=-1)
            union = (gt_fg_mask + th_attn[k]) > 0
            union = torch.sum(torch.sum(union, dim=-1), dim=-1)
            jacs += intersection / union
        return jacs
