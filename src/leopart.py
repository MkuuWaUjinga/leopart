# Some methods adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import distributed as dist
from torch import nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torchvision.ops import roi_align
from typing import Callable, Optional, List, Any, Iterator, Tuple, Dict

from experiments.utils import PredsmIoUKmeans, process_attentions, cosine_scheduler
from src.vit import vit_small, vit_base, vit_large


class Leopart(pl.LightningModule):

    def __init__(self, gpus: int, num_samples: int, batch_size: int, max_epochs: int, lr_heads: float,
                 lr_backbone: float, final_lr: float, weight_decay_end: float, weight_decay: float, epsilon: float,
                 temperature: float, projection_hidden_dim: int, projection_feat_dim: int,
                 n_layers_projection_head: int, nmb_prototypes: int, sinkhorn_iterations: int,
                 crops_for_assign: List[int], nmb_crops: List[int], num_classes: int, val_iters: int,
                 num_clusters_kmeans: List[int], use_teacher: bool = True, loss_mask: str = 'all',
                 queue_length: int = 0, momentum_teacher: float = 0.9995, momentum_teacher_end: float = 1.,
                 exclude_norm_bias: bool = True, optimizer: str = 'adam', num_nodes: int = 1,
                 patch_size: int = 16, roi_align_kernel_size: int = 7, val_downsample_masks: bool = True,
                 arch: str = 'vit-small'):
        """
        Initializes the Leopart for training. We use pytorch lightning as framework.
        :param gpus: number of gpus used per node
        :param num_samples: number of samples in train data
        :param batch_size: batch size per GPU
        :param max_epochs: the number of epochs
        :param lr_heads: learning rate for clustering projection head
        :param lr_backbone: learning rate for ViT backbone
        :param final_lr: final learning rate for cosine learning rate schedule
        :param weight_decay_end: final weight decay for cosine weight decay schedule
        :param weight_decay: weight decay for optimizer
        :param epsilon: regularization parameter for sinkhorn-knopp clustering
        :param temperature: temperature applied before softmaxing the cluster assignment predictions
        :param projection_hidden_dim: embedding dimensionality of hidden layers in projection head
        :param projection_feat_dim: embedding dimensionality of output layer in projection head
        :param n_layers_projection_head: number of layers for projection head
        :param nmb_prototypes: number of prototypes
        :param sinkhorn_iterations: number of iterations in Sinkhorn-Knopp algorithm
        :param crops_for_assign: list of crop ids for computing optimal cluster assignment
        :param nmb_crops: number of global and local crops to be used during training
        :param num_classes: number of gt classes of validation data
        :param val_iters: number of validation iterations per epoch.
        :param num_clusters_kmeans: list of clustering granularities to be used to evaluate learnt feature space
        :param use_teacher: flag to indicate whether a teacher network should be used for computing the optimal cluster
        assignments
        :param loss_mask: indicates masking mode for computing cross entropy. Choose from 'fg', 'all' and 'bg'.
        :param queue_length: length of queue. used for stabilizing sinkhorn-knopp for small batch sizes.
        :param momentum_teacher: start value of momentum for teacher network
        :param momentum_teacher_end: end value of momentum for teacher network
        :param exclude_norm_bias: flag to exclude norm and bias from weight decay
        :param optimizer: type of optimizer to use. Currently only supports adamw
        :param num_nodes: number of nodes to train on
        :param patch_size: patch size used for vision transformer
        :param roi_align_kernel_size: kernel size to be used for aligning the predicted and optimal cluster assignments
        each crop's bounding box.
        :param val_downsample_masks: flag to downsample masks for evaluation. If set mIoU is evaluated on 100x100 masks.
        :param arch: architecture of model to be fine-tuned. Currently supports vit-small, vit-base and vit-large.
        """
        super().__init__()
        self.save_hyperparameters()
        self.roi_align_kernel_size = roi_align_kernel_size
        self.lr_heads = lr_heads
        self.patch_size = patch_size
        self.projection_hidden_dim = projection_hidden_dim
        self.n_layers_projection_head = n_layers_projection_head
        self.val_downsample_masks = val_downsample_masks
        self.arch = arch
        self.gpus = gpus
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.projection_feat_dim = projection_feat_dim
        self.nmb_prototypes = nmb_prototypes
        self.sinkhorn_iterations = sinkhorn_iterations
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops
        self.optim = optimizer
        self.exclude_norm_bias = exclude_norm_bias
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.temperature = temperature
        self.final_lr = final_lr
        self.lr_backbone = lr_backbone
        self.max_epochs = max_epochs
        self.val_iters = val_iters
        self.num_clusters_kmeans = num_clusters_kmeans
        self.num_classes = num_classes
        self.loss_mask = loss_mask
        self.use_teacher = use_teacher

        # queue params
        self.queue_length = queue_length
        self.queue_path = 'queue'
        self.queue = None

        # init sinkhorn method
        if self.gpus * self.num_nodes > 1:
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

        # init model
        if self.use_teacher:
            self.teacher = None
        self.model = self.init_model()  # inits teacher as well
        self.softmax = nn.Softmax(dim=1)

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # init wd and momentum schedule
        self.wd_schedule = cosine_scheduler(self.weight_decay, weight_decay_end,
                                            self.max_epochs, self.train_iters_per_epoch)
        if self.use_teacher:
            self.momentum_schedule = cosine_scheduler(momentum_teacher, momentum_teacher_end,
                                                      self.max_epochs, self.train_iters_per_epoch)

        # init metric modules
        self.preds_miou_layer4 = PredsmIoUKmeans(num_clusters_kmeans, num_classes)

    def init_model(self):
        # Initialize model and optionally the teacher
        if self.arch == 'vit-small':
            model_func = vit_small
        elif self.arch == 'vit-base':
            model_func = vit_base
        elif self.arch == 'vit-large':
            model_func = vit_large
        else:
            raise ValueError(f"{self.arch} is not supported")
        if self.use_teacher:
            self.teacher = model_func(patch_size=self.patch_size,
                                      output_dim=self.projection_feat_dim,
                                      hidden_dim=self.projection_hidden_dim,
                                      nmb_prototypes=self.nmb_prototypes,
                                      n_layers_projection_head=self.n_layers_projection_head)
        return model_func(patch_size=self.patch_size,
                         drop_path_rate=0.1,
                         output_dim=self.projection_feat_dim,
                         hidden_dim=self.projection_hidden_dim,
                         nmb_prototypes=self.nmb_prototypes,
                         n_layers_projection_head=self.n_layers_projection_head)

    def on_train_epoch_start(self):
        # Init queue if queue is None
        if self.queue_length > 0 and self.queue is None:
            self.queue = torch.zeros(
                len(self.crops_for_assign),
                self.queue_length // self.gpus,  # change to nodes * gpus once multi-node
                self.projection_feat_dim,
                )
            if self.gpus > 0:
                self.queue = self.queue.cuda()

        self.use_the_queue = False

    def configure_optimizers(self):
        # Separate head params from backbone params
        head_params_named = []
        backbone_params_named = []
        for name, param in self.model.named_parameters():
            if name.startswith("projection_head")  or name.startswith("prototypes"):
                head_params_named.append((name, param))
            else:
                backbone_params_named.append((name, param))

        # Prepare param groups. Exclude norm and bias from weight decay if flag set.
        if self.exclude_norm_bias:
            backbone_params = self.exclude_from_wt_decay(backbone_params_named,
                                                         weight_decay=self.weight_decay,
                                                         lr=self.lr_backbone)
            head_params = self.exclude_from_wt_decay(head_params_named,
                                                     weight_decay=self.weight_decay,
                                                     lr=self.lr_heads)
            params = backbone_params + head_params
        else:
            backbone_params = [param for _, param in backbone_params_named]
            head_params = [param for _, param in head_params_named]
            params = [{'params': backbone_params, 'lr': self.lr_backbone},
                      {'params': head_params, 'lr': self.lr_heads}]

        assert len(list(self.model.parameters())) == len(backbone_params_named) + len(head_params_named)

        # Init optimizer and lr schedule
        if self.optim == 'adamw':
            optimizer = torch.optim.AdamW(params, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optim} not supported')
        scheduler = CosineAnnealingLR(optimizer, T_max=self.train_iters_per_epoch * self.max_epochs,
                                      eta_min=self.final_lr)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @staticmethod
    def exclude_from_wt_decay(named_params: Iterator[Tuple[str, nn.Parameter]], weight_decay: float, lr: float):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            # do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                excluded_params.append(param)
            else:
                params.append(param)
        return [{'params': params, 'weight_decay': weight_decay, 'lr': lr},
                {'params': excluded_params, 'weight_decay': 0., 'lr': lr}]

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer: Optimizer = None,
                       optimizer_idx: int = None, optimizer_closure: Optional[Callable] = None,
                       on_tpu: bool = None, using_native_amp: bool = None, using_lbfgs: bool = None,):
        # Step weight decay schedule
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0 or i == 2:
                param_group["weight_decay"] = self.wd_schedule[self.trainer.global_step]

        if not isinstance(optimizer, LightningOptimizer):
            # wraps into LightingOptimizer only for running step
            optimizer = LightningOptimizer._to_lightning_optimizer(optimizer, self.trainer, optimizer_idx)
        optimizer.step(closure=optimizer_closure)

    def shared_step(self, batch: Tuple[List[torch.Tensor], Dict]) -> float:
        inputs, bboxes = batch

        # 1. normalize the student and optionally the teacher prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)
            if self.use_teacher:
                w = self.teacher.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.teacher.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        last_self_attention = True
        if self.loss_mask == "all":
            last_self_attention = False
        bs = inputs[0].size(0)
        if self.use_teacher:
            res_forward_teacher = self.teacher(inputs[:2], last_self_attention=last_self_attention)
        else:
            res_forward_teacher = self.model(inputs[:2], last_self_attention=last_self_attention)
        res_forward_student = self.model(inputs)
        if self.loss_mask == "all":
            teacher_gc_spatial_emb, teacher_gc_spatial_output = res_forward_teacher
        else:
            teacher_gc_spatial_emb, teacher_gc_spatial_output, teacher_gc_attn = res_forward_teacher
        _, student_spatial_output = res_forward_student

        # 3. calculate gc and lc resolutions. Split student output in gc and lc embeddings
        gc_res_w = inputs[0].size(2) / self.patch_size
        gc_res_h = inputs[0].size(3) / self.patch_size
        assert gc_res_w.is_integer() and gc_res_w.is_integer(), "Image dims need to be divisible by patch size"
        assert gc_res_w == gc_res_h, f"Only supporting square images not {inputs[0].size(2)}x{inputs[0].size(3)}"
        gc_spatial_res = int(gc_res_w)
        gc_student_spatial_output, lc_student_spatial_output = \
            student_spatial_output[:bs * self.nmb_crops[0] * gc_spatial_res ** 2], \
            student_spatial_output[bs * self.nmb_crops[0] * gc_spatial_res ** 2:]

        attn_hard = None
        if self.loss_mask != "all":
            # Merge attention heads and threshold attentions
            attn_smooth = sum(teacher_gc_attn[:, i] * 1 / teacher_gc_attn.size(1) for i
                              in range(teacher_gc_attn.size(1)))
            attn_smooth = attn_smooth.reshape(bs * self.nmb_crops[0], 1, gc_spatial_res, gc_spatial_res)
            attn_hard = process_attentions(attn_smooth, gc_spatial_res, threshold=0.6, blur_sigma=0.6)
            if self.loss_mask == 'bg':
                attn_hard = torch.abs(attn_hard - 1) # invert 1-0 mask if we want to train on bg tokens

        # Calculate loss
        spatial_loss = self.spatial_loss(teacher_gc_spatial_output, gc_student_spatial_output,
                                         lc_student_spatial_output, teacher_gc_spatial_emb, bboxes, bs, gc_spatial_res,
                                         attn_hard=attn_hard)
        return spatial_loss

    def spatial_loss(self, gc_teacher_output: torch.Tensor, gc_student_output: torch.Tensor,
                     lc_student_output: torch.Tensor, gc_teacher_emb: torch.Tensor, bboxes: Dict, bs: int,
                     gc_spatial_res: int, attn_hard: torch.Tensor = None) -> float:
        # 3. compute lc spatial res
        lc_spatial_res = np.sqrt(lc_student_output.size(0) / (self.nmb_crops[-1] * bs))
        assert lc_spatial_res.is_integer(), "spatial crops should have same x and y dim"
        lc_spatial_res = int(lc_spatial_res)

        # 4. swav loss computation
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                # Select spatial cluster preds for global crop with crop_id
                out = gc_teacher_output[bs * gc_spatial_res ** 2 * crop_id:bs * gc_spatial_res ** 2 * (crop_id + 1)]
                num_spatial_vectors_for_pred = out.size(0)

                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(self.queue[i], torch.flatten(self.model.prototypes.weight, 1).t()),
                                         out))

                    # Add spatial embeddings to queue
                    # Use attention to determine number of foreground embeddings to be stored
                    emb_gc = gc_teacher_emb[bs * gc_spatial_res ** 2 * crop_id:bs * gc_spatial_res ** 2 * (crop_id + 1)]
                    if attn_hard is not None:
                        # only add fg embeddings to queue
                        flat_mask = attn_hard.permute(0, 2, 3, 1).flatten().bool()
                        gc_fg_mask = flat_mask[bs * gc_spatial_res**2 * crop_id: bs * gc_spatial_res**2 * (crop_id+1)]
                        emb_gc = emb_gc[gc_fg_mask]
                    num_vectors_to_store = min(bs * 10, self.queue_length // self.gpus)
                    idx = torch.randperm(emb_gc.size(0))[:num_vectors_to_store]
                    self.queue[i, num_vectors_to_store:] = self.queue[i, :-num_vectors_to_store].clone()
                    self.queue[i, :num_vectors_to_store] = emb_gc[idx]

                # 5. get assignments
                q = torch.exp(out / self.epsilon).t()
                q = self.get_assignments(q, self.sinkhorn_iterations)[-num_spatial_vectors_for_pred:]

            # 6. Roi align cluster assignments
            q_reshaped = q.reshape(bs, gc_spatial_res, gc_spatial_res, -1).permute(0, 3, 1, 2)
            downsampled_current_crop_boxes = torch.unbind(bboxes["gc"][:, crop_id] / self.patch_size)
            aligned_soft_clusters = roi_align(q_reshaped, downsampled_current_crop_boxes,
                                              self.roi_align_kernel_size, aligned=True)  # (bs * num_crops, 7, 7, 2048)
            if attn_hard is not None:
                # 6.5 Roi align mask
                gc_hard_mask = attn_hard[bs * crop_id: bs * (crop_id+1)]  # select attn for crop_id
                aligned_mask = roi_align(gc_hard_mask, downsampled_current_crop_boxes, self.roi_align_kernel_size,
                                         aligned=True)
                thresholded_mask = (aligned_mask >= 1.0)  # Make mask 1-0

            # 7 .cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                if v in self.crops_for_assign:
                    # Code prediction from other global crop
                    out = gc_student_output[bs * gc_spatial_res ** 2 * v:bs * gc_spatial_res ** 2 * (v + 1)]
                    spatial_res = gc_spatial_res
                else:
                    # Code prediction from local crop
                    lc_index = v - self.nmb_crops[0]
                    out = lc_student_output[bs * lc_spatial_res**2 * lc_index:bs * lc_spatial_res**2 * (lc_index + 1)]
                    spatial_res = lc_spatial_res
                # Roi align cluster predictions
                aligned_out = roi_align(out.reshape(bs, spatial_res, spatial_res, -1).permute(0, 3, 1, 2),
                                        torch.unbind(bboxes["all"][:, v, crop_id].unsqueeze(1) / self.patch_size),
                                        self.roi_align_kernel_size,
                                        aligned=True)
                aligned_p = self.softmax(aligned_out / self.temperature)
                aligned_q = aligned_soft_clusters[v::np.sum(self.nmb_crops)]
                # Mask cross entropy if attn mask was passed
                if attn_hard is not None:
                    mask = thresholded_mask[v::np.sum(self.nmb_crops)].squeeze().float()
                    if torch.sum(mask).item()!=0:
                        subloss -= torch.sum(torch.sum(aligned_q * torch.log(aligned_p), dim=1) * mask) / torch.sum(mask)
                else:
                    # otherwise apply loss on all spatial tokens.
                    subloss -= torch.mean(torch.sum(aligned_q * torch.log(aligned_p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        return loss

    def training_step(self, batch: Tuple[List[torch.Tensor], Dict], batch_idx: int) -> float:
        if isinstance(batch[1], dict):
            loss = self.shared_step(batch)
        else:
            raise ValueError("No rrc boxes passed")

        if self.use_teacher:
            # EMA update for the teacher using the momentum_schedule
            with torch.no_grad():
                m = self.momentum_schedule[self.trainer.global_step]  # momentum parameter
                for param_q, param_k in zip(self.model.parameters(), self.teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        self.log('lr_backbone', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False)
        self.log('lr_heads', self.optimizers().param_groups[2]['lr'], on_step=True, on_epoch=False)
        self.log('weight_decay', self.optimizers().param_groups[0]['weight_decay'], on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def sinkhorn(self, Q: torch.Tensor, nmb_iters: int) -> torch.Tensor:
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q: torch.Tensor, nmb_iters: int) -> torch.Tensor:
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # Validate for self.val_iters. Constrained to only parts of the validation set as mIoU calculation
        # would otherwise take too long.
        if batch_idx < self.val_iters:
            with torch.no_grad():
                imgs, mask = batch

                # Normalize prototype layer
                w = self.model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.prototypes.weight.copy_(w)

                # Process gt seg masks
                bs = imgs.size(0)
                assert torch.max(mask).item() <= 1 and torch.min(mask).item() >= 0
                gt = mask * 255
                if self.val_downsample_masks:
                    size_masks = 100
                    gt = nn.functional.interpolate(gt, size=(size_masks, size_masks), mode='nearest')
                valid = (gt != 255)  # mask to remove object boundary class

                # Get backbone embeddings
                backbone_embeddings = self.model.forward_backbone(imgs)[:, 1:]

                # store embeddings, valid masks and gt for clustering after validation end
                res_w = int(np.sqrt(backbone_embeddings.size(1)))
                backbone_embeddings = backbone_embeddings.permute(0, 2, 1).reshape(bs, self.model.embed_dim,
                                                                                   res_w, res_w)
                self.preds_miou_layer4.update(valid, backbone_embeddings, gt)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # Trigger computations for rank 0 process
        res_kmeans = self.preds_miou_layer4.compute(self.trainer.is_global_zero)
        self.preds_miou_layer4.reset()
        if res_kmeans is not None:  # res_kmeans is none for all processes with rank != 0
            for k, name, res_k in res_kmeans:
                miou_kmeans, tp, fp, fn, _, matched_bg = res_k
                self.print(miou_kmeans)
                self.logger.experiment.log_metric(f'K={name}_miou_layer4', round(miou_kmeans, 8))
                # Log precision and recall values for each class
                for i, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn)):
                    class_name = self.trainer.datamodule.class_id_to_name(i)
                    self.logger.experiment.log_metric(f'K={name}_{class_name}_precision',
                                                      round(tp_class / max(tp_class + fp_class, 1e-8), 8))
                    self.logger.experiment.log_metric(f'K={name}_{class_name}_recall',
                                                      round(tp_class / max(tp_class + fn_class, 1e-8), 8))
                if k > self.num_classes:
                    # Log percentage of clusters assigned to background class
                    self.logger.experiment.log_metric(f'K={name}-percentage-bg-cluster', round(matched_bg, 8))
