import faiss
import timm
import numpy as np
import os
import time
import torch
import torch.nn as nn

from collections import defaultdict
from functools import partial
from skimage.measure import label
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchvision.transforms import GaussianBlur
from torchvision import models
from typing import Optional, List, Tuple, Dict

from data.VOCdevkit.vocdata import VOCDataModule


class PredsmIoU(Metric):
    """
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """
    def __init__(self,
                 num_pred_classes: int,
                 num_gt_classes: int):
        """
        :param num_pred_classes: The number of predicted classes.
        :param num_gt_classes: The number of gt classes.
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_pred_classes = num_pred_classes
        self.num_gt_classes = num_gt_classes
        self.add_state("gt", [])
        self.add_state("pred", [])
        self.n_jobs = -1

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        self.gt.append(gt)
        self.pred.append(pred)

    def compute(self, is_global_zero: bool, many_to_one: bool = False,
                precision_based: bool = False, linear_probe : bool = False) -> Tuple[float, List[np.int64],
                                                                                     List[np.int64], List[np.int64],
                                                                                     List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param is_global_zero: Flag indicating whether process is rank zero. Computation of metric is only triggered
        if True.
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        if is_global_zero:
            pred = torch.cat(self.pred).cpu().numpy().astype(int)
            gt = torch.cat(self.gt).cpu().numpy().astype(int)
            assert len(np.unique(pred)) <= self.num_pred_classes
            assert np.max(pred) <= self.num_pred_classes
            return self.compute_miou(gt, pred, self.num_pred_classes, self.num_gt_classes, many_to_one=many_to_one,
                                     precision_based=precision_based, linear_probe=linear_probe)

    def compute_miou(self, gt: np.ndarray, pred: np.ndarray, num_pred: int, num_gt:int,
                     many_to_one=False, precision_based=False, linear_probe=False) -> Tuple[float, List[np.int64], List[np.int64], List[np.int64],
                                                  List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        assert pred.shape == gt.shape
        print(f"seg map preds have size {gt.shape}")
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        jac = [0] * num_gt

        if linear_probe:
            reordered_preds = pred
            matched_bg_clusters = {}
        else:
            if many_to_one:
                match = self._original_match(num_pred, num_gt, pred, gt, precision_based=precision_based)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, matched_preds in match.items():
                    for pred_i in matched_preds:
                        reordered_preds[pred == int(pred_i)] = int(target_i)
                matched_bg_clusters = len(match[0]) / num_pred
            else:
                match = self._hungarian_match(num_pred, num_gt, pred, gt)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, pred_i in zip(*match):
                    reordered_preds[pred == int(pred_i)] = int(target_i)
                # merge all unmatched predictions to background
                for unmatched_pred in np.delete(np.arange(num_pred), np.array(match[1])):
                    reordered_preds[pred == int(unmatched_pred)] = 0
                matched_bg_clusters = 1/num_gt

        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            tmp_all_gt = (gt == i_part)
            tmp_pred = (reordered_preds == i_part)
            tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        print("IoUs computed")
        return np.mean(jac), tp, fp, fn, reordered_preds.astype(int).tolist(), matched_bg_clusters

    @staticmethod
    def get_score(flat_preds: np.ndarray, flat_targets: np.ndarray, c1: int, c2: int, precision_based: bool = False) \
            -> float:
        """
        Calculates IoU given gt class c1 and prediction class c2.
        :param flat_preds: flattened predictions
        :param flat_targets: flattened gt
        :param c1: ground truth class to match
        :param c2: predicted class to match
        :param precision_based: flag to calculate precision instead of IoU.
        :return: The score if gt-c1 was matched to predicted c2.
        """
        tmp_all_gt = (flat_targets == c1)
        tmp_pred = (flat_preds == c2)
        tp = np.sum(tmp_all_gt & tmp_pred)
        fp = np.sum(~tmp_all_gt & tmp_pred)
        if not precision_based:
            fn = np.sum(tmp_all_gt & ~tmp_pred)
            jac = float(tp) / max(float(tp + fp + fn), 1e-8)
            return jac
        else:
            prec = float(tp) / max(float(tp + fp), 1e-8)
            return prec

    def compute_score_matrix(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray,
                             precision_based: bool = False) -> np.ndarray:
        """
        Compute score matrix. Each element i, j of matrix is the score if i was matched j. Computation is parallelized
        over self.n_jobs.
        :param num_pred: number of predicted classes
        :param num_gt: number of ground-truth classes
        :param pred: flattened predictions
        :param gt: flattened gt
        :param precision_based: flag to calculate precision instead of IoU.
        :return: num_pred x num_gt matrix with A[i, j] being the score if ground-truth class i was matched to
        predicted class j.
        """
        print("Parallelizing iou computation")
        start = time.time()
        score_mat = Parallel(n_jobs=self.n_jobs)(delayed(self.get_score)(pred, gt, c1, c2, precision_based=precision_based)
                                                 for c2 in range(num_pred) for c1 in range(num_gt))
        print(f"took {time.time() - start} seconds")
        score_mat = np.array(score_mat)
        return score_mat.reshape((num_pred, num_gt)).T

    def _hungarian_match(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray,
                                                                                                      np.ndarray]:
        # do hungarian matching. If num_pred > num_gt match will be partial only.
        iou_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt)
        match = linear_sum_assignment(1 - iou_mat)
        print("Matched clusters to gt classes:")
        print(match)
        return match

    def _original_match(self, num_pred, num_gt, pred, gt, precision_based=False) -> Dict[int, list]:
        score_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt, precision_based=precision_based)
        preds_to_gts = {}
        preds_to_gt_scores = {}
        # Greedily match predicted class to ground-truth class by best score.
        for pred_c in range(num_pred):
            for gt_c in range(num_gt):
                score = score_mat[gt_c, pred_c]
                if (pred_c not in preds_to_gts) or (score > preds_to_gt_scores[pred_c]):
                    preds_to_gts[pred_c] = gt_c
                    preds_to_gt_scores[pred_c] = score
        gt_to_matches = defaultdict(list)
        for k,v in preds_to_gts.items():
            gt_to_matches[v].append(k)
        print("matched clusters to gt classes:")
        return gt_to_matches


class PredsmIoUKmeans(PredsmIoU):
    """
    Used to track k-means cluster correspondence to ground-truth categories during fine-tuning.
    """
    def __init__(self,
                 clustering_granularities: List[int],
                 num_gt_classes: int,
                 pca_dim : int = 50):
        """
        :param clustering_granularities: list of clustering granularities for embeddings
        :param num_gt_classes: number of ground-truth classes
        :param pca_dim: target dimensionality of PCA
        """
        super(PredsmIoU, self).__init__(compute_on_step=False, dist_sync_on_step=False) # Init Metric super class
        self.pca_dim = pca_dim
        self.num_pred_classes = clustering_granularities
        self.num_gt_classes = num_gt_classes
        self.add_state("masks", [])
        self.add_state("embeddings", [])
        self.add_state("gt", [])
        self.n_jobs = -1 # num_jobs = num_cores
        self.num_train_pca = 4000000 # take num_train_pca many vectors at max for training pca

    def update(self, masks: torch.Tensor, embeddings: torch.Tensor, gt: torch.Tensor) -> None:
        self.masks.append(masks)
        self.embeddings.append(embeddings)
        self.gt.append(gt)

    def compute(self, is_global_zero:bool) -> List[any]:
        if is_global_zero:
            # interpolate embeddings to match ground-truth masks spatially
            embeddings = torch.cat([e.cpu() for e in self.embeddings], dim=0) # move everything to cpu before catting
            valid_masks = torch.cat(self.masks, dim=0).cpu().numpy()
            res_w = valid_masks.shape[2]
            embeddings = nn.functional.interpolate(embeddings, size=(res_w, res_w), mode='bilinear')
            embeddings = embeddings.permute(0, 2, 3, 1).reshape(valid_masks.shape[0] * res_w**2, -1).numpy()

            # Normalize embeddings and reduce dims of embeddings by PCA
            normalized_embeddings = (embeddings - np.mean(embeddings, axis=0)) / (np.std(embeddings, axis=0, ddof=0) + 1e-5)
            d_orig = embeddings.shape[1]
            pca = faiss.PCAMatrix(d_orig, self.pca_dim)
            pca.train(normalized_embeddings[:self.num_train_pca])
            assert pca.is_trained
            transformed_feats = pca.apply_py(normalized_embeddings)

            # Cluster transformed feats with kmeans
            results = []
            gt = torch.cat(self.gt, dim=0).cpu().numpy()[valid_masks]
            for k in self.num_pred_classes:
                kmeans = faiss.Kmeans(self.pca_dim, k, niter=50, nredo=5, seed=1, verbose=True, gpu=False, spherical=False)
                kmeans.train(transformed_feats)
                _, pred_labels = kmeans.index.search(transformed_feats, 1)
                clusters = pred_labels.squeeze()

                # Filter predictions by valid masks (removes voc boundary gt class)
                pred_flattened = clusters.reshape(valid_masks.shape[0], 1, res_w, res_w)[valid_masks]
                assert len(np.unique(pred_flattened)) == k
                assert np.max(pred_flattened) == k - 1

                # Calculate mIoU. Do many-to-one matching if k > self.num_gt_classes.
                if k == self.num_gt_classes:
                    results.append((k, k, self.compute_miou(gt, pred_flattened, k, self.num_gt_classes,
                                                            many_to_one=False)))
                else:
                    results.append((k, k, self.compute_miou(gt, pred_flattened, k, self.num_gt_classes,
                                                            many_to_one=True)))
                    results.append((k, f"{k}_prec", self.compute_miou(gt, pred_flattened, k, self.num_gt_classes,
                                                                      many_to_one=True, precision_based=True)))
            return results


def eval_jac(gt: torch.Tensor, pred_mask: torch.Tensor, with_boundary: bool = True) -> float:
    """
    Calculate Intersection over Union averaged over all pictures. with_boundary flag, if set, doesn't filter out the
    boundary class as background.
    """
    jacs = 0
    for k, mask in enumerate(gt):
        if with_boundary:
            gt_fg_mask = (mask != 0).float()
        else:
            gt_fg_mask = ((mask != 0) & (mask != 255)).float()
        intersection = gt_fg_mask * pred_mask[k]
        intersection = torch.sum(torch.sum(intersection, dim=-1), dim=-1)
        union = (gt_fg_mask + pred_mask[k]) > 0
        union = torch.sum(torch.sum(union, dim=-1), dim=-1)
        jacs += intersection / union
    res = jacs / gt.size(0)
    print(res)
    return res.item()


def process_and_store_attentions(attns: List[torch.Tensor], threshold: float, spatial_res: int, split: str,
                                 experiment_folder: str):
    # Concat and average attentions over all heads.
    attns_processed = torch.cat(attns, dim = 0)
    attns_processed = sum(attns_processed[:, i] * 1 / attns_processed.size(1) for i in range(attns_processed.size(1)))
    attns_processed = attns_processed.reshape(-1, 1, spatial_res, spatial_res)
    # Transform attentions to binary fg mask
    th_attns = process_attentions(attns_processed, spatial_res, threshold=threshold, blur_sigma=0.6)
    torch.save(th_attns, os.path.join(experiment_folder, f'attn_{split}.pt'))


def compute_features(loader: DataLoader, model: nn.Module, device: str, spatial_res: int) -> \
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    # Compute ViT model features on all data provided by loader. Also return attentions and gt masks.
    model.to(device)
    feats = []
    all_masks = []
    attns = []
    for i, (imgs, mask) in enumerate(loader):
        bs = imgs.size(0)
        assert torch.max(mask).item() <= 1 and torch.min(mask).item() >= 0
        gt = mask * 255
        # Get backbone embeddings for batch
        with torch.no_grad():
            embeddings, attn = model.forward_backbone(imgs.to(device), last_self_attention=True)
            embeddings = embeddings[:, 1:].reshape(bs * spatial_res**2, model.embed_dim)
        attns.append(attn.cpu())
        feats.append(embeddings.cpu())
        all_masks.append(gt.cpu())
    model.cpu()
    return feats, all_masks, attns


def store_and_compute_features(datamodule: VOCDataModule, pca_dim: int, model: nn.Module, device: str,
                               spatial_res: int, experiment_folder: str, gt_save_folder: str = None,
                               save_attn : bool = True):
    train_feats, train_gt, train_attns = compute_features(datamodule.train_dataloader(),
                                                          model, device, spatial_res)
    print("computed train features")
    val_feats, val_gt, val_attns = compute_features(datamodule.val_dataloader(),
                                               model, device, spatial_res)
    print("computed val features")

    transformed_feats = normalize_and_transform(torch.cat((
        torch.cat(train_feats, dim=0), torch.cat(val_feats, dim=0)), dim=0), pca_dim)
    transformed_feats = transformed_feats.reshape(len(datamodule.voc_train) + len(datamodule.voc_val),
                                                  spatial_res**2, pca_dim)
    print(f"Normalized and PCA to {pca_dim} dims with shape {transformed_feats.size()}")

    # Store to disk
    print("Storing to disk")
    os.makedirs(experiment_folder, exist_ok=True)
    torch.save(transformed_feats[:len(datamodule.voc_train)], os.path.join(experiment_folder, "all_pascal_train.pt"))
    torch.save(transformed_feats[len(datamodule.voc_train):], os.path.join(experiment_folder, "all_pascal_val.pt"))

    if gt_save_folder is not None:
        train_path = os.path.join(gt_save_folder, "all_gt_masks_train_voc12.pt")
        if not os.path.exists(train_path):
            torch.save(torch.cat(train_gt), train_path)
        val_path = os.path.join(gt_save_folder, "all_gt_masks_val_voc12.pt")
        if not os.path.exists(val_path):
            torch.save(torch.cat(val_gt), val_path)

    # postprocess attentions
    if save_attn:
        process_and_store_attentions(train_attns, 0.65, spatial_res, "train", experiment_folder)
        process_and_store_attentions(val_attns, 0.65, spatial_res, "val", experiment_folder)


def get_backbone_weights(arch: str, method: str, patch_size: int = None,
                         weight_prefix: Optional[str]= "model", ckpt_path: str = None) -> Dict[str, torch.Tensor]:
    """
    Load backbone weights into formatted state dict given arch, method and patch size as identifiers.
    :param arch: Target architecture. Currently supports resnet50, vit-small and vit-base.
    :param method: Method identifier.
    :param patch_size: Patch size of ViT. Ignored if arch is not ViT.
    :param weight_prefix: Optional prefix of weights to match model naming.
    :param ckpt_path: Optional path to checkpoint containing state_dict to be processed.
    :return: Dictionary mapping to weight Tensors.
    """
    def identity_transform(x): return x
    arch_to_args = {
        'vit-small16-dino': ("https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth",
                             torch.hub.load_state_dict_from_url,
                             lambda x: x["teacher"]),
        'vit-small16-ours': (ckpt_path,
                             partial(torch.load, map_location=torch.device('cpu')),
                             lambda x: {k: v for k, v in x.items() if k.startswith('model')}),
        'vit-small16-mocov3': ("https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar",
                               torch.hub.load_state_dict_from_url,
                               lambda x: {k: v for k, v in x.items() if k.startswith('module.base_encoder')}),
        'vit-small16-sup_vit': ('vit_small_patch16_224',
                            lambda x: timm.create_model(x,  pretrained=True).state_dict(),
                            identity_transform),
        'vit-base16-dino': ("https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth",
                            torch.hub.load_state_dict_from_url,
                            lambda x: x["teacher"]),
        'vit-base8-dino': ("https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth",
                           torch.hub.load_state_dict_from_url,
                           lambda x: x["teacher"]),
        'vit-base16-mae': ("https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
                           torch.hub.load_state_dict_from_url,
                           lambda x: x["model"]),
        'resnet50-sup_resnet': ("",
                                lambda x: models.resnet50(pretrained=True).state_dict(),
                                identity_transform),
        'resnet50-maskcontrast': (ckpt_path,
                                  torch.load,
                                  lambda  x: x["model"]),
        'resnet50-swav': ("https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
                          torch.hub.load_state_dict_from_url,
                          identity_transform),
        'resnet50-moco': ("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
                          torch.hub.load_state_dict_from_url,
                          identity_transform),
        'resnet50-densecl': (ckpt_path,
                             torch.load,
                             identity_transform),
    }
    arch_to_args['vit-base16-ours'] = arch_to_args['vit-base8-ours'] = arch_to_args['vit-small16-ours']

    if "vit" in arch:
        url, loader, weight_transform = arch_to_args[f"{arch}{patch_size}-{method}"]
    else:
        url, loader, weight_transform = arch_to_args[f"{arch}-{method}"]
    weights = loader(url)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    weights = weight_transform(weights)
    prefix_idx, prefix = get_backbone_prefix(weights, arch)
    if weight_prefix:
        return {f"{weight_prefix}.{k[prefix_idx:]}": v for k, v in weights.items() if k.startswith(prefix)
                and "head" not in k and "prototypes" not in k}
    return {f"{k[prefix_idx:]}": v for k, v in weights.items() if k.startswith(prefix)
            and "head" not in k and "prototypes" not in k}


def get_backbone_prefix(weights: Dict[str, torch.Tensor], arch: str) -> Optional[Tuple[int, str]]:
    # Determine weight prefix if returns empty string as prefix if not existent.
    if 'vit' in arch:
        search_suffix = 'cls_token'
    elif 'resnet' in arch:
        search_suffix = 'conv1.weight'
    else:
        raise ValueError()
    for k in weights:
        if k.endswith(search_suffix):
            prefix_idx = len(k) - len(search_suffix)
            return prefix_idx, k[:prefix_idx]


def cosine_scheduler(base_value: float, final_value: float, epochs: int, niter_per_ep: int):
    # Construct cosine schedule starting at base_value and ending at final_value with epochs * niter_per_ep values.
    iters = np.arange(epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def process_attentions(attentions: torch.Tensor, spatial_res: int, threshold: float = 0.5, blur_sigma: float = 0.6) \
        -> torch.Tensor:
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    attentions = attentions.reshape(attentions.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k
            if np.sum(mask) <= 2:
                th_attn[j, 0][mask] = 0
    return th_attn.detach()


def normalize_and_transform(feats: torch.Tensor, pca_dim: int) -> torch.Tensor:
    feats = feats.numpy()
    # Iteratively train scaler to normalize data
    bs = 100000
    num_its = (feats.shape[0] // bs) + 1
    scaler = StandardScaler()
    for i in range(num_its):
        scaler.partial_fit(feats[i * bs:(i + 1) * bs])
    print("trained scaler")
    for i in range(num_its):
        feats[i * bs:(i + 1) * bs] = scaler.transform(feats[i * bs:(i + 1) * bs])
    print(f"normalized feats to {feats.shape}")
    # Do PCA
    pca = faiss.PCAMatrix(feats.shape[-1], pca_dim)
    pca.train(feats)
    assert pca.is_trained
    transformed_val = pca.apply_py(feats)
    print(f"val feats transformed to {transformed_val.shape}")
    return torch.from_numpy(transformed_val)


def cluster(pca_dim: int, transformed_feats: np.ndarray, spatial_res: int, k: int, seed: int = 1,
            mask: torch.Tensor = None, spherical: bool = False):
    """
    Computes k-Means and retrieve assignments for each feature vector. Optionally the clusters are only computed on
    foreground vectors if a mask is provided. In this case tranformed_feats is already expected to contain only the
    foreground vectors.
    """
    print(f"start clustering with {seed}")
    kmeans = faiss.Kmeans(pca_dim, k, niter=100, nredo=5, verbose=True,
                          gpu=False, spherical=spherical, seed=seed)
    kmeans.train(transformed_feats)
    print("kmeans trained")
    _, pred_labels = kmeans.index.search(transformed_feats, 1)
    clusters = pred_labels.squeeze()
    print("index search done")

    # Apply fg mask if provided.
    if mask is not None:
        preds = torch.zeros_like(mask) + k
        preds[mask.bool()] = torch.from_numpy(clusters).float()
    else:
        preds = torch.from_numpy(clusters.reshape(-1, 1, spatial_res, spatial_res))
    return preds
