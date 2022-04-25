import random
import torch
import torchvision

from PIL import ImageFilter, Image
from typing import List, Tuple, Dict
from torch import Tensor
from torchvision.transforms import functional as F


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709 following
    https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/src/multicropdataset.py#L64
    """
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x: Image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class LeopartTransforms:

    def __init__(self,
                 size_crops: List[int],
                 nmb_crops: List[int],
                 min_scale_crops: List[float],
                 max_scale_crops: List[float],
                 jitter_strength: float = 0.2,
                 min_intersection: float = 0.01,
                 blur_strength: float = 1):
        """
        Main transform used for fine-tuning with Leopart. Implements multi-crop and calculates the corresponding
        crop bounding boxes for each crop-pair.
        :param size_crops: size of global and local crop
        :param nmb_crops: number of global and local crop
        :param min_scale_crops: the lower bound for the random area of the global and local crops before resizing
        :param max_scale_crops: the upper bound for the random area of the global and local crops before resizing
        :param jitter_strength: the strength of jittering for brightness, contrast, saturation and hue
        :param min_intersection: minimum percentage of intersection of image ares for two sampled crops from the
        same picture should have. This makes sure that we can always calculate a loss for each pair of
        global and local crops.
        :param blur_strength: the maximum standard deviation of the Gaussian kernel
        """
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        assert 0 < min_intersection < 1
        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops
        self.min_intersection = min_intersection

        # Construct color transforms
        self.color_jitter = torchvision.transforms.ColorJitter(
            0.8 * jitter_strength, 0.8 * jitter_strength, 0.8 * jitter_strength,
            0.2 * jitter_strength
        )
        color_transform = [torchvision.transforms.RandomApply([self.color_jitter], p=0.8),
                           torchvision.transforms.RandomGrayscale(p=0.2)]
        blur = GaussianBlur(sigma=[blur_strength * .1, blur_strength * 2.])
        color_transform.append(torchvision.transforms.RandomApply([blur], p=0.5))
        self.color_transform = torchvision.transforms.Compose(color_transform)

        # Construct final transforms
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.final_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

        # Construct randomly resized crops transforms
        self.rrc_transforms = []
        for i in range(len(self.size_crops)):
            random_resized_crop = torchvision.transforms.RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )
            self.rrc_transforms.extend([random_resized_crop] * self.nmb_crops[i])

    def __call__(self, sample: torch.Tensor) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        multi_crops = []
        crop_bboxes = torch.zeros(len(self.rrc_transforms), 4)

        for i, rrc_transform in enumerate(self.rrc_transforms):
            # Get random crop params
            y1, x1, h, w = rrc_transform.get_params(sample, rrc_transform.scale, rrc_transform.ratio)
            if i > 0:
                # Check whether crop has min overlap with existing global crops. If not resample.
                while True:
                    # Calculate intersection between sampled crop and all sampled global crops
                    bbox = torch.Tensor([x1, y1, x1 + w, y1 + h])
                    left_top = torch.max(bbox.unsqueeze(0)[:, None, :2],
                                         crop_bboxes[:min(i, self.nmb_crops[0]), :2])
                    right_bottom = torch.min(bbox.unsqueeze(0)[:, None, 2:],
                                             crop_bboxes[:min(i, self.nmb_crops[0]), 2:])
                    wh = _upcast(right_bottom - left_top).clamp(min=0)
                    inter = wh[:, :, 0] * wh[:, :, 1]

                    # set min intersection to at least 1% of image area
                    min_intersection = int((sample.size[0] * sample.size[1]) * self.min_intersection)
                    # Global crops should have twice the min_intersection with each other
                    if i in list(range(self.nmb_crops[0])):
                        min_intersection *= 2
                    if not torch.all(inter > min_intersection):
                        y1, x1, h, w = rrc_transform.get_params(sample, rrc_transform.scale, rrc_transform.ratio)
                    else:
                        break

            # Apply rrc params and store absolute crop bounding box
            img = F.resized_crop(sample, y1, x1, h, w, rrc_transform.size, rrc_transform.interpolation)
            crop_bboxes[i] = torch.Tensor([x1, y1, x1 + w, y1 + h])

            # Apply color transforms
            img = self.color_transform(img)

            # Apply final transform
            img = self.final_transform(img)
            multi_crops.append(img)

        # Calculate relative bboxes for each crop pair from aboslute bboxes
        gc_bboxes, otc_bboxes = self.calculate_bboxes(crop_bboxes)

        return multi_crops, {"gc": gc_bboxes, "all": otc_bboxes}

    def calculate_bboxes(self, crop_bboxes: Tensor):
        # 1. Calculate two intersection bboxes for each global crop - other crop pair
        gc_bboxes = crop_bboxes[:self.nmb_crops[0]]
        left_top = torch.max(gc_bboxes[:, None, :2], crop_bboxes[:, :2])  # [nmb_crops[0], sum(nmb_crops), 2]
        right_bottom = torch.min(gc_bboxes[:, None, 2:], crop_bboxes[:, 2:])  # [nmb_crops[0], sum(nmb_crops), 2]
        # Testing for non-intersecting crops. This should always be true, just as safe-guard.
        assert torch.all((right_bottom - left_top) > 0)

        # 2. Scale intersection bbox with crop size
        # Extract height and width of all crop bounding boxes. Each row contains h and w of a crop.
        ws_hs = torch.stack((crop_bboxes[:, 2] - crop_bboxes[:, 0], crop_bboxes[:, 3] - crop_bboxes[:, 1])).T[:, None]

        # Stack global crop sizes for each bbox dimension
        crops_sizes = torch.repeat_interleave(torch.Tensor([self.size_crops[0]]), self.nmb_crops[0] * 2)\
            .reshape(self.nmb_crops[0], 2)
        if len(self.size_crops) == 2:
            lc_crops_sizes = torch.repeat_interleave(torch.Tensor([self.size_crops[1]]), self.nmb_crops[1] * 2)\
                .reshape(self.nmb_crops[1], 2)
            crops_sizes = torch.cat((crops_sizes, lc_crops_sizes))[:, None]  # [sum(nmb_crops), 1, 2]

        # Calculate x1s and y1s of each crop bbox
        x1s_y1s = crop_bboxes[:, None, :2]

        # Scale top left and right bottom points by percentage of width and height covered
        left_top_scaled_gc = crops_sizes[:2] * ((left_top - x1s_y1s[:2]) / ws_hs[:2])
        right_bottom_scaled_gc = crops_sizes[:2] * ((right_bottom - x1s_y1s[:2]) / ws_hs[:2])
        left_top_otc_points_per_gc = torch.stack([left_top[i] for i in range(self.nmb_crops[0])], dim=1)
        right_bottom_otc_points_per_gc = torch.stack([right_bottom[i] for i in range(self.nmb_crops[0])], dim=1)
        left_top_scaled_otc = crops_sizes * ((left_top_otc_points_per_gc - x1s_y1s) / ws_hs)
        right_bottom_scaled_otc = crops_sizes * ((right_bottom_otc_points_per_gc - x1s_y1s) / ws_hs)

        # 3. Construct bboxes in x1, y1, x2, y2 format from left top and right bottom points
        gc_bboxes = torch.cat((left_top_scaled_gc, right_bottom_scaled_gc), dim=2)
        otc_bboxes = torch.cat((left_top_scaled_otc, right_bottom_scaled_otc), dim=2)
        return gc_bboxes, otc_bboxes


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()
