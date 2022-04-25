import numpy as np
import torch

from experiments.bf1_eval.bfscore import bfscore
from tqdm import tqdm


def evaluate_bf_score(path_to_segmentation, gt_mask_path, match_threshold: int = 16):
    print(path_to_segmentation)
    gt = torch.load(gt_mask_path)
    scores = []
    pred_fg = torch.nn.functional.interpolate(torch.load(path_to_segmentation).float().cpu(), size=(448, 448), mode='nearest')
    for k in tqdm(range(gt.size(0))):
        gt_fg_mask = (gt[k] == 0).squeeze().numpy().astype(np.uint8)
        pred_mask = pred_fg[k].squeeze().numpy().astype(np.uint8)
        if len(np.unique(pred_mask)) == 1:
            print(np.unique(pred_mask))
            print("empty fg mask. f1 of 0")
            score = [0]
        else:
            score, areas_gt = bfscore(gt_fg_mask, pred_mask, threshold=match_threshold)
        scores.append(score[0])

    print("overall boundary score")
    print(np.nanmean(np.array(scores)))

if __name__ == "__main__":
    # Predicted foreground segmentations
    predicted_fg_mask = "<path_to_fg_mask_prediction>"

    # gt masks
    gt_mask = "<path_to_pvoc12_val_gt_mask>"

    evaluate_bf_score(predicted_fg_mask, gt_mask)