import numpy as np
import torch
import wandb
from tqdm import tqdm
from torchmetrics.functional import dice_score

def calc_iou(pred, target, num_classes=3):
    
    pred = pred.argmax(1)
    ious = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        union = ((target == (i)) | (pred == (i))).sum().float()
        intersection = ((target == (i)) & (pred == (i))).sum().float()
        ious[i] = intersection/(union+1e-6)
    return ious

def log_iou_dice(model, cfg):

    model.eval()
    val_dataloader = model.val_dataloader()
    n = len(val_dataloader)
    val_it = iter(val_dataloader)
    ious = []
    dice_scores = []
    for _ in tqdm(range(n)):
        data_dict = next(val_it)
        image = data_dict['image']
        label = data_dict['label']

        pred = model.forward(image)
        # calc miou
        iou = calc_iou(pred, label, cfg.MODEL.classes)
        dice_sc = dice_score(pred, label, bg=True)
        ious.append(iou.unsqueeze(0))
        dice_scores.append(dice_sc.item())
    ious = torch.cat(ious, 0).mean(0)
    avg_dice = np.array(dice_scores).mean()

    data = [ious.tolist()]
    wandb.log({"ious_classwise": wandb.Table(
        data=data, columns=['1', '2', '3']),
        'dice_score': avg_dice})
    return ious, avg_dice