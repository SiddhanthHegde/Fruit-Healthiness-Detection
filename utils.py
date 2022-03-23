import torch

def calc_iou(pred, target, num_classes=3):
    
    pred = pred.argmax(1)
    ious = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        union = ((target == (i)) | (pred == (i))).sum().float()
        intersection = ((target == (i)) & (pred == (i))).sum().float()
        ious[i] = intersection/(union+1e-6)
    return ious