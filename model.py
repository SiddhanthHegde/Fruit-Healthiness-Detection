import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet, UnetPlusPlus, MAnet, Linknet, PSPNet, PAN, DeepLabV3, DeepLabV3Plus
import pytorch_lightning as pl
import wandb
from torchmetrics.functional import dice_score
from dataset2 import SegmentationDataset
from utils import calc_iou

class SegmentationModel(pl.LightningModule):
    def __init__(self, cfg):
        super(SegmentationModel, self).__init__()
        self.cfg = cfg
        model_arch = cfg.MODEL.model_arch.lower()
        all_models = [Unet, UnetPlusPlus, MAnet, Linknet, PSPNet, PAN, DeepLabV3, DeepLabV3Plus]
        model_dict = {x.__name__.lower() : x for x in all_models}
        if model_arch not in model_dict.keys():
            print(f'{model_arch} not available, available architectures: {str(model_dict.keys())}')
        
        if model_arch == 'panet' or model_arch == 'deeplabv3' or model_arch == 'deeplabv3plus':
            assert cfg.TRAIN.batch_size != 1, 'Batch size 1 cannot be used as batch normalisation requires batch size > 1'

        architecture = model_dict[model_arch]
        
        self.seg_model = architecture(
            encoder_name = cfg.MODEL.encoder_name,
            in_channels = cfg.MODEL.in_channels,
            classes = cfg.MODEL.classes,
            encoder_weights = cfg.MODEL.encoder_weights
        )

        del(all_models)
        del(model_dict)

        class_weights = torch.tensor(
            cfg.TRAIN.class_weights).float().to('cuda')
        
        self.class_weights = F.softmax(class_weights, dim=-1)
        self.train_dataset = SegmentationDataset(self.cfg, self.cfg.DATASET.train_json)
        self.val_dataset = SegmentationDataset(self.cfg, self.cfg.DATASET.val_json)
    
    def forward(self, x):
        x = self.seg_model(x)
        return x

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc
    
    def training_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['label'].long()

        output = self.forward(image)
        entropy_loss = F.cross_entropy(output, label, weight=self.class_weights)
        acc = self.pixel_acc(output, label)
        miou = calc_iou(output, label).mean()

        dice_sc = dice_score(output, label, bg=True)

        if(batch_idx % self.cfg.TRAIN.wandb_iters == 0):
            self.log('train_loss',entropy_loss)

        return {
            'loss': entropy_loss,
            'train_acc': acc,
            'miou': miou,
            'dice_score': dice_sc
        }

    def training_epoch_end(self, outputs):
        miou = torch.stack([x['miou'] for x in outputs]).mean()
        accs = torch.stack([x['train_acc'] for x in outputs]).mean()
        self.log('train_acc',accs)
        self.log('train_miou',miou)

    def validation_step(self, batch, batch_idx):
        image= batch['image']
        label = batch['label'].long()

        output = self.forward(image)
        entropy_loss = F.cross_entropy(output, label)
        miou = calc_iou(output, label).mean()
        dice_sc = dice_score(output, label, bg=True)

        return {
            'val_loss': entropy_loss,
            'val_dice_score': dice_sc,
            'val_miou': miou
        }

    def validation_epoch_end(self, outputs):
        val_entropy_loss = torch.stack(
            [x['val_loss'] for x in outputs]).mean()
        val_dice_score = torch.stack([x['val_dice_score']
                                     for x in outputs]).mean()
        val_miou = torch.stack([x['val_miou'] for x in outputs]).mean()
        logs = {'val_entropy_loss': val_entropy_loss,
                'val_dice_score': val_dice_score,
                'val_miou': val_miou
                }
        self.log('val_entropy_loss',val_entropy_loss)
        self.log('val_dice_score', val_dice_score)
        self.log('val_miou',val_miou)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                                lr=self.cfg.TRAIN.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            patience=self.cfg.TRAIN.scheduler_patience,
            factor=self.cfg.TRAIN.lr_reduce_factor
        )
        return {
            "optimizer": opt, 
            "lr_scheduler":{
                "scheduler": lr_scheduler,
                "monitor": "val_entropy_loss",
                }
            }

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.TRAIN.batch_size,
            shuffle=self.cfg.TRAIN.shuffle,
            num_workers=self.cfg.TRAIN.num_workers,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.VAL.batch_size,
            shuffle=self.cfg.VAL.shuffle,
            num_workers=self.cfg.VAL.num_workers,
            drop_last=True
        )