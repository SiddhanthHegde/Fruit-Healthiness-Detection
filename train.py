from logging import config
import os
import wandb
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor  
from pytorch_lightning import Trainer, seed_everything
from model import SegmentationModel
from utils import log_iou_dice
from configs.defaults_model_config import _C as cfg

#-----------------------------------------------------------------------

def main(cfg):

    model = SegmentationModel(cfg)
    hyperparams = {
        'lr': cfg.TRAIN.lr,
        'batch_size': cfg.TRAIN.batch_size,
        'class_weights': cfg.TRAIN.class_weights,
        'scheduler_patience': cfg.TRAIN.scheduler_patience,
        'lr_reduce_factor': cfg.TRAIN.lr_reduce_factor
    }
    wandb_run_name = f"{cfg.MODEL.model_arch}_{cfg.MODEL.encoder_name}_{str(cfg.MODEL.encoder_weights)}"
    wandb_logger = WandbLogger(name=wandb_run_name,
                                project=cfg.TRAIN.wandb_project_name)
    wandb_logger.log_hyperparams(hyperparams)
    wandb_logger.watch(model)
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    ckpt_path = cfg.TRAIN.init_ckpt

    trainer = Trainer(logger=[wandb_logger], 
                    max_epochs=cfg.TRAIN.n_epochs,
                    gpus=1)

    trainer.fit(model, ckpt_path=ckpt_path)
    ious, dice_sc = log_iou_dice(model, cfg)
    print('Iou: ', ious)
    print('Dice scores: ', dice_sc)

    ckpt_path = os.path.join('./ckpt', f"{wandb_run_name}_{cfg.TRAIN.n_epochs}.ckpt")
    ckpt_base_path = os.path.dirname(ckpt_path)
    os.makedirs(ckpt_base_path, exist_ok=True)

    trainer.save_checkpoint(ckpt_path)
    wandb.save(ckpt_path)

#-----------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'training the required segmentation model and log results')
    parser.add_argument('--cfg', metavar = 'FILE', help = 'path to config file', type = str, default='configs/model_config.yaml')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)

    seed_everything(123)
    main(cfg)