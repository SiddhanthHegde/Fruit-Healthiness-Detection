import segmentation_models_pytorch as smp
import pytorch_lightning as pl

class SegmentationTestModel(pl.LightningModule):
    def __init__(self):
        super(SegmentationTestModel, self).__init__()
        self.seg_model = smp.Linknet('resnet50', classes=3)
    
    def forward(self, x):
        return self.seg_model(x)