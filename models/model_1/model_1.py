"""Anomaly Detection via Reverse Distillation from One-Class Embedding.

https://arxiv.org/abs/2201.10703v2
"""

from __future__ import annotations

import numpy as np
import cv2
from matplotlib import pyplot as plt
import gc
import logging
from pathlib import Path
import os
from abc import ABC
from typing import Any, OrderedDict
from warnings import warn
from kornia.filters import gaussian_blur2d
from omegaconf import ListConfig, OmegaConf
from argparse import ArgumentParser, Namespace

import torch
from torch import Tensor, nn, optim
from torchmetrics import Metric
import torchvision.transforms as T
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

#from dataset import CustomDataModule, get_configurable_parameters
#from tiler import Tiler
#from models.model_1 import (
#    loss_function, 
#    de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50, 
#    resnet18, resnet34, resnet50, wide_resnet50_2,
#)
from .de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from .resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from .loss import loss_function, loss_concat

logger = logging.getLogger(__name__)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of model to train/evaluate")
    args = parser.parse_args()
    return args

class AnomalyMapGenerator(nn.Module): 
    """Generate Anomaly Heatmap"""

    def __init__(self, image_size: ListConfig | tuple, sigma: int = 4, mode: str = "multiply") -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma
        self.kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

        if mode not in ("add", "multiply"):
            raise ValueError(f"Found mode {mode}. Only multiply and add are supported.")
        self.mode = mode

    def forward(self, student_features: list[Tensor], teacher_features: list[Tensor]) -> Tensor:
        """Computes anomaly map given encoder and decoder features."""

        if self.mode == "multiply":
            anomaly_map = torch.ones(
                [student_features[0].shape[0], 1, *self.image_size], device=student_features[0].device
            )  # b c h w
        elif self.mode == "add":
            anomaly_map = torch.zeros(
                [student_features[0].shape[0], 1, *self.image_size], device=student_features[0].device
            )

        for student_feature, teacher_feature in zip(student_features, teacher_features):
            distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
            distance_map = torch.unsqueeze(distance_map, dim=1)
            distance_map = F.interpolate(distance_map, size=self.image_size, mode="bilinear", align_corners=True)
            if self.mode == "multiply":
                anomaly_map *= distance_map
            elif self.mode == "add":
                anomaly_map += distance_map

        anomaly_map = gaussian_blur2d(
            anomaly_map, kernel_size=(self.kernel_size, self.kernel_size), sigma=(self.sigma, self.sigma)
        )

        return anomaly_map # return anomaly maps of length batch
        
##################################################################################################################### 

class Model1(pl.LightningModule, ABC): #ReverseDistillationModel

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__()
        logger.info("Initializing model: %s", self.__class__.__name__)
        
        #self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
        #self.loss = ReverseDistillationLoss()
        self.learning_rate = hparams.model.lr
        self.beta1 = hparams.model.beta1
        self.beta2 = hparams.model.beta2
        
        #self.encoder = FeatureExtractor(backbone=hparams.model.backbone, pre_trained=hparams.model.pre_trained, layers=hparams.model.layers)
        #self.bottleneck = get_bottleneck_layer(hparams.model.backbone)
        #self.decoder = get_decoder(hparams.model.backbone)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #print("Device: {}".format(device))
        encoder, bottleneck = wide_resnet50_2(pretrained=True)
        self.encoder = encoder.to(device)
        self.bottleneck = bottleneck.to(device)
        self.decoder = de_wide_resnet50_2(pretrained=False).to(device)

        self.tiler: Tiler | None = None
        if self.tiler:
            image_size = (self.tiler.tile_size_h, self.tiler.tile_size_w)
        else:
            image_size = (hparams.dataset.image_size, hparams.dataset.image_size)
        # print('>>>>>>>>>>> Image size for anomaly map generator: {}'.format(image_size)) # Result: [256, 256]
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=image_size, mode=hparams.model.anomaly_map_mode)
        # self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size, mode=hparams.model.anomaly_map_mode)
        
        #self.callbacks: list[Callback]
        #self.threshold_method = hparams.metrics.threshold.method
        #self.image_threshold = AnomalyScoreThreshold().cpu()
        #self.pixel_threshold = AnomalyScoreThreshold().cpu()
        #self.normalization_metrics: Metric
        #self.image_metrics: AnomalibMetricCollection
        #self.pixel_metrics: AnomalibMetricCollection

    #def configure_callbacks(self) -> list[EarlyStopping]:
        #early_stopping = EarlyStopping(
            #monitor=self.hparams.model.early_stopping.metric,
            #patience=self.hparams.model.early_stopping.patience,
            #mode=self.hparams.model.early_stopping.mode,
        #)
        #return [early_stopping]

    def configure_optimizers(self):
    #when initializing an optimizer, you explicitly tell it what parameters (params) of the model it should be updating. The gradients are "stored"
    #by the tensors themselves (they have a 'grad' and 'requires_grad' attributes) once you call backward() on the loss. After computing the gradients
    #for all tensors in the model, calling optimizer.step() makes the optimizer iterate over all parameters (tensors) it is supposed to update and use
    #their internally stored grad to update their values.
        return optim.Adam(
            params=list(self.decoder.parameters()) + list(self.bottleneck.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

    def training_step(self, batch: dict[str, str | Tensor]):
        loss = loss_function(self.forward(batch["image"]))
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss} # Feature Map

    def validation_step(self, batch: dict[str, str | Tensor], batch_idx): #*args, **kwargs):
        del batch_idx #args, kwargs  # These variables are not used.
        batch["anomaly_maps"] = self.forward(batch["image"])
        return batch

    #def forward(self, batch: dict[str, str | Tensor]):
        #return self.model(batch)

    def forward(self, images: Tensor) -> Tensor | list[Tensor] | tuple[list[Tensor]]:

        self.encoder.eval()

        if self.training: # NOTE: self.training is an attribute of pl.LightningModule class -> confirm
            encoder_features = self.encoder(images)
            #encoder_features = list(encoder_features.values())
            decoder_features = self.decoder(self.bottleneck(encoder_features))
        else:
            with torch.no_grad():
                encoder_features = self.encoder(images)
                #encoder_features = list(encoder_features.values())
                decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.training: # NOTE: self.training is an attribute of pl.LightningModule class -> confirm
            output = encoder_features, decoder_features
        else:
            output = self.anomaly_map_generator(encoder_features, decoder_features)

        return output

#####################################################################################################################
# The below is just for testing
#####################################################################################################################

def get_configurable_parameters(
    model_name: str | None = None,
    config_path: Path | str | None = None,
    weight_file: str | None = None,
    #config_filename: str | None = "config",
    #config_file_extension: str | None = "yaml",
) -> DictConfig | ListConfig:

    config_path = Path(os.getcwd(), 'models', model_name, 'config.yaml')
    config = OmegaConf.load(config_path)

    # keep track of the original config file because it will be modified
    config_original: DictConfig = config.copy()

    # if the seed value is 0, notify a user that the behavior of the seed value zero has been changed.
    if config.project.get("seed") == 0:
        warn(
            "The seed value is now fixed to 0. "
            "Up to v0.3.7, the seed was not fixed when the seed value was set to 0. "
            "If you want to use the random seed, please select `None` for the seed value "
            "(`null` in the YAML file) or remove the `seed` key from the YAML file."
        )
        
    if not isinstance(config.dataset.center_crop, int):
        #print("center_crop changed to None type")
        config.dataset.center_crop = None

    return config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# below is for testing model code is working properly
if __name__ == "__main__":
    args = get_args()
    config = get_configurable_parameters(model_name=args.model) #, config_path=args.config)
        
    model = Model1(config)
    print("Total number of trainable model parameters: {}".format(count_parameters(model)))

# Command Line:
# conda deactivate
# conda activate anomalib_env2
# cd '/home/brionyf/Documents/GitHub/anomaly-detection/models/model_1'
# python model_1.py --model 'model_1'
