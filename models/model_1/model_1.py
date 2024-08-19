"""Anomaly Detection via Reverse Distillation from One-Class Embedding.

https://arxiv.org/abs/2201.10703v2
"""

import os
import logging
from pathlib import Path
from abc import ABC

from warnings import warn
from kornia.filters import gaussian_blur2d
from omegaconf import ListConfig, OmegaConf, DictConfig
from argparse import ArgumentParser

import torch
from torch import Tensor, optim
import torch.nn.functional as F
from torchmetrics import Metric
import pytorch_lightning as pl
import torchvision.transforms as T

#from dataset import CustomDataModule, get_configurable_parameters
#from tiler import Tiler
#from models.model_1 import (
#    loss_function, 
#    de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50, 
#    resnet18, resnet34, resnet50, wide_resnet50_2,
#)
from not_needed.tiler import Tiler
# from .de_resnet import de_resnet50
# from .resnet import resnet50
# from .loss import loss_function, loss_concat, loss_custom
from .loss import ReverseDistillationLoss
from .components import (
    FeatureExtractor,
    get_bottleneck_layer,
    get_decoder,
)

logger = logging.getLogger(__name__)

def calculate_anomaly_map(encoder_features, decoder_features, image_size, sigma=4, mode="multiply"):
    """Computes anomaly heatmap given encoder and decoder features."""

    # image_size = image_size if isinstance(image_size, tuple) else tuple(image_size,image_size)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

    anomaly_map = torch.ones([encoder_features[0].shape[0], 1, image_size[0], image_size[1]], device=encoder_features[0].device)  # b c h w
    if mode == "add":
        anomaly_map *= 0

    for student_feature, teacher_feature in zip(encoder_features, decoder_features):
        distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
        distance_map = torch.unsqueeze(distance_map, dim=1)
        distance_map = F.interpolate(distance_map, size=image_size, mode="bilinear", align_corners=True)
        if mode == "multiply":
            anomaly_map *= distance_map
        elif mode == "add":
            anomaly_map += distance_map

    anomaly_map = gaussian_blur2d(anomaly_map, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))

    return anomaly_map # return anomaly maps of length batch (type Tensor)
        
##################################################################################################################### 

class Model1(pl.LightningModule, ABC): #ReverseDistillationModel

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__()
        logger.info("Initializing model: %s", self.__class__.__name__)
        self.save_hyperparameters(hparams)
        self.learning_rate = hparams.model.lr
        self.beta1 = hparams.model.beta1
        self.beta2 = hparams.model.beta2
        self.image_size = hparams.dataset.image_size
        self.resize_ratio = hparams.dataset.resize_ratio
        self.mode = hparams.model.anomaly_map_mode
        self.layers = hparams.model.layers
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #print("Device: {}".format(device))

        # encoder_backbone = hparams.model.backbone
        # self.encoder = FeatureExtractor(backbone=encoder_backbone, pre_trained=hparams.model.pre_trained, layers=hparams.model.layers)
        # self.bottleneck = get_bottleneck_layer(hparams.model.backbone)
        # self.decoder = get_decoder(hparams.model.backbone)
        self.anomalib = True
        if self.anomalib:
            self.encoder = FeatureExtractor(backbone=hparams.model.backbone, pre_trained=hparams.model.pre_trained, layers=self.layers)
            self.bottleneck = get_bottleneck_layer(hparams.model.backbone)
            self.decoder = get_decoder(hparams.model.backbone)
        else:
            pass
            # encoder, bottleneck = resnet50(pretrained=True) # TODO: don't hardcode backbone, read in from config (like above)
            # self.encoder = encoder.to(device)
            # self.bottleneck = bottleneck.to(device)
            # self.decoder = de_resnet50(pretrained=False).to(device) # TODO: don't hardcode backbone, read in from config (like above)

        # self.tiler: Tiler | None = None
        if hparams.dataset.tiling.apply:
            self.tiler = Tiler(hparams.dataset.tiling.tile_size)
        else:
            self.tiler = None

        self.loss_function = ReverseDistillationLoss()
        self.normal_loss = 0
        self.anomaly_loss = 0

        # self.threshold_value = 0.
        # self.norm_metric = MinMax()
        self.normalization_metrics: Metric # TODO: rename to 'norm_metric'

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

    def forward(self, images: Tensor) -> Tensor | list[Tensor] | tuple[list[Tensor]]:
    # def forward(self, images: Tensor, images_rot: Tensor) -> Tensor | list[Tensor] | tuple[list[Tensor]]:
        # print(">>>>>>>>>>>>>>> now in forward...")
        self.encoder.eval()

        if self.tiler:
            images = self.tiler.tile(images)
            images = T.Resize((256, 256))(images)  # CUSTOM: TODO - replace (256,256) with input_size
            # print(">>>>>>>>>>>>>>> forward image size after tiling: {}".format(images.shape))  # result is: torch.Size([12, 3, 256, 256])

        if self.training: # TODO: self.training is an attribute of pl.LightningModule class -> confirm
            encoder_features = self.encoder(images)
            if self.anomalib: encoder_features = list(encoder_features.values())
            decoder_features = self.decoder(self.bottleneck(encoder_features))
        else:
            with torch.no_grad():
                encoder_features = self.encoder(images)
                if self.anomalib: encoder_features = list(encoder_features.values())
                decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.training: # TODO: self.training is an attribute of pl.LightningModule class -> confirm
            output = encoder_features, decoder_features
        else:
            image_size = images.shape[2:] #[int(val / self.resize_ratio) for val in images.shape[2:]]
            # print(">>>>>>>>>>>>>>>> Anomaly Maps: image size = {}".format(image_size))
            output = calculate_anomaly_map(encoder_features, decoder_features, image_size, mode=self.mode)
            if self.tiler:
                output = self.tiler.untile(output)

        return output

    def training_step(self, batch: dict[str, str | Tensor]):
        # loss = loss_function(self.forward(batch["image"]), self.image_size)
        batch["model_outputs"] = self.forward(batch["image"]) #, batch["rotated"])
        # loss = self.loss(batch, self.image_size)
        self.normal_loss, self.anomaly_loss = self.loss_function(batch)#, self.resize_ratio)
        loss = 0.3 * self.normal_loss + 1.0 * self.anomaly_loss
        # loss = self.normal_loss * self.anomaly_loss
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        # print(">>>>>>>>>>>>>>>> Normal Loss: {} \tAnomaly Loss: {}".format(self.normal_loss, self.anomaly_loss))
        pass

    def validation_step(self, batch, *args, **kwargs):
        del args, kwargs  # Unused variables
        # print(">>>>>>>>>>>>>>> here in validation...")
        batch["anomaly_maps"] = self.forward(batch["image"]) #, batch["rotated"])
        return batch

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        del batch_idx, dataloader_idx  # Unused variables

        # print(">>>>>>>>>>>>>>> predict batch size: {} item(s) \{} shape".format(len(batch), batch["image"].shape))
        outputs = self.validation_step(batch) # get anomaly maps
        # print('>>>>>>>>>>>>>>> predict outputs size: {}'.format(outputs["anomaly_maps"].shape))  # result is: torch.Size([1, 1, 542, 800])

        # self._compute_adaptive_threshold(outputs["anomaly_maps"]) #, batch["mask"])
        # # print(">>>>>>>>>>>>>>> Computed threshold value to be: {}".format(self.threshold_value))
        #
        # if outputs is not None and isinstance(outputs, dict):
        #     outputs["pred_masks"] = outputs["anomaly_maps"] >= self.threshold_value

        # gc.collect()
        return outputs

    def test_step(self, batch: dict[str, str | Tensor], batch_idx: int, *args, **kwargs):
        del args, kwargs  # These variables are not used.
        # return self.predict_step(batch, batch_idx)
        return self.validation_step(batch, batch_idx)

    # def _compute_adaptive_threshold(self, pred, target=None):
    #     # Compute the threshold that yields the optimal F1 score.
    #     # if not any(1 in gt for gt in target):
    #     if target is None:
    #         warnings.warn(
    #             "The validation set does not contain any anomalous images. Adaptive threshold will be based on the "
    #             "highest anomaly score observed in the normal validation images."
    #         )
    #         # print(">>>>>>>>>>>>>>> pred to mask info: {} device \t{} shape".format(pred.device, pred.shape))
    #         target = torch.zeros(size=pred.shape).to('cuda')
    #         # target = torch.from_numpy(np.zeros(pred.shape))
    #     pr_curve = PrecisionRecallCurve(task="binary")
    #     precision, recall, thresholds = pr_curve(pred, target) # super().compute()
    #     f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
    #     self.threshold_value = thresholds
    #     if thresholds.dim() != 0:
    #         self.threshold_value = thresholds[torch.argmax(f1_score)]
    #     # print(">>>>>>>>>>>>>>> image min: {} image max: {} threshold: {} precision: {} recall: {} f1_score: {}".format(pred.min(), pred.max(), self.threshold_value, precision, recall, f1_score))
    #

#####################################################################################################################
# The below is just for testing
#####################################################################################################################

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of model to train/evaluate")
    return parser.parse_args()

def get_config_params(
    model_name: str | None = None,
    config_path: Path | str | None = None,
    weight_file: str | None = None,
) -> DictConfig | ListConfig:

    config_path = Path(os.getcwd(), 'models', model_name, 'config.yaml')
    config = OmegaConf.load(config_path)

    # # keep track of the original config file because it will be modified
    # config_original = config.copy()

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
    config = get_config_params(model_name=args.model) #, config_path=args.config)
        
    model = Model1(config)
    print("Total number of trainable model parameters: {}".format(count_parameters(model)))

# Command Line:
# conda deactivate
# conda activate anomalib_env2
# cd '/home/brionyf/Documents/GitHub/anomaly-detection/models/model_1'
# python model_1.py --model 'model_1'
