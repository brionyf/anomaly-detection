"""Anomaly Detection via Reverse Distillation from One-Class Embedding.

https://arxiv.org/abs/2201.10703v2
"""

from __future__ import annotations

from abc import ABC
from typing import Any, OrderedDict
from warnings import warn
from kornia.filters import gaussian_blur2d
from omegaconf import ListConfig

import torch
from torch import Tensor, nn, optim
from torchmetrics import Metric
import torchvision.transforms as T
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from models.model_1 import ReverseDistillationLoss
from not_needed.tiler import Tiler




from anomalib.models.reverse_distillation.components import get_bottleneck_layer, get_decoder

from anomalib.models.components import FeatureExtractor
from anomalib.data.utils import boxes_to_anomaly_maps, boxes_to_masks, masks_to_boxes
from anomalib.utils.metrics import AnomalibMetricCollection, AnomalyScoreDistribution, AnomalyScoreThreshold, MinMax
 
 



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
        logger.info("Initializing %s model.", self.__class__.__name__)
        
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
        self.loss = ReverseDistillationLoss()
        self.learning_rate = hparams.model.lr
        self.beta1 = hparams.model.beta1
        self.beta2 = hparams.model.beta2

        self.callbacks: list[Callback]
        
        encoder_backbone = hparams.model.backbone
        self.encoder = FeatureExtractor(backbone=encoder_backbone, pre_trained=hparams.model.pre_trained, layers=hparams.model.layers)
        self.bottleneck = get_bottleneck_layer(hparams.model.backbone)
        self.decoder = get_decoder(hparams.model.backbone)

        self.tiler: Tiler | None = None
        if self.tiler:
            image_size = (self.tiler.tile_size_h, self.tiler.tile_size_w)
        else:
            image_size = (hparams.dataset.image_size, hparams.dataset.image_size)
        # print('>>>>>>>>>>> Image size for anomaly map generator: {}'.format(image_size)) # Result: [256, 256]
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=image_size, mode=hparams.model.anomaly_map_mode)
        # self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size, mode=hparams.model.anomaly_map_mode)
        
        self.threshold_method = hparams.metrics.threshold.method
        self.image_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_threshold = AnomalyScoreThreshold().cpu()
        self.normalization_metrics: Metric
        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

    def configure_callbacks(self) -> list[EarlyStopping]:
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]

    def configure_optimizer(self):
        return optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

    def training_step(self, batch: dict[str, str | Tensor]):
        loss = self.loss(*self.model(batch["image"]))
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss} # Feature Map

    def validation_step(self, batch: dict[str, str | Tensor]):
        batch["anomaly_maps"] = self.model(batch["image"])
        return batch

    #def forward(self, batch: dict[str, str | Tensor]):
        #return self.model(batch)

    def forward(self, images: Tensor) -> Tensor | list[Tensor] | tuple[list[Tensor]]:
        """
        During the training mode the model extracts features from encoder and decoder networks.
        During evaluation mode, it returns the predicted anomaly map.

        Args:
            images (Tensor): Batch of images

        Returns:
            Tensor | list[Tensor] | tuple[list[Tensor]]: Encoder and decoder features in training mode,
                else anomaly maps.
        """
        self.encoder.eval()

        if self.tiler:
            images = self.tiler.tile(images)
            images = T.Resize((256, 256))(images)  # CUSTOM: TODO - replace (256,256) with input_size
        # tiler = Tiler(images, 1024)
        # images = tiler.unfold()
        # images = T.Resize((256, 256))(images)  # CUSTOM: TODO - replace (256,256) with input_size
        # # print(">>>>>>>>>>>>>>>>>>>>> Tiled Images: {}".format(images.shape))

        if self.training:
            encoder_features = self.encoder(images)
            encoder_features = list(encoder_features.values())
            decoder_features = self.decoder(self.bottleneck(encoder_features))
        else:
            with torch.no_grad():
                encoder_features = self.encoder(images)
                encoder_features = list(encoder_features.values())
                decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.training:
            output = encoder_features, decoder_features
        else:
            output = self.anomaly_map_generator(encoder_features, decoder_features)
            if output.shape[0] != images.shape[0]:
                print("Tiled Images: {} {}".format(images.shape, images.dtype))
                print("Anomaly Maps: {} {}".format(output.shape, output.dtype))
            if self.tiler:
                output = self.tiler.untile(output)
            # output = tiler.fold(output)
            # # print("Untiled Image: {} {}".format(output.shape, output.dtype))

        return output

    def predict_step(self, batch: Any):
        #By default, it calls :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`.
        outputs: Tensor | dict[str, Any] = self.validation_step(batch)  # NOTE: Refers to validation_step(...) method in 'lightning_model.py'
        # print('>>>>>>>>>>>>>>> Outputs size: {}'.format(outputs["anomaly_maps"].shape))  # result is: torch.Size([1, 1, 542, 775])

        # image = np.transpose(outputs["anomaly_maps"].cpu().squeeze(0).numpy(), (1, 2, 0))
        # # print('>>>>>>>>>>>>>>> Image size: {}'.format(image.shape))
        # plt.imshow(image, interpolation='nearest')
        # plt.show()

        self._post_process(outputs)  # NOTE: outputs["pred_scores"] created here from outputs["anomaly_maps"]
        if outputs is not None and isinstance(outputs, dict):
            outputs["pred_labels"] = outputs["pred_scores"] >= self.image_threshold.value
            if "anomaly_maps" in outputs.keys():
                outputs["pred_masks"] = outputs["anomaly_maps"] >= self.pixel_threshold.value
                if "pred_boxes" not in outputs.keys():
                    outputs["pred_boxes"], outputs["box_scores"] = masks_to_boxes(
                        outputs["pred_masks"], outputs["anomaly_maps"]
                    )
                    outputs["box_labels"] = [torch.ones(boxes.shape[0]) for boxes in outputs["pred_boxes"]]
            # apply thresholding to boxes
            if "box_scores" in outputs and "box_labels" not in outputs:
                # apply threshold to assign normal/anomalous label to boxes
                is_anomalous = [scores > self.pixel_threshold.value for scores in outputs["box_scores"]]
                outputs["box_labels"] = [labels.int() for labels in is_anomalous]

        # gc.collect()
        return outputs

    def test_step(self, batch: dict[str, str | Tensor], batch_idx: int):
        return self.predict_step(batch, batch_idx)

    def validation_step_end(self, val_step_outputs: STEP_OUTPUT):
        self._outputs_to_cpu(val_step_outputs)
        self._post_process(val_step_outputs)
        return val_step_outputs

    def test_step_end(self, test_step_outputs: STEP_OUTPUT):
        self._outputs_to_cpu(test_step_outputs)
        self._post_process(test_step_outputs)
        return test_step_outputs

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if self.threshold_method == 'adaptive':
            self._compute_adaptive_threshold(outputs)
        self._collect_outputs(self.image_metrics, self.pixel_metrics, outputs)
        self._log_metrics()

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._collect_outputs(self.image_metrics, self.pixel_metrics, outputs)
        self._log_metrics()

    def _compute_adaptive_threshold(self, outputs: EPOCH_OUTPUT) -> None:
        self.image_threshold.reset()
        self.pixel_threshold.reset()
        self._collect_outputs(self.image_threshold, self.pixel_threshold, outputs)
        self.image_threshold.compute()
        if "mask" in outputs[0].keys() and "anomaly_maps" in outputs[0].keys():
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value

        self.image_metrics.set_threshold(self.image_threshold.value.item())
        self.pixel_metrics.set_threshold(self.pixel_threshold.value.item())

    @staticmethod
    def _collect_outputs(
        image_metric: AnomalibMetricCollection,
        pixel_metric: AnomalibMetricCollection,
        outputs: EPOCH_OUTPUT,
    ) -> None:
        for output in outputs:
            image_metric.cpu()
            image_metric.update(output["pred_scores"], output["label"].int())
            if "mask" in output.keys() and "anomaly_maps" in output.keys():
                # CUSTOM - added ...
                if output["anomaly_maps"].shape != output["mask"].shape:
                    # print(">>>>>>>>>>>>>>>>>>>>>> Anomaly map shape: {} {}".format(output["anomaly_maps"].shape, list(output["anomaly_maps"].shape)[2:])) # returns: torch.Size([1, 1, 542, 775]) [542, 775]
                    # print(">>>>>>>>>>>>>>>>>>>>>> Anomaly map shape: {}".format(output["mask"].shape))  # returns: torch.Size([1, 2168, 3100])
                    output["mask"] = (F.interpolate(output["mask"].unsqueeze(0), size=tuple(list(output["anomaly_maps"].shape)[2:]), mode='nearest')).squeeze(0)
                    # print(">>>>>>>>>>>>>>>>>>>>>> Anomaly map shape: {}".format(output["mask"].shape))
                # CUSTOM - end
                pixel_metric.cpu()
                pixel_metric.update(output["anomaly_maps"], output["mask"].int())

    @staticmethod
    def _post_process(outputs: STEP_OUTPUT) -> None:
        """Compute labels based on model predictions."""
        if isinstance(outputs, dict):
            if "pred_scores" not in outputs and "anomaly_maps" in outputs:
                # infer image scores from anomaly maps
                outputs["pred_scores"] = (
                    outputs["anomaly_maps"].reshape(outputs["anomaly_maps"].shape[0], -1).max(dim=1).values
                )
            elif "pred_scores" not in outputs and "box_scores" in outputs:
                # infer image score from bbox confidence scores
                outputs["pred_scores"] = torch.zeros_like(outputs["label"]).float()
                for idx, (boxes, scores) in enumerate(zip(outputs["pred_boxes"], outputs["box_scores"])):
                    if boxes.numel():
                        outputs["pred_scores"][idx] = scores.max().item()

            if "pred_boxes" in outputs and "anomaly_maps" not in outputs:
                # create anomaly maps from bbox predictions for thresholding and evaluation
                image_size: tuple[int, int] = outputs["image"].shape[-2:]
                true_boxes: list[Tensor] = outputs["boxes"]
                pred_boxes: Tensor = outputs["pred_boxes"]
                box_scores: Tensor = outputs["box_scores"]

                outputs["anomaly_maps"] = boxes_to_anomaly_maps(pred_boxes, box_scores, image_size)
                outputs["mask"] = boxes_to_masks(true_boxes, image_size)

    def _outputs_to_cpu(self, output):
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = self._outputs_to_cpu(value)
        elif isinstance(output, list):
            output = [self._outputs_to_cpu(item) for item in output]
        elif isinstance(output, Tensor):
            output = output.cpu()
        return output

    def _log_metrics(self) -> None:
        """Log computed performance metrics."""
        if self.pixel_metrics.update_called:
            self.log_dict(self.pixel_metrics, prog_bar=True)
            self.log_dict(self.image_metrics, prog_bar=False)
        else:
            self.log_dict(self.image_metrics, prog_bar=True)

    def _load_normalization_class(self, state_dict: OrderedDict[str, Tensor]) -> None:
        """Assigns the normalization method to use."""
        if "normalization_metrics.max" in state_dict.keys():
            self.normalization_metrics = MinMax()
        elif "normalization_metrics.image_mean" in state_dict.keys():
            self.normalization_metrics = AnomalyScoreDistribution()
        else:
            warn("No known normalization found in model weights.")

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        """Load state dict from checkpoint.

        Ensures that normalization and thresholding attributes is properly setup before model is loaded.
        """
        # Used to load missing normalization and threshold parameters
        self._load_normalization_class(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

