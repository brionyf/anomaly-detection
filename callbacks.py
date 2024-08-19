import logging
import os
from pathlib import Path
import warnings
from warnings import warn
from importlib import import_module

import math
import cv2
from omegaconf import DictConfig, ListConfig, OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

from images import visualize_predict_batch, visualize_predict_batch2
from anomaly_normalisation import MinMax
# from visualiser import Visualiser

logger = logging.getLogger(__name__)

def get_callbacks(config):

    logger.info("Loading the callbacks...")

    callbacks: list[Callback] = []

    monitor_metric = None if "early_stopping" not in config.model.keys() else config.model.early_stopping.metric
    monitor_mode = "max" if "early_stopping" not in config.model.keys() else config.model.early_stopping.mode
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=config.model.early_stopping.patience,
        mode=monitor_mode,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(config.project.path, "weights"),
        filename="model",
        monitor=monitor_metric,
        mode=monitor_mode,
        auto_insert_metric_name=False,
    )

    callbacks.extend([early_stopping, model_checkpoint])

    return callbacks

class LoadModelCallback(Callback):
    """Callback that loads the model weights from the state dict."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def setup(self, trainer, pl_module, stage): # Called when inference begins
        del trainer, stage  # Unused variables
        logger.info("Loading the model from %s...", self.weights_path)
        state_dict = torch.load(self.weights_path, map_location='cpu')["state_dict"]
        if "normalization_metrics.max" in state_dict.keys():
            pl_module.normalization_metrics = MinMax()
        elif "normalization_metrics.image_mean" in state_dict.keys():
            # pl_module.normalization_metrics = AnomalyScoreDistribution()
            pass
        else:
            warn("No known normalization found in model weights...")

        pl_module.load_state_dict(state_dict)

class SaveImageCallback(Callback):
    def __init__(self, save_path):
        self.image_save_path = save_path

    def on_predict_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx):
        del trainer, pl_module, batch, batch_idx, dataloader_idx  # unused variables

        assert outputs is not None
        # print('>>>>>>>>>>>>>>> Save Callback - outputs: {} {}'.format(len(outputs), outputs["pred_masks"].shape))  # result is:
        for i, (image, heatmap) in enumerate(visualize_predict_batch(outputs)):
        # for i, (image, heatmap) in enumerate(visualize_predict_batch2(outputs)):

            filename = Path(outputs["image_path"][i])

            file_path = Path(self.image_save_path, filename.name)
            # print("Saving image to: {}".format(str(file_path)))
            self.save_image(file_path, heatmap)
            # file_path = self.image_save_path / filename.parent.name / filename.name

    def on_test_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx):
        del trainer, pl_module, batch, batch_idx, dataloader_idx  # unused variables

        assert outputs is not None
        # print('>>>>>>>>>>>>>>> Save Callback - outputs: {} {}'.format(len(outputs), outputs["pred_masks"].shape))  # result is:
        for i, (image, heatmap) in enumerate(visualize_predict_batch(outputs)):
        # for i, (image, heatmap) in enumerate(visualize_predict_batch2(outputs)):

            filename = Path(outputs["image_path"][i])

            # file_path = Path(self.image_save_path, filename.name)
            file_path = Path(self.image_save_path, filename.parent.name, filename.name)
            # print("Saving image to: {}".format(str(file_path)))
            self.save_image(file_path, heatmap)
            # file_path = self.image_save_path / filename.parent.name / filename.name

    def save_image(self, save_path, image):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # print('>>>>>>>>>>>>>>> Save Callback - image shape: {}'.format(image.shape))  # result is:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), image)

# class OutputVisualiserCallback(Callback):
#     def __init__(
#         self,
#         image_save_path: str,
#         inputs_are_normalized: bool = True,
#         save_images: bool = True,
#     ) -> None:
#
#         self.inputs_are_normalized = inputs_are_normalized
#         self.save_images = save_images
#         self.image_save_path = Path(image_save_path)
#         self.visualizer = Visualiser()
#
#     def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
#
#         del batch, batch_idx, dataloader_idx  # These variables are not used.
#         assert outputs is not None
#         # print('>>>>>>>>>>>>>>> Outputs: {} {}'.format(len(outputs), outputs["image"][0].shape))  # result is: 12 torch.Size([1, 3, 2168, 3100])
#         # print('>>>>>>>>>>>>>>> Filename: {} '.format(outputs["image_path"][0]))  # result is: files in test/bad folder and then train/good folder until 18
#
#         if self.save_images:
#
#             for i, (image, heatmap) in enumerate(self.visualizer.visualize_batch(outputs)):
#                 if "image_path" in outputs.keys():
#                     filename = Path(outputs["image_path"][i])
#                     # print('>>>>>>>>>>>>>>> original filename: {}'.format(str(filename)))  # result is:
#                 else:
#                     raise KeyError("Batch must have 'image_path' defined.")
#
#                 # print('>>>>>>>>>>>>>>> {} {}'.format(image.shape, image.dtype))  # result is: 'tuple' object has no attribute 'shape'
#                 # print('>>>>>>>>>>>>>>> Saved File: {}'.format(filename.name))
#                 file_path = self.image_save_path / filename.parent.name / filename.name
#                 # print('>>>>>>>>>>>>>>> Save file path: {}'.format(str(file_path)))
#                 self.visualizer.save(file_path, heatmap)

