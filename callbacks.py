import logging
import os
from pathlib import Path
import warnings
from warnings import warn
from importlib import import_module

import cv2
from omegaconf import DictConfig, ListConfig, OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

from images import visualize_predict_batch
from anomaly_normalisation import MinMax

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

                filename = Path(outputs["image_path"][i])

                file_path = Path(self.image_save_path, filename.name)
                # print("Saving image to: {}".format(str(file_path)))
                self.save_image(file_path, heatmap)
                # file_path = self.image_save_path / filename.parent.name / filename.name

                # if self.show_images:
                #     self.visualizer.show(str(filename), heatmap)

    def save_image(self, save_path, image):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), image)
