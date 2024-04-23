import logging
import os
import warnings
from importlib import import_module

import yaml
from jsonargparse.namespace import Namespace
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

from .model_1 import Model1
from .de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from .resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from .loss import loss_function, loss_concat

__all__ = ["Model1", "de_resnet18", "de_resnet34", "de_wide_resnet50_2", "de_resnet50", "resnet18", "resnet34", "resnet50", "wide_resnet50_2", "loss_function", "loss_concat"]

logger = logging.getLogger(__name__)

def get_callbacks(config: DictConfig | ListConfig) -> list[Callback]:

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

    callbacks.extend([early_stopping, model_checkpoint]) #, TimerCallback()])

    return callbacks
