from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import os
import torch
from pytorch_lightning import Trainer, Callback
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .dataset import InferenceDataset
from .transforms import get_transforms

from anomalib.config import get_configurable_parameters
from anomalib.data.utils import InputNormalizationMethod, get_image_filenames
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

#logger = logging.getLogger(__name__)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to image(s) to infer.")
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument("--visualization_mode", type=str, required=False, default="simple", help="Visualization mode.", choices=["full", "simple"],)
    parser.add_argument("--show", action="store_true", required=False, help="Show the visualized predictions on the screen.")
    args = parser.parse_args()
    return args


class LoadModelCallback(Callback):
    """Callback that loads the model weights from the state dict."""

    def __init__(self, weights_path) -> None:
        self.weights_path = weights_path

    def setup(self, trainer: Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        """Call when inference begins.

        Loads the model weights from ``weights_path`` into the PyTorch module.
        """
        del trainer, stage  # These variables are not used.

        logger.info("Loading the model from %s", self.weights_path)
        pl_module.load_state_dict(torch.load(self.weights_path, map_location=pl_module.device)["state_dict"])
        
        
def infer():
    args = get_args()
    config = get_configurable_parameters(config_path=args.config)
    config.trainer.resume_from_checkpoint = str(args.weights)
    config.visualization.show_images = args.show
    config.visualization.mode = args.visualization_mode
    if args.output:  # overwrite save path
        if not os.path.exists(args.output): os.makedirs(args.output)  # CUSTOM - added ...
        config.visualization.save_images = True
        config.visualization.image_save_path = args.output
    else:
        config.visualization.save_images = False
    tiling = config.dataset.tiling.apply

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # get the transforms
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    # print('>>>>>>>>>>>>>>> Image size: {}'.format(image_size))  # result is: (256, 256)
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(config=transform_config,
                               image_size=image_size,
                               center_crop=center_crop,
                               normalization=normalization,
                               tiling=tiling,
                               infer=True,
                               )

    image_filenames = np.array(sorted(get_image_filenames(args.input)))
    print('>>>>>>>>>>>>>>> Total number of images is: {}'.format(len(image_filenames.tolist())))  # result is:
    num_batches = 10  # split the entire dataset by this number and take in separately for batching and loading as normal
    for files in np.array_split(image_filenames, num_batches):
        print('>>>>>>>>>>>>>>> Number of images next: {}'.format(len(files.tolist())))
        dataset = InferenceDataset(files.tolist(), image_size=tuple(config.dataset.image_size), transform=transform)
        dataloader = DataLoader(dataset, num_workers=4)  # CUSTOM - added num_workers
        with torch.no_grad():
            trainer.predict(model=model, dataloaders=[dataloader])


if __name__ == "__main__":
    infer()

