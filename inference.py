import os
import numpy as np

import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer, Callback
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transforms import get_transforms
from dataset import InferenceDataset, get_config_params
from images import get_image_filenames
from train import get_model, configure_logger
from callbacks import get_callbacks, LoadModelCallback, SaveImageCallback
from anomaly_normalisation import NormalisationCallback

logger = logging.getLogger("inference.py")

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to image(s) to infer.")
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")
    return parser.parse_args()
        
def infer():
    args = get_args()
    configure_logger(level=args.log_level)
    config = get_config_params(config_path=args.config, infer=True)
    config.trainer.resume_from_checkpoint = os.getcwd() + str(args.weights)
    # config.visualization.show_images = args.show
    # config.visualization.mode = args.visualization_mode
    if args.output:  # overwrite save path
        if not os.path.exists(args.output): os.makedirs(args.output)  # CUSTOM - added ...
        config.visualization.save_images = True
        config.visualization.image_save_path = args.output
    else:
        config.visualization.save_images = False

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    callbacks.append(LoadModelCallback(config.trainer.resume_from_checkpoint))
    callbacks.append(NormalisationCallback(config.model.normalization_method))
    if config.visualization.save_images:
        callbacks.append(SaveImageCallback(config.visualization.image_save_path))
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # get the transforms
    # image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    # tiling = config.dataset.tiling.apply
    # print('>>>>>>>>>>>>>>> Image size: {}'.format(image_size))  # result is: (256, 256)
    transform = get_transforms(image_size=config.dataset.image_size,
                               normalization=config.dataset.normalization,
                               tiling=config.dataset.tiling.apply,
                               infer=True,
                               )

    image_filenames = np.array(sorted(get_image_filenames(args.input)))
    print('>>>>>>>>>>>>>>> Total number of images is: {}'.format(len(image_filenames.tolist())))  # result is:
    num_batches = 10  # split the entire dataset by this number and take in separately for batching and loading as normal
    for files in np.array_split(image_filenames, num_batches):
        print('>>>>>>>>>>>>>>> Number of images next: {}'.format(len(files.tolist())))
        # dataset = InferenceDataset(files.tolist(), image_size=tuple(config.dataset.image_size), transform=transform)
        dataset = InferenceDataset(files.tolist(), transform=transform)
        dataloader = DataLoader(dataset, num_workers=4)  # CUSTOM - added num_workers
        with torch.no_grad():
            trainer.predict(model=model, dataloaders=[dataloader])


if __name__ == "__main__":
    infer()

# Command to perform inference:
# (anomalib_env2) brionyf@brionyf-Precision-3650-Tower:~/Documents/GitHub/anomaly-detection$ python inference.py --config /models/model_1/config.yaml --weights /results/model_1/carpet/2024-05-02_15-18-47/weights/model.ckpt --input '/media/brionyf/T7/AKL Finishing Line/Images - Basler/2024-04-24 12-24-08' --output '/media/brionyf/T7/AKL Finishing Line/Images - Basler/2024-04-24 12-24-08/detections'
