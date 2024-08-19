import os
import numpy as np
import math
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from dataset_inference import InferenceDataset
from images import get_image_filenames
from train import get_model, configure_logger
from callbacks import get_callbacks, LoadModelCallback, SaveImageCallback
from anomaly_normalisation import NormalisationCallback

from warnings import warn
import logging
logger = logging.getLogger("inference.py")

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to image(s) to infer.")
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")
    return parser.parse_args()


def get_config_params(
    model_name: str | None = None,
    config_path: Path | str | None = None,
    weight_file: str | None = None,
    #config_filename: str | None = "config",
    #config_file_extension: str | None = "yaml",
    infer = False,
) -> DictConfig | ListConfig:

    if config_path is None:
        config_path = Path(os.getcwd(), '../models', model_name, 'config.yaml')
    else:
        config_path = Path(os.getcwd() + config_path)
    config = OmegaConf.load(config_path)

    # keep track of the original config file because it will be modified
    config_original = config.copy()

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

    # Project Configs
    if not infer:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        project_path = Path(config.project.path) / config.model.name / config.dataset.category
        project_path = project_path / current_datetime
        (project_path / "weights").mkdir(parents=True, exist_ok=True)
        (project_path / "images").mkdir(parents=True, exist_ok=True)
        # write the original config for eventual debug (modified config at the end of the function)
        (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_original))

        config.project.path = str(project_path)

        # loggers should write to results/model/dataset/category/ folder
        config.trainer.default_root_dir = str(project_path)

    if weight_file:
        config.trainer.resume_from_checkpoint = weight_file

    #(project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))

    return config


def infer():
    args = get_args()
    configure_logger(level=args.log_level)
    config = get_config_params(config_path=args.config, infer=True)
    config.trainer.resume_from_checkpoint = os.getcwd() + str(args.weights)
    # # config.visualization.show_images = args.show
    # # config.visualization.mode = args.visualization_mode
    # if args.output:  # overwrite save path
    #     if not os.path.exists(args.output): os.makedirs(args.output)  # CUSTOM - added ...
    #     config.visualization.save_images = True
    #     config.visualization.image_save_path = args.output
    # else:
    #     config.visualization.save_images = False

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    callbacks.append(LoadModelCallback(config.trainer.resume_from_checkpoint))
    callbacks.append(NormalisationCallback(config.model.normalization_method))
    config.visualization.save_images = True
    if config.visualization.save_images:
        config.visualization.image_save_path = os.path.join(args.input, 'no-detections') #args.output
        Path(config.visualization.image_save_path).mkdir(parents=True, exist_ok=True)
        # print(">>>>>>>>>>>> Created save folder: {}".format(config.visualization.image_save_path))
        callbacks.append(SaveImageCallback(config.visualization.image_save_path))
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # # get the transforms
    # # image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    # # tiling = config.dataset.tiling.apply
    # # print('>>>>>>>>>>>>>>> Image size: {}'.format(image_size))  # result is: (256, 256)
    # transform = get_transforms(image_size=config.dataset.image_size,
    #                            resize_ratio=config.dataset.resize_ratio,
    #                            normalization=config.dataset.normalization,
    #                            tiling=config.dataset.tiling.apply,
    #                            infer=True,
    #                            )

    image_filenames = np.array(sorted(get_image_filenames(args.input)))
    print('>>>>>>>>>>>>>>> Total number of images is: {}'.format(len(image_filenames.tolist())))  # result is:
    # num_batches = 10  # split the entire dataset by this number and take in separately for batching and loading as normal
    # for files in np.array_split(image_filenames, num_batches):
    chunk_size = 100
    for files in np.array_split(image_filenames, math.ceil(len(image_filenames)/chunk_size)):
        print('>>>>>>>>>>>>>>> Number of images next: {}'.format(len(files.tolist())))
        # dataset = InferenceDataset(files.tolist(), image_size=tuple(config.dataset.image_size), transform=transform)
        dataset = InferenceDataset(files.tolist(), config.dataset.grayscale, config.dataset.resize_ratio) #, transform=transform)
        dataloader = DataLoader(dataset, num_workers=4)  # CUSTOM - added num_workers
        with torch.no_grad():
            trainer.predict(model=model, dataloaders=[dataloader])
    # except KeyboardInterrupt:
    #     print(">>>>>>> program execution cancelled")
    #     break


if __name__ == "__main__":
    infer()

# Command to perform inference:
# (anomalib_env2) brionyf@brionyf-Precision-3650-Tower:~/Documents/GitHub/anomaly-detection$ python inference.py --config /models/model_1/config.yaml --weights /results/model_1/carpet/2024-05-02_15-18-47/weights/model.ckpt --input '/media/brionyf/T7/AKL Finishing Line/Images - Basler/2024-04-24 12-24-08' --output '/media/brionyf/T7/AKL Finishing Line/Images - Basler/2024-04-24 12-24-08/detections'
