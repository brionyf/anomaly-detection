import os
import logging
import warnings
from argparse import ArgumentParser, Namespace
from importlib import import_module
from omegaconf import DictConfig, ListConfig

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import Logger

from dataset import CustomDataModule, get_configurable_parameters

#from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
#from anomalib.utils.loggers import configure_logger, get_experiment_logger

from models.model_1 import get_callbacks, Model1

__all__ = [
    "Model1",
]

logger = logging.getLogger("train.py")


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    #parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")
    parser.add_argument("--test", type=str, default="True", help="Run model on test set? [DEFAULT: True]")
    args = parser.parse_args()
    return args
    

def configure_logger(level: int | str = logging.INFO):

    if isinstance(level, str):
        level = logging.getLevelName(level)

    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string, level=level)

    # Set Pytorch Lightning logs to have a the consistent formatting with anomalib.
    for handler in logging.getLogger("pytorch_lightning").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)
        
    #return logger.addHandler(handler)
        
        
def get_model(config: DictConfig | ListConfig):

    logger.info("Loading the model.")

    model_list = ["model_1"]

    if config.model.name in model_list:
        module = import_module(f"models.{config.model.name}")
        model_file = "".join([split.capitalize() for split in config.model.name.split("_")])
        model = getattr(module, f"{model_file}")(config)

    else:
        raise ValueError(f"Unknown model {config.model.name}!")

    #if "init_weights" in config.keys() and config.init_weights:
    #    model.load_state_dict(load(os.path.join(config.project.path, config.init_weights))["state_dict"], strict=False)

    return model
    
    
def train():

    args = get_args()
    args.test = True if args.test == 'True' else False
    configure_logger(level=args.log_level)

    warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model) #, config_path=args.config)
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)
    print("Project path: {}".format(config.project.path))

    datamodule = CustomDataModule(
            root=config.dataset.path,
            category=config.dataset.category,
            #image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            image_size=(config.dataset.image_size, config.dataset.image_size),
            center_crop=config.dataset.center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
            #tiling=tiling,
        )

    model = get_model(config)
    #experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks) #, logger=experiment_logger
    logger.info("Training the model...")
    trainer.fit(model=model, datamodule=datamodule)

    #logger.info("Loading the best model weights...")
    #load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    #trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

    #if config.dataset.test_split_mode == None:
    #    logger.info("No test set provided. Skipping test stage.")
    #else:
    #    if bool(args.test):
    #        print("Testing the model...")
    #        logger.info("Testing the model...")
    #        trainer.test(model=model, datamodule=datamodule)
    #    # CUSTOM - just train, don't test
    #    else:
    #        print("Opted out of testing. Skipping test stage.")
    #        logger.info("Opted out of testing. Skipping test stage.")
    #        # CUSTOM - end


if __name__ == "__main__":
    train()
