import os
import logging
import warnings
from argparse import ArgumentParser, Namespace
from importlib import import_module
from omegaconf import DictConfig, ListConfig

from pytorch_lightning import Trainer, seed_everything
#from pytorch_lightning.loggers import Logger

# from dataset import CustomDataModule
from dataset_sampler import SampleDataModule, get_config_params
from anomaly_normalisation import NormalisationCallback

from models.model_1 import Model1
from models.model_2 import Model2
from callbacks import get_callbacks, LoadModelCallback, SaveImageCallback #, OutputVisualiserCallback
__all__ = ["Model1", "Model2", "get_model", "configure_logger"]

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
    # Set Pytorch Lightning logs to have consistent formatting with anomalib.
    for handler in logging.getLogger("pytorch_lightning").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)
    #return logger.addHandler(handler)

def get_model(config: DictConfig | ListConfig):
    logger.info("Loading the model...")
    model_list = ["model_1", "model_2"]
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

    config = get_config_params(model_name=args.model) #, config_path=args.config)
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)
    print("Project path: {}".format(config.project.path))

    datamodule = SampleDataModule( #CustomDataModule(
            root=config.dataset.path,
            category=config.dataset.category,
            #image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            image_size=(config.dataset.image_size, config.dataset.image_size),
            resize_ratio=config.dataset.resize_ratio,
            center_crop=config.dataset.center_crop,
            normalization=config.dataset.normalization,
            # grayscale=config.dataset.grayscale,
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
    callbacks.append(NormalisationCallback(config.model.normalization_method))

    trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks) #, logger=experiment_logger
    logger.info("Training the model...")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Training complete :)")

    if config.dataset.test_split_mode == None:
        logger.info("No test set provided. Skipping test stage.")
    else:
        if bool(args.test):
            logger.info("Loading the best model weights for testing...")
            load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
            trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member
            # trainer.callbacks.insert(-1, OutputVisualiserCallback(os.path.join(config.project.path, "images")))
            trainer.callbacks.insert(-1, SaveImageCallback(os.path.join(config.project.path, "images")))
            # print(trainer.callbacks)
            # print("Testing the model...")
            logger.info("Testing the model...")
            trainer.test(model=model, datamodule=datamodule)
        # CUSTOM - just train, don't test
        else:
            # print("Opted out of testing. Skipping test stage.")
            logger.info("Opted out of testing. Skipping test stage.")
            # CUSTOM - end

if __name__ == "__main__":
    train()

# Command to train:
# (anomalib_env2) brionyf@brionyf-Precision-3650-Tower:~/Documents/GitHub/anomaly-detection$ python train.py --model model_1 --test True



# def train (): #from AE code (model_2)
#     current_folder = os.getcwd()
#     train_patches = load_patches(os.path.join('Dataset',dataset,category,'Normal'), patch_size=patch_size, n_patches=n_patches, random=True, preprocess_limit=0, resize=None)
#     x_train = preprocess_data(train_patches)
#     print (np.shape(x_train))
#
#     #for x in x_train:
#         #plt.imshow(np.squeeze(x))
#         #plt.show()
#
#     tf.keras.backend.set_floatx('float64')
#
#     loss_function = None
#     for loss in [cwssim_loss, ssim_loss, ms_ssim_loss, l2_loss]:
#         if (loss.__name__ == loss_type):
#             loss_function = loss
#
#     callbacks = []
#     callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
#     callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('Weights','new_weights','check_epoch{epoch:02d}.h5'), save_weights_only=True, period=save_period))
#
#     autoencoder = Model_noise_skip(input_shape=(patch_size,patch_size,1))
#     autoencoder.summary()
#     #autoencoder.load_weights('Weights\\new_weights\\check_epoch95.h5')
#
#     autoencoder.compile(optimizer='adam', loss=loss_function)
#
#     #autoencoder.fit(x_train, x_train, epochs=epoch, shuffle=True, batch_size=batch_size, callbacks=callbacks, initial_epoch=95)
#     autoencoder.fit(x_train, x_train, epochs=epoch, shuffle=True, batch_size=batch_size, callbacks=callbacks)
