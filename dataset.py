import os
import numpy as np
import cv2
import math
from skimage.filters.rank import median
#import torchvision.transforms.functional as TF
from omegaconf import DictConfig, ListConfig, OmegaConf
#import time
from datetime import datetime
import copy

import logging
from pathlib import Path
from typing import Sequence

import albumentations as A
from pandas import DataFrame
from argparse import ArgumentParser, Namespace
from abc import ABC

from torch import Tensor
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.dataset import Dataset, random_split

from transforms import get_transforms
from images import read_image, get_image_filenames

logger = logging.getLogger(__name__)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of model to train/evaluate")
    return parser.parse_args()
    
def collate_fn(batch: list):
    return default_collate(batch)

def get_dataframe(root: str | Path, split: str | None = None) -> DataFrame:

    root = Path(root)
    samples_list = [(str(root),) + f.parts[-3:] for f in root.glob(r"**/*") if f.suffix in ['.png']]
    if not samples_list:
        raise RuntimeError(f"Found 0 images in {root}")

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = 0
    samples.loc[(samples.label != "good"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    # separate masks from samples
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

    # assign mask paths to anomalous test images
    samples["mask_path"] = ""
    samples.loc[(samples.split == "test") & (samples.label_index == 1), "mask_path"] = mask_samples.image_path.values

    # assert that the right mask files are associated with the right test images
    assert (
        samples.loc[samples.label_index == 1]
        .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
        .all()
    ), "Filenames of anomalous images don't match ground truth masks. \
              (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    #print("df_samples information: \tNumber of image paths: {} \t Columns: {}".format(len(samples), samples.columns))
    return samples

class CustomDataset(Dataset, ABC):

    def __init__(self, task: str, transform: A.Compose, root: Path | str, category: str, split: str | None = None,) -> None:
        super().__init__() #task=task, transform=transform)
        self.root_category = Path(root) / Path(category)
        self.split = split # train or test
        self.task = task # 'classification', 'detection' or 'segmentation'
        self.transform = transform
        self.df_images = get_dataframe(self.root_category, split=self.split)

    def __len__(self) -> int:
        return len(self.df_images)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        image_path = self.df_images.iloc[index].image_path
        mask_path = self.df_images.iloc[index].mask_path
        label_index = self.df_images.iloc[index].label_index

        image = read_image(image_path)
        item = dict(image_path=image_path, label=label_index)

        if self.task == 'classification':
            transformed = self.transform(image=image)
            item["image"] = transformed["image"]
        elif self.task in ('detection', 'segmentation'):
            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.
            if label_index == 0:
                mask = np.zeros(shape=image.shape[:2])
            else:
                mask = cv2.imread(mask_path, flags=0) / 255.0

            transformed = self.transform(image=image, mask=mask)

            item["image"] = transformed["image"]
            item["mask_path"] = mask_path
            item["mask"] = transformed["mask"]

            if self.task == 'detection':
                # create boxes from masks for detection task
                boxes, _ = masks_to_boxes(item["mask"])
                item["boxes"] = boxes[0]
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        return item


class CustomDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        root: Path | str,
        category: str,
        image_size: int | tuple[int, int] | None = None,
        center_crop: int | tuple[int, int] | None = None,
        normalization: str = 'imagenet',
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: str = 'segmentation',
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: str = 'from_dir',
        test_split_ratio: float = 0.2,
        val_split_mode: str = 'same_as_test',
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.category = Path(category)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.test_split_mode = test_split_mode
        self.test_split_ratio = test_split_ratio
        self.val_split_mode = val_split_mode
        self.val_split_ratio = val_split_ratio
        self.seed = seed

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=normalization,
            training=True,
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=normalization,
        )

        self.train_data = CustomDataset(task=task, transform=transform_train, split='train', root=root, category=category)
        self.test_data = CustomDataset(task=task, transform=transform_eval, split='test', root=root, category=category)
        assert self.train_data is not None
        assert self.test_data is not None
        
        self._samples: DataFrame | None = None

    def setup(self, stage: str): # TODO: double check original setup function that was deleted
	# This method is used to define the process that is meant to be performed by all the available GPU. 
	# It’s usually used to handle the task of loading the data. 
        if self.test_split_ratio is not None:
            self.train_data, normal_test_data = random_split(self.train_data, [1-self.test_split_ratio, self.test_split_ratio])#, seed=self.seed)
            self.test_data += normal_test_data

        if self.val_split_mode == 'from_test': # randomly sampled from test set
            self.test_data, self.val_data = random_split(self.test_data, self.val_split_ratio, label_aware=True, seed=self.seed)
        elif self.val_split_mode == 'same_as_test': # same as test set
            self.val_data = self.test_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data, 
            shuffle=True, 
            batch_size=self.train_batch_size, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn, # TODO: add collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn, # TODO: add collate_fn
        )

#####################################################################################################################

class InferenceDataset(Dataset):
    def __init__(
        self,
        files,
        transform: A.Compose | None = None,
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()

        self.image_filenames = files

        if transform is None:
            self.transform = get_transforms(image_size=image_size)
        else:
            self.transform = transform

        self.old_min = 0  # CUSTOM - added ...

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int):
        """Get the image based on the `index`."""
        image_filename = self.image_filenames[index]
        image = read_image(path=image_filename)
        # new_x_min = self.auto_crop_image(image)  # OPTIONS: 800, 1100, self.auto_crop_image(image)
        # image = image[:, new_x_min:3900, :]
        # if index == 0: print(new_x_min)
        image = self._auto_crop_image(image)
        # print('>>>>>>>>>>>>>>> Image shape: {}'.format(image.shape))  # result is: (2168, 3100, 3)
        pre_processed = self.transform(image=image)
        # print('>>>>>>>>>>>>>>> {} {} {}'.format(image.shape, pre_processed["image"].shape, pre_processed["image"].dtype))  # result is: torch.Size([3, 256, 256]) torch.float32
        pre_processed["image_path"] = str(image_filename)
        pre_processed["original_image"] = image

        return pre_processed

    def _auto_crop_image(self, img):  # TODO: some values are hardcoded, so if images are different sizes, will result in wrong crops

        resize_ratio = 0.2
        if img.shape[:2] != (2168, 4096):
            img = cv2.resize(img, (4096, 2168))
        # print('>>>>>>>>>>>>>>> Image shape: {}'.format(img.shape))  # result is: (2168, 3100, 3)
        img_resized = cv2.resize(img, (int(img.shape[1]*resize_ratio), int(img.shape[0]*resize_ratio)))
        # print('>>>>>>>>>>>>>>> Image shape: {}'.format(img_resized.shape))  # result is: (2168, 3100, 3)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        blur_k = 9  # kernel size
        edge_k = 5  # kernel size
        footprint_size = 100  # img_gray.shape[0]
        footprint = np.ones((footprint_size, 1), np.uint8)
        img_blur = cv2.GaussianBlur(img_gray, (blur_k, blur_k), sigmaX=0, sigmaY=0)
        # edges = cv2.Canny(img, 100, 150)
        edges = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=edge_k)
        edges_norm = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        edges = median(edges_norm, footprint)
        edges = cv2.Canny(edges, 50, 200)

        x_max = 3900  # TODO: hardcoded
        base_crop = 100  # TODO: hardcoded
        padding = 25  # TODO: hardcoded
        delta = 20  # TODO: hardcoded
        pts = np.argwhere(edges[:, base_crop:] > 0)
        try:
            _, x = pts.min(axis=0)
            if abs(x - self.old_min) > delta: x = self.old_min  # only use x if not "too" different from old_min, else use old_min
            self.old_min = x
        except ValueError:  # raised if `min()` is empty
            x = self.old_min
        x_resized = int((x+base_crop+padding) / resize_ratio)
        x_resized = int(math.ceil(x_resized / 100.0)) * 100  # ADDED to fix problem of weird, regular, rectangular detection in bottom of images
        # print(x_resized)

        return img[:, x_resized:x_max, :] #x_resized

#####################################################################################################################
# The below is just for testing
#####################################################################################################################

def get_configurable_parameters(
    model_name: str | None = None,
    config_path: Path | str | None = None,
    weight_file: str | None = None,
    #config_filename: str | None = "config",
    #config_file_extension: str | None = "yaml",
) -> DictConfig | ListConfig:

    config_path = Path(os.getcwd(), 'models', model_name, 'config.yaml')
    config = OmegaConf.load(config_path)

    # keep track of the original config file because it will be modified
    config_original: DictConfig = config.copy()

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
    
if __name__ == "__main__":
    args = get_args()
    config = get_configurable_parameters(model_name=args.model) #, config_path=args.config)
    print("Project path: {}".format(config.project.path))

    datamodule = CustomDataModule(
            root=config.dataset.path,
            category=config.dataset.category,
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
        )

    dataloader = datamodule.train_dataloader()

    dataiter = iter(dataloader)

    for i in range(5):
        img_batch = next(dataiter) # extract batch
        print(img_batch.keys())
        print(img_batch['image'].shape)
        print([img_name[-10:] for img_name in img_batch['image_path']])

# Command Line:
# conda deactivate
# conda activate anomalib_env2
# cd '/home/brionyf/Documents/GitHub/anomaly-detection'
# python dataset.py --model 'model_1'
