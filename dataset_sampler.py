import os
import numpy as np
import cv2
import math
import random
from skimage.filters.rank import median
from sklearn.model_selection import train_test_split
from collections import defaultdict
#import torchvision.transforms.functional as TF
from omegaconf import DictConfig, ListConfig, OmegaConf
#import time
from datetime import datetime
import copy

import warnings
from warnings import warn
import logging
from pathlib import Path
from typing import Sequence

# import albumentations as A
import pandas as pd
from pandas import DataFrame
from argparse import ArgumentParser, Namespace
from abc import ABC

import torch
from torch import Tensor
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.dataset import Dataset, random_split

import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image

# from transforms import get_transforms
from images import read_image
# from dataset import collate_fn, get_dataframe, get_config_params

logger = logging.getLogger(__name__)

class FewShotBatchSampler:
    def __init__(self, dataset_targets, K_shot, N_way=2, include_query=False, shuffle=True, shuffle_once=False):
        super().__init__()
        self.dataset_targets = torch.Tensor(dataset_targets).type(torch.int64) # tensor of the data labels (0 - good, 1 - bad)
        # print(">>>>>>>>>>>>>>>> targets = {}".format(self.dataset_targets[:5]))
        self.N_way = N_way # Number of classes to sample per batch
        self.K_shot = K_shot # Number of examples to sample per class in the batch
        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch
        self.shuffle = shuffle # data is shuffled each iteration (during training)

        self.classes = torch.unique(self.dataset_targets).tolist()
        # print('Classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_shot
        # self.iterations = sum(self.batches_per_class.values()) // self.N_way
        self.iterations = min(self.batches_per_class.values())
        print(">>>>>>>>>>>>>>>> classes: {} \tbatch size: {} \tbatches per Class 0: {} \tbatches per Class 1: {} \titerations: {}".format(self.classes, self.batch_size, self.batches_per_class[0], self.batches_per_class[1], self.iterations))
        print(">>>>>>>>>>>>>>>> indices per Class 0: {} \tindices per Class 1: {}".format(len(self.indices_per_class[0]), len(self.indices_per_class[1])))

        if shuffle_once or self.shuffle: self.shuffle_data() # shuffle_once - shuffled once in the beginning, but kept constant across iterations (for validation)
        # self.include_query = include_query
        # if self.include_query: self.K_shot *= 2

    def shuffle_data(self):
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]

    def __iter__(self):
        if self.shuffle: self.shuffle_data()
        start_index = defaultdict(int)
        for _ in range(self.iterations):
            index_batch = []
            for c in self.classes:
                for _ in range(100):  # TODO: hack solution to annoying error (https://discuss.pytorch.org/t/runtimeerror-setstorage/188897)
                    try:
                        index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c] + self.K_shot])
                        start_index[c] += self.K_shot
                        break
                    except RuntimeError:
                        logger.warning("This nasty bug again! Grrr")
                    else:
                        break
            yield index_batch

    def __len__(self):
        return self.iterations

class SampledDataset(Dataset, ABC):

    def __init__(self, image_paths, resize_ratio, training=False):
        super().__init__()
        self.resize_ratio = resize_ratio
        self.image_paths = image_paths
        # print(">>>>>>>>>>>>>>>> length of dataset = {}".format(len(image_paths)))
        self.image_paths.reset_index(inplace=True, drop=True) #.to_dict() # dataframe
        self.targets = image_paths['label_index'].tolist() # .to_numpy()
        # print(self.image_paths[:5])
        self.training = training

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        # print(">>>>>>>>>>>>>>>> index = {}".format(index))
        image_path = self.image_paths.iloc[int(index)].image_path
        mask_path = self.image_paths.iloc[int(index)].mask_path
        label_index = self.image_paths.iloc[int(index)].label_index

        preprocessed = False  # TODO: change from being manually coded here to command line option
        if not preprocessed:
            image = read_image(image_path)  # , grayscale=self.grayscale)
        else:
            file_comps = os.path.split(image_path)
            preprocessed_path = str(Path(file_comps[0], 'preprocessed', 'new_'+file_comps[1]))
            image = read_image(preprocessed_path)

        item = dict(image_path=image_path, label=label_index)

        if label_index == 0:
            mask = np.zeros(shape=image.shape[:2])  # create empty mask for normal (0) images
        else:
            mask = cv2.imread(mask_path, flags=0) / 255.0  # load mask for anomalous (1) images

        transformed = self.transform_images(Image.fromarray(image), Image.fromarray(mask))
        # print(">>>>>>>>>>>>>>>> image = {} \tmask = {}".format(transformed["image"].shape, transformed["mask"].shape))

        item["image"] = transformed["image"]
        item["mask_path"] = mask_path
        item["mask"] = transformed["mask"]
        item["rotated"] = transformed["rotated"]
        return item

    def transform_images(self, image, mask):
        # resize = T.Resize(size=(math.ceil(image.size[1] / self.resize_ratio), math.ceil(image.size[0] / self.resize_ratio)))
        model_factor = 32 # otherwise size of encoder and decoder feature sets don't match up
        resize = T.Resize(size=(round((image.size[1] / self.resize_ratio) / model_factor) * model_factor, round((image.size[0] / self.resize_ratio) / model_factor) * model_factor))
        image = resize(image)
        mask = resize(mask)

        if self.training:

            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

            if random.random() > 0.5:
                transform_colour = T.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5)
                image = transform_colour(image)

            if random.random() > 0.5:  # 20240721 added
                image = F.rotate(image, angle=90)
                mask = F.rotate(mask, angle=90)

        image_rot = F.rotate(image, angle=90)

        image = T.ToTensor()(image) #F.to_tensor(image)
        mask = T.ToTensor()(mask) #F.to_tensor(mask)
        image_rot = T.ToTensor()(image_rot)

        normalise = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image = normalise(image)
        image_rot = normalise(image_rot)

        return {"image": image, "mask": mask, "rotated": image_rot}

class SampleDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        root: Path | str,
        category: str,
        image_size: int | tuple[int, int] | None = None,
        resize_ratio: int = 1,
        center_crop: int | tuple[int, int] | None = None,
        normalization: str = 'imagenet',
        grayscale: bool = False,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: str = 'segmentation',
        transform_config_train: str | None = None,
        transform_config_eval: str | None = None,
        test_split_mode: str = 'from_dir',
        test_split_ratio: float = 0.2,
        val_split_mode: str = 'same_as_test',
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.root = root #Path(root)
        self.category = Path(category)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.test_split_mode = test_split_mode
        self.test_split_ratio = test_split_ratio
        self.val_split_mode = val_split_mode
        self.val_split_ratio = val_split_ratio
        self.seed = seed

        # class_list = [dir for dir in os.listdir(self.root + "carpet/images")] # get ['good', 'bad']
        sub_categories = ['train/good', 'test/good', 'test/bad']
        labels = [0, 0, 1]
        all_data: dict[str, list] = {}
        all_data['image_path'] = []
        all_data['mask_path'] = []
        all_data['label_index'] = []
        # for label, class_name in enumerate(class_list):
        for i, category in enumerate(sub_categories):
            label = labels[i]
            images_path = self.root + "carpet/" + category
            file_names = sorted([file_name for file_name in os.listdir(images_path) if '.png' in file_name])
            # print('>>>>>>>>>>>>>>>> image path: {} \tnumber of files: {}'.format(images_path, len(file_names)))
            for file_name in file_names:
                all_data['image_path'].append(os.path.join(images_path, file_name))
                all_data['label_index'].append(label)
                if label == 1:  # anomalous images need to have ground truth images of the same name
                    mask_path = (str(os.path.join(images_path, file_name))[:-4] + '_mask.png').replace("test", "ground_truth")
                    all_data['mask_path'].append(mask_path)  # TODO: test here that mask image exists, otherwise give error message
                else:
                    all_data['mask_path'].append("")
                # idx_to_class[idx] = (class_name, len(img_paths))
        all_data = pd.DataFrame(all_data)
        # train_data, test_data = train_test_split(all_data, test_size=self.test_split_ratio, random_state=self.seed)
        train_data, test_data = train_test_split(all_data[all_data['label_index'] == 0].copy(), test_size=self.test_split_ratio, random_state=self.seed)
        train_bad, test_bad = train_test_split(all_data[all_data['label_index'] == 1].copy(), test_size=self.test_split_ratio, random_state=self.seed)
        # print('>>>>>>>>>>>>>>>> train: {} \ttest: {} \ttrain_bad: {} \ttrain_good: {}'.format(len(train_data), len(test_data), len(train_bad), len(test_bad)))
        train_data = pd.concat([train_data, train_bad], axis=0) #train_data.loc[len(train_data)] = train_bad
        train_data.reset_index(inplace=True)
        test_data = pd.concat([test_data, test_bad], axis=0) #test_data.loc[len(test_data)] = test_bad
        test_data.reset_index(inplace=True)

        self.train_dataset = SampledDataset(train_data, resize_ratio, training=True)
        self.test_dataset = SampledDataset(test_data, resize_ratio)
        assert self.train_dataset is not None
        assert self.test_dataset is not None
        
        self._samples: DataFrame | None = None

    def setup(self, stage): # TODO: double check original setup function that was deleted
        if self.val_split_mode == 'from_test': # randomly sampled from test set
            self.test_dataset, self.val_dataset = random_split(self.test_dataset, self.val_split_ratio, label_aware=True, seed=self.seed) # random_split() function works specifically on Dataset objects
        elif self.val_split_mode == 'same_as_test': # same as test set
            self.val_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=FewShotBatchSampler(self.train_dataset.targets, include_query=True, K_shot=self.train_batch_size, N_way=2, shuffle=True),
            num_workers=self.num_workers,
            # collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_sampler=FewShotBatchSampler(self.val_dataset.targets, include_query=True, K_shot=self.eval_batch_size, N_way=2, shuffle=False, shuffle_once=True),
            num_workers=self.num_workers,
            # collate_fn=collate_fn, # TODO: add collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_sampler=FewShotBatchSampler(self.test_dataset.targets, include_query=True, K_shot=self.eval_batch_size, N_way=2, shuffle=False, shuffle_once=True),
            num_workers=self.num_workers,
            # collate_fn=collate_fn, # TODO: add collate_fn
        )

def collate_fn(batch):
    elem = batch[0]  # sample an element from the batch to check the type.
    out_dict = {}
    if isinstance(elem, dict):
        out_dict.update({key: default_collate([item[key] for item in batch]) for key in elem})
        return out_dict
    return default_collate(batch)
    # return tuple(zip(*batch))


def get_config_params(
    model_name: str | None = None,
    config_path: Path | str | None = None,
    weight_file: str | None = None,
    #config_filename: str | None = "config",
    #config_file_extension: str | None = "yaml",
    infer = False,
) -> DictConfig | ListConfig:

    if config_path is None:
        config_path = Path(os.getcwd(), 'models', model_name, 'config.yaml')
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


#####################################################################################################################
# The below is just for testing
#####################################################################################################################

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of model to train/evaluate")
    return parser.parse_args()

if __name__ == "__main__":
    # random_state = 42

    args = get_args()
    config = get_config_params(model_name=args.model) #, config_path=args.config)
    print("Project path: {}".format(config.project.path))

    datamodule = SampleDataModule(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size, config.dataset.image_size),
            resize_ratio=config.dataset.resize_ratio,
            center_crop=config.dataset.center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=2, #config.dataset.train_batch_size,
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

    for i in range(3):
        img_batch = next(dataiter) # extract batch
        print(img_batch.keys())
        print(img_batch['image'].shape)
        print([img_name[-10:] for img_name in img_batch['image_path']])
        print([mask_name[-15:] for mask_name in img_batch['mask_path']])

# Command Line:
# conda deactivate
# conda activate anomalib_env2
# cd '/home/brionyf/Documents/GitHub/anomaly-detection'
# python dataset_sampler.py --model 'model_1'
