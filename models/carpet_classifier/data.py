import logging
from pathlib import Path
from typing import Sequence

import os
import numpy as np
import cv2
import math
import random
from skimage.filters.rank import median
from skimage import exposure
from sklearn.model_selection import train_test_split
from collections import defaultdict
#import torchvision.transforms.functional as TF
from omegaconf import DictConfig, ListConfig, OmegaConf
#import time
from datetime import datetime
import copy

import warnings
from warnings import warn

import albumentations as A
import pandas as pd
from pandas import DataFrame
from abc import ABC

import torch
# torch.manual_seed(17)
from torch import Tensor
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.dataset import Dataset, random_split

import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class FewShotBatchSampler:
    def __init__(self, dataset_targets, K_shot, N_way, include_query=False, shuffle=True, shuffle_once=False):
        super().__init__()
        self.dataset_targets = torch.Tensor(dataset_targets).type(torch.int64)  # tensor of the data labels (0 - good, 1 - bad)
        self.N_way = N_way  # Number of classes to sample per batch
        self.K_shot = K_shot  # Number of examples to sample per class in the batch
        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch
        self.shuffle = shuffle  # data is shuffled each iteration (during training)

        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_shot
        self.iterations = min(self.batches_per_class.values())
        # # print(">>>>>>>>>>>> indices per class: \t{}".format(' '.join(f'{self.indices_per_class[c]}' for c in self.classes)))
        # print(">>>>>>>>>>>> batches per class: \t{}".format(' '.join(f'Class Index {c}: {self.batches_per_class[c]} |' for c in self.classes)))
        # print(">>>>>>>>>>>> num of iterations: {}".format(self.iterations))

        if shuffle_once or self.shuffle: self.shuffle_data()  # shuffle_once - shuffled once in the beginning, but kept constant across iterations (for validation)
        # self.include_query = include_query
        # if self.include_query: self.K_shot *= 2

    def shuffle_data(self):
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]

    def __iter__(self):
        if self.shuffle:
            self.shuffle_data()
        start_index = defaultdict(int)
        for _ in range(self.iterations):
            index_batch = []
            for c in self.classes:
                for i in range(100):  # TODO: hack solution to annoying error (https://discuss.pytorch.org/t/runtimeerror-setstorage/188897)
                    try:
                        index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c] + self.K_shot])
                        start_index[c] += self.K_shot
                        break
                    except RuntimeError:
                        logger.warning("This nasty bug again! Grrr")
                    else:
                        break
            # index_batch = []
            # for c in self.classes:  # For each class, select the next K examples and add them to the batch
            #     index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c] + self.K_shot])
            #     start_index[c] += self.K_shot
            # # if self.include_query:  # If we return support+query set, sort them so that they are easy to split
            # #     index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations


class SampledDataset(Dataset, ABC):

    def __init__(self, image_paths, training=False):
        super().__init__()
        self.image_paths = image_paths
        self.image_paths.reset_index(inplace=True, drop=True)  # .to_dict() # dataframe
        self.targets = image_paths['label'].tolist()  # .to_numpy()
        self.training = training
        self.transforms = self.get_transforms()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        image_path = self.image_paths.iloc[int(index)].image_path
        label = self.image_paths.iloc[int(index)].label
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        item = dict(image_path=image_path, label=label)
        # transformed = self.transform_images(Image.fromarray(image))
        transformed = self.transforms(Image.fromarray(image))
        # print(">>>>>>>>>> min: {} \tmax: {}".format(transformed.min(), transformed.max()))
        item["image"] = transformed #transformed["image"]
        item["class"] = label
        return item

    def transform_images(self, image):
        resize = T.Resize(size=(256, 256))  # TODO: don't hardcode image resize here...
        image = resize(image)
        if self.training:
            if random.random() > 0.5: # Random horizontal flipping
                image = F.hflip(image)
            if random.random() > 0.5: # Random vertical flipping
                image = F.vflip(image)
            if random.random() > 0.5:
                transform_colour = T.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5)
                image = transform_colour(image)
            if random.random() > 0.5:  # 20240721 added
                image = F.rotate(image, angle=90)
        image = T.ToTensor()(image) #F.to_tensor(image)
        normalise = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image = normalise(image)
        return {"image": image}

    def get_transforms(self):
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            T.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15), #resample=False, fillcolor=0
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


class SampleDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        root: Path | str,
        image_size: int | tuple[int, int] | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.root = root  # Path(root)
        #self.batch_size = batch_size
        self.num_workers = 4 #num_workers
        self.test_split_ratio = 0.25
        self.seed = seed
        num_test_classes = 2

        self.classes = [sub_dir for sub_dir in os.listdir(self.root)]
        print(">>>>>>>>>> all classes: {}".format(self.classes))
        self.test_classes = ['carpet3', 'carpet7'] #[self.classes[random.randint(0, len(self.classes))] for _ in range(num_test_classes)]
        print(">>>>>>>>>> test classes: {}".format(self.test_classes))

        self.nway = len(self.classes)   # Number of classes
        self.kshot = 2  # Number of images per class

        all_data: dict[str, list] = {}
        all_data['image_path'] = []
        all_data['label'] = []
        for idx, class_name in enumerate(self.classes):
            label = idx
            images_path = str(Path(self.root, class_name))
            file_names = sorted([file_name for file_name in os.listdir(images_path) if '.png' in file_name])
            for file_name in file_names:
                all_data['image_path'].append(os.path.join(images_path, file_name))
                all_data['label'].append(label)
            # idx_to_class[idx] = (class_name, len(file_names))
            # print(">>>>>>>>>> Class {}: {} instances".format(class_name, len(file_names)))
        all_data = pd.DataFrame(all_data)

        train_data = None
        test_data = None
        for i, class_name in enumerate(self.classes):

            if class_name not in self.test_classes:
                train, test = train_test_split(all_data[all_data['label'] == i].copy(), test_size=self.test_split_ratio, random_state=self.seed)
            else:
                train = None
                test = all_data[all_data['label'] == i].copy()

            if train_data is None:
                train_data = train
            else:
                train_data = pd.concat([train_data, train], axis=0)
                train_data.reset_index(inplace=True)
                train_data.pop("index")

            if test_data is None:
                test_data = test
            else:
                test_data = pd.concat([test_data, test], axis=0)
                test_data.reset_index(inplace=True)
                test_data.pop("index")

        self.train_dataset = SampledDataset(train_data, training=True)
        self.test_dataset = SampledDataset(test_data)
        print(">>>>>>>>>> train dataset: {} \ttest dataset: {}".format(len(self.train_dataset), len(self.test_dataset)))
        assert self.train_dataset is not None
        assert self.test_dataset is not None

        self._samples: DataFrame | None = None

    def setup(self):
        self.val_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=FewShotBatchSampler(self.train_dataset.targets, K_shot=self.kshot, N_way=self.nway, include_query=True, shuffle=True),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_sampler=FewShotBatchSampler(self.val_dataset.targets, K_shot=1, N_way=self.nway, include_query=True, shuffle=False, shuffle_once=True),
            num_workers=self.num_workers,
            # collate_fn=collate_fn, # TODO: add collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_sampler=FewShotBatchSampler(self.test_dataset.targets, K_shot=1, N_way=self.nway, include_query=True, shuffle=False, shuffle_once=True),
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
