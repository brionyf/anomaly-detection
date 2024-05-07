# Variation of original Code:
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# https://github.com/openvinotoolkit/anomalib

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from pytorch_lightning.callbacks import Callback
from scipy.stats import norm
from torch.distributions import Normal, LogNormal

NORM_METHODS = ['MINMAX', 'CDF']


class NormalisationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using min-max normalization."""

    def __init__(self, norm_method='min_max'):
        if norm_method in NORM_METHODS:
            self.norm_method = norm_method
        else:
            self.norm_method = 'min_max'

    def setup(self, trainer, pl_module, stage):
        """Adds min_max metrics to normalization metrics."""
        del trainer, stage  # Unused variables

        if not hasattr(pl_module, "normalization_metrics"):
            if self.norm_method == 'min_max':
                pl_module.normalization_metrics = MinMax().cpu() # TODO: rename to 'norm_metric'
            elif self.norm_method == 'cdf':
                pass
                # pl_module.normalization_metrics = AnomalyScoreDistribution().cpu()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    # def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx): # TODO: look into updating these metrics at end of training, when anomaly scores have 'settled'
        del trainer, batch, batch_idx, dataloader_idx  # Unused variables
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): # NOTE: can't do during train, as 'anomaly_maps' are only created during validation
    #     del trainer, batch, batch_idx  # Unused variables

        if "anomaly_maps" in outputs:
            pl_module.normalization_metrics(outputs["anomaly_maps"]) # update the (min and max) or (mean and std) observed values

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        del trainer, batch, batch_idx, dataloader_idx  # Unused variables

        stats = pl_module.normalization_metrics.cpu()
        pixel_threshold = 10 * stats.max # pl_module.pixel_threshold.value.cpu()
        mask_threshold = 0
        if "anomaly_maps" in outputs: # normalize a batch of predictions (anomaly maps)
            if self.norm_method == 'min_max':
                outputs["anomaly_maps"] = minmax_normalise(outputs["anomaly_maps"], pixel_threshold, stats.min, stats.max)
                outputs["pred_masks"] = outputs["anomaly_maps"] > mask_threshold
            elif self.norm_method == 'cdf':
                pass
                # outputs["anomaly_maps"] = standardize(outputs["anomaly_maps"], stats.pixel_mean, stats.pixel_std, center_at=stats.image_mean)
                # outputs["anomaly_maps"] = cdf_normalize(outputs["anomaly_maps"], pixel_threshold)


class MinMax(Metric):
    """Track the min and max values of the observations in each batch."""

    full_state_update: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("min", torch.tensor(float("inf")), persistent=True)  # persistent: whether the state will be saved as part of the modules 'state_dict'
        self.add_state("max", torch.tensor(float("-inf")), persistent=True)

        self.min = torch.tensor(float("inf"))
        self.max = torch.tensor(float("-inf"))
        # print('>>>>>>>>>>>>>>> Initialising MinMax Metric...')

    def update(self, predictions, *args, **kwargs):
        del args, kwargs  # These variables are not used.

        if self.max == torch.tensor(float("-inf")):
            self.max = torch.max(self.max, torch.max(predictions))
        else:
            self.max = torch.min(self.max, torch.max(predictions))
        self.min = torch.min(self.min, torch.min(predictions))
        # print('>>>>>>>>>>>>>>> MinMax Metric: {} {}'.format(self.min, self.max))

    def compute(self):
        """Return min and max values."""
        return self.min, self.max


def minmax_normalise(anomaly_maps, threshold, min_val, max_val):
    """Apply min-max normalisation and shift the values such that the threshold value is centered at 0.5."""
    # normalized = ((anomaly_maps - threshold) / (max_val - min_val)) + 0.5
    normalized = ((anomaly_maps - threshold) / (max_val - min_val))
    if isinstance(anomaly_maps, (np.ndarray, np.float32, np.float64)):
        normalized = np.minimum(normalized, 1) # convert >1 values to = 1
        normalized = np.maximum(normalized, 0) # convert <0 (-ve) values to = 0
    elif isinstance(anomaly_maps, Tensor):
        normalized = torch.minimum(normalized, torch.tensor(1))
        normalized = torch.maximum(normalized, torch.tensor(0))
    else:
        raise ValueError(f"Anomaly maps must be either Tensor or Numpy array. Received {type(anomaly_maps)}")
    return normalized


# class AnomalyScoreDistribution(Metric):
#     """Mean and standard deviation of the anomaly scores of normal training data."""
#
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.add_state("pixel_mean", torch.empty(0), persistent=True)
#         self.add_state("pixel_std", torch.empty(0), persistent=True)
#         self.pixel_mean = torch.empty(0)
#         self.pixel_std = torch.empty(0)
#         self.count = 0
#
#     def update(self, *args, anomaly_maps, **kwargs):
#         del args, kwargs  # These variables are not used.
#
#         if anomaly_maps:
#             anomaly_maps = torch.vstack(anomaly_maps)
#             anomaly_maps = torch.log(anomaly_maps).cpu()
#             self.pixel_mean += anomaly_maps.mean(dim=0).squeeze()
#             self.pixel_std += anomaly_maps.std(dim=0).squeeze()
#             self.count += len(anomaly_maps)
#
#     def compute(self):
#         """Return mean and std values."""
#         return self.pixel_mean/self.count, self.pixel_std/self.count


# def standardize(anomaly_maps, mean, std, center_at):
#     """Standardize the targets to the z-domain."""
#     if isinstance(anomaly_maps, np.ndarray):
#         anomaly_maps = np.log(anomaly_maps)
#     elif isinstance(anomaly_maps, Tensor):
#         anomaly_maps = torch.log(anomaly_maps)
#     else:
#         raise ValueError(f"Anomaly maps must be either Tensor or Numpy array. Received {type(anomaly_maps)}")
#     standardized = (anomaly_maps - mean) / std
#     if center_at:
#         standardized -= (center_at - mean) / std
#     return standardized
#
# def cdf_normalize(anomaly_maps, threshold): #
#     """Normalize the targets by using the cumulative density function."""
#     if isinstance(anomaly_maps, Tensor):
#         image_threshold = threshold.cpu()
#         dist = Normal(torch.Tensor([0]), torch.Tensor([1]))
#         return dist.cdf(anomaly_maps.cpu() - image_threshold).to(anomaly_maps.device) # PyTorch cumulative density function
#     elif isinstance(anomaly_maps, np.ndarray):
#         return norm.cdf(anomaly_maps - threshold) # Numpy cumulative density function
#     else:
#         raise ValueError(f"Anomaly maps must be either Tensor or Numpy array. Received {type(anomaly_maps)}")


