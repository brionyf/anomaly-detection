from __future__ import annotations

import os
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from skimage.segmentation import mark_boundaries

def get_image_filenames(path):
    # # image_filenames = sorted([Path(path, file_name) for file_name in os.listdir(path) if '.png' in file_name and '0_' in file_name])
    # image_filenames = sorted([Path(path, file_name) for file_name in os.listdir(path) if '_0.png' in file_name])
    image_filenames = sorted([Path(path, file_name) for file_name in os.listdir(path) if '.png' in file_name])
    if isinstance(image_filenames, str):
        image_filenames = Path(image_filenames)
    if not image_filenames:
        raise ValueError(f"Found 0 images in {path}")
    return image_filenames

def equalise_hist(img, adaptive=False): # HE is a statistical approach for spreading out intensity values
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if not adaptive:
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # equalize the histogram of the Y channel (brightness)
    else:
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16,16))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

def read_image(path, grayscale=False): #, image_size):
    path = path if isinstance(path, str) else str(path)
    image = cv2.imread(path)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, :, np.newaxis]
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = equalise_hist(image, adaptive=True)
    # if isinstance(image_size, int):
    #     height, width = (image_size, image_size)
    # elif isinstance(image_size, tuple):
    #     height, width = int(image_size[0]), int(image_size[1])
    # image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
    return image

def visualize_predict_batch(batch):
    batch_size, _num_channels, _, _ = batch["image"].size()
    _, _, height, width = batch["anomaly_maps"].size()
    # print('>>>>>>>>>>>>>>> {} {} {} {}'.format(batch_size, _num_channels, height, width))  # result is: 1 3 542 775
    for i in range(batch_size):
        # CUSTOM - only save images with detections
        # print('>>>>>>>>>>>>>>> Before squeeze: {} {}'.format(batch["anomaly_maps"][i].shape, batch["pred_masks"][i].shape))  # result is: torch.Size([1, 542, 800]) torch.Size([1, 542, 800])
        anomaly_map = batch["anomaly_maps"][i].squeeze().cpu().numpy() if "anomaly_maps" in batch else None
        pred_mask = batch["pred_masks"][i].squeeze().int().cpu().numpy() if "pred_masks" in batch else None
        # print('>>>>>>>>>>>>>>> After squeeze: {} {}'.format(anomaly_map.shape, pred_mask.shape))  # result is: (542, 800) (542, 800)
        # if pred_mask.max() == 1.0:  # CUSTOM - added to only save images that have anomalous areas
        # if pred_mask.max() == 0.0:  # CUSTOM - added to only save images that have NO anomalous areas
            # CUSTOM - end
        if "image_path" in batch:
            if "original_image" in batch:
                image = batch["original_image"][i].cpu().numpy()
            else:
                image = read_image(path=batch["image_path"][i])#, image_size=(height, width), crop=True)
            image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
            # print(image.shape)
        else:
            raise KeyError("No save path specified, image results won't be saved")

        if '_0' in batch["image_path"][i]:
            pred_mask[:,-100:] *= 0 # TODO: code to ignore predictions on mask-end of Camera 0 image

        image_result = process_predict_images(image, anomaly_map, pred_mask)

        yield image, image_result  # CUSTOM - return original image too

def process_predict_images(image, anomaly_map, pred_mask):
    if anomaly_map is not None:
        # print('>>>>>>>>>>>>>>> Heatmap: {} \tImage: {}'.format(self.anomaly_map.shape, self.image.shape))  # result is:
        heat_map = overlay_anomaly_map(anomaly_map, image, normalize=False)
    if pred_mask is not None and pred_mask.max() <= 1.0:
        pred_mask *= 255
    # final_image = mark_boundaries(heat_map, pred_mask, color=(1, 0, 0), mode="thick")
    final_image = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")
    return (final_image * 255).astype(np.uint8)

def overlay_anomaly_map(anomaly_map, image, alpha=0.4, gamma=0, normalize=False):
    # print('>>>>>>>>>>>>>>> anomaly map: {} \timage: {}'.format(anomaly_map.shape, image.shape))  # result is: anomaly map: (542, 800) 	image: (542, 800, 3)
    # anomaly_map = anomaly_map.squeeze()
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)
    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)


def visualize_predict_batch2(batch):
    batch_size, _num_channels, _, _ = batch["image"].size()
    _, _, height, width = batch["recon_images"].size()
    # print('>>>>>>>>>>>>>>> {} {} {} {}'.format(batch_size, _num_channels, height, width))  # result is: 1 3 542 775
    for i in range(batch_size):
        # print('>>>>>>>>>>>>>>> Before squeeze: {} {}'.format(batch["original_image"][i].shape, batch["anomaly_maps"][i].shape))  # result is: torch.Size([1, 542, 800]) torch.Size([1, 542, 800])

        recon_image = batch["recon_images"][i].squeeze().cpu().numpy() if "recon_images" in batch else None
        print(">>>>>>>>>>>>>>> 1) Recon image - min: {} \tmax: {}".format(recon_image.min(), recon_image.max()))  # result is:
        recon_image = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min()) #np.ptp(recon_image)
        recon_image = recon_image * 255
        recon_image = recon_image.astype(np.uint8)
        print(">>>>>>>>>>>>>>> 2) Recon image - min: {} \tmax: {}".format(recon_image.min(), recon_image.max()))  # result is:

        image = batch["original_image"][i].cpu().numpy()
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)

        yield image, recon_image  # CUSTOM - return original image too
