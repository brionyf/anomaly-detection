import numpy as np
import os
import cv2
import math

from skimage.filters.rank import median

from enum import Enum
from itertools import product
from math import ceil
from typing import Sequence

import torch
import torchvision.transforms as T
from torch import Tensor
from torch.nn import functional as F

from argparse import ArgumentParser

OLD_MIN = 0

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Directory to load/save images from/to")
    parser.add_argument("--tile_size", type=int, required=True, help="Tile size [Example: 1024]")
    return parser

class Tiler:

    def __init__(self, image, tile_size):
        self.image = image
        self.tile_size = tile_size
        self.device = self.image.device

        # self.stride_h, self.stride_w = self.__validate_size_type(stride)
        self.batch_size, self.num_channels, self.image_h, self.image_w = image.shape
        self.num_patches_h = ceil(self.image_h / self.tile_size)
        self.num_patches_w = ceil(self.image_w / self.tile_size)

    def unfold(self): # tile

        # create an empty torch tensor for output
        tiles = torch.zeros((self.num_patches_h, self.num_patches_w, self.batch_size, self.num_channels, self.tile_size, self.tile_size), device=self.device)

        # h_overlap = int(((self.num_patches_h * self.tile_size) - self.image_h) / (self.num_patches_h - 1))
        # w_overlap = int(((self.num_patches_w * self.tile_size) - self.image_w) / (self.num_patches_w - 1))
        h_step = int((self.image_h - self.tile_size) / (self.num_patches_h - 1))  # resized_tile_h - h_overlap
        w_step = int((self.image_w - self.tile_size) / (self.num_patches_w - 1))  # resized_tile_w - w_overlap
        num = (self.num_patches_h, self.num_patches_w)
        # print('>>>>>>>>>>>>>>> Unfold: {} {} {} {} {}'.format(num, self.input_w, self.tile_size_w, w_overlap, w_step))  # result was:

        for (tile_i, tile_j), (loc_i, loc_j) in zip(product(range(self.num_patches_h), range(self.num_patches_w)),
                                                    product(range(0, self.image_h-self.tile_size+1, h_step), range(0, self.image_w-self.tile_size+1, w_step))):
            tiles[tile_i, tile_j, :] = self.image[:, :, loc_i:(loc_i + self.tile_size), loc_j:(loc_j + self.tile_size)]

        # rearrange the tiles in order [tile_count * batch, channels, tile_height, tile_width]
        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        tiles = tiles.contiguous().view(-1, self.num_channels, self.tile_size, self.tile_size)

        return tiles

    def fold(self, tiles: Tensor) -> Tensor: # untile

        # number of channels differs between image and anomaly map, so infer from input tiles.
        _, tile_channels, resized_tile_h, resized_tile_w = tiles.shape
        scale_h, scale_w = 1.0, 1.0  #(tile_size_h / self.tile_size_h), (tile_size_w / self.tile_size_w)  # CUSTOM - adjusted scale to = 1.0
        resized_h = int((self.image_h / self.tile_size) * resized_tile_h)
        resized_w = int((self.image_w / self.tile_size) * resized_tile_w)

        # reconstructed image dimension
        image_size = (self.batch_size, tile_channels, int(resized_h * scale_h), int(resized_w * scale_w))

        # rearrange input tiles in format [tile_count, batch, channel, tile_h, tile_w]
        tiles = tiles.contiguous().view(
            self.batch_size,
            self.num_patches_h,
            self.num_patches_w,
            tile_channels,
            resized_tile_h,
            resized_tile_w,
        )
        tiles = tiles.permute(0, 3, 1, 2, 4, 5)
        tiles = tiles.contiguous().view(self.batch_size, tile_channels, -1, resized_tile_h, resized_tile_w)
        tiles = tiles.permute(2, 0, 1, 3, 4)

        # create tensors to store intermediate results and outputs
        image = torch.zeros(image_size, device=self.device)
        lookup = torch.zeros(image_size, device=self.device)
        ones = torch.ones(resized_tile_h, resized_tile_w, device=self.device)

        # h_overlap = int(((self.num_patches_h * resized_tile_h) - resized_h) / (self.num_patches_h - 1))
        # w_overlap = int(((self.num_patches_w * resized_tile_w) - resized_w) / (self.num_patches_w - 1))
        h_step = int((resized_h - resized_tile_h) / (self.num_patches_h - 1))  # resized_tile_h - h_overlap
        w_step = int((resized_w - resized_tile_w) / (self.num_patches_w - 1))  # resized_tile_w - w_overlap
        num = (self.num_patches_h, self.num_patches_w)
        # print('>>>>>>>>>>>>>>> Fold: {} {} {} {} {}'.format(num, resized_w, resized_tile_w, w_overlap, w_step))  # result was:

        # reconstruct image by adding patches to their respective location and create a lookup for patch count in every location
        for patch, (loc_i, loc_j) in zip(tiles, product(range(0, resized_h-resized_tile_h+1, h_step), range(0, resized_w-resized_tile_w+1, w_step))):
            # print('>>>>>>>>>>>>>>> (h,w): ({},{})'.format(loc_i, loc_j))  # result was:
            image[:, :, loc_i : (loc_i + resized_tile_h), loc_j : (loc_j + resized_tile_w)] += patch
            lookup[:, :, loc_i : (loc_i + resized_tile_h), loc_j : (loc_j + resized_tile_w)] += ones

        # divide the reconstructed image by the lookup to average out the values
        image = torch.divide(image, lookup)

        # alternative way of removing nan values (isnan not supported by openvino)
        image[image != image] = 0  # pylint: disable=comparison-with-itself

        return image

def auto_crop_image(img):

    resize_ratio = 0.2
    img_resized = cv2.resize(img, (int(img.shape[1]*resize_ratio), int(img.shape[0]*resize_ratio)))
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

    base_crop = 100
    padding = 25
    delta = 25
    pts = np.argwhere(edges[:, base_crop:] > 0)
    try:
        _, x = pts.min(axis=0)
        # TODO: only use x if not "too" different from old_min, else use old_min
        # if abs(x - OLD_MIN) > delta: x = OLD_MIN
        OLD_MIN = x
    except ValueError:  # raised if `min()` is empty
        x = OLD_MIN
    x_resized = int((x + base_crop + padding) / resize_ratio)
    x_resized = int(math.ceil(x_resized / 100.0)) * 100  # ADDED to fix problem of weird, regular, rectangular detection in bottom of images
    # print(x_resized)

    return img[:, x_resized:3900, :]

def convert_to_tensor(img):
    np_img = np.asarray(img)  # (H, W, C)
    np_img_batch = np.expand_dims(np_img, axis=0)  # (1, H, W, C)
    np_final = np.moveaxis(np_img_batch, -1, 1)  # (1, C, H, W)
    return torch.from_numpy(np_final)

def get_image_filenames(folder):
    return sorted([folder + file_name for file_name in os.listdir(folder) if '.png' in file_name and 'marked' not in file_name])

if __name__ == '__main__':

    args = get_parser().parse_args()

    images = get_image_filenames(args.img_dir)

    for filename in images:

        img = cv2.imread(filename)#, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"

        cropped_img = auto_crop_image(img)
        print("Cropped Image: {}".format(cropped_img.shape, cropped_img.dtype))

        tensor_img = convert_to_tensor(cropped_img)
        print("Tensor Image: {}".format(tensor_img.shape, tensor_img.dtype))

        tiler = Tiler(tensor_img, args.tile_size)

        tiled_img = tiler.unfold()
        tiled_img = T.Resize((256, 256))(tiled_img)
        print("Tiled Image: {}".format(tiled_img.shape, tiled_img.dtype))

        untiled_img = np.asarray(tiler.fold(tiled_img))
        untiled_img = np.moveaxis(untiled_img[0], 0, -1)  # (H, W, C)
        print("Untiled Image: {}".format(untiled_img.shape, untiled_img.dtype))

        cv2.imwrite(filename.replace('.png','') + '_untiled.png', untiled_img)

# Command Line:
# conda deactivate
# conda activate anomalib_env2
# cd '/home/brionyf/Desktop/Code/00 generic code'
# python tiling_test.py --img_dir '/home/brionyf/Desktop/Images/tiling tests/' --tile_size 1024
