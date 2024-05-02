from itertools import product
from math import ceil
import torch

class Tiler:

    def __init__(self, tile_size):
        self.tile_size = tile_size
        self.device = None
        self.image_h = None
        self.image_w = None
        self.num_channels = None
        self.batch_size = None
        self.num_patches_h = None
        self.num_patches_w = None

    def tile(self, image):

        self.batch_size, self.num_channels, self.image_h, self.image_w = image.shape
        self.device = image.device
        self.num_patches_h = ceil(self.image_h / self.tile_size)
        self.num_patches_w = ceil(self.image_w / self.tile_size)

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
            tiles[tile_i, tile_j, :] = image[:, :, loc_i:(loc_i + self.tile_size), loc_j:(loc_j + self.tile_size)]

        # rearrange the tiles in order [tile_count * batch, channels, tile_height, tile_width]
        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        tiles = tiles.contiguous().view(-1, self.num_channels, self.tile_size, self.tile_size)

        return tiles

    def untile(self, tiles):

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
