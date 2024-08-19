import numpy as np
import cv2
import math
from skimage.filters.rank import median
from PIL import Image

import torchvision.transforms as T
from torch.utils.data.dataset import Dataset, random_split

from images import read_image

import logging
logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    def __init__(self, files, grayscale, resize_ratio=1): #, transform: A.Compose | None = None):
        super().__init__()
        self.image_filenames = files
        # self.transform = transform
        self.old_min = 0  # CUSTOM - added ...
        self.grayscale = True if grayscale == 'True' else False
        self.resize_ratio = resize_ratio

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int):
        """Get the image based on the `index`."""
        image_filename = self.image_filenames[index]
        image = read_image(path=image_filename, grayscale=self.grayscale)
        # print('>>>>>>>>>>>>>>> Image shape: {}'.format(image.shape))  # result is: (2168, 4096, 3)
        # image = self._auto_crop_image(image)
        image = self._process_image(image)
        # print('>>>>>>>>>>>>>>> Post auto-crop: {}'.format(image.shape))  # result is: (2168, 3200, 3)
        # pre_processed = self.transform(image=image)
        pre_processed = self._transform_images(Image.fromarray(image))
        # print('>>>>>>>>>>>>>>> Post transforms: {} {} {}'.format(image.shape, pre_processed["image"].shape, pre_processed["image"].dtype))  # result is: torch.Size([3, 256, 256]) torch.float32
        pre_processed["image_path"] = str(image_filename)
        pre_processed["original_image"] = image
        return pre_processed

    def _process_image(self, img):
        camera_factor = 1.3
        crop_ratios = (0.15, 0.85)
        camera_small = (2168, 4096, 3)
        camera_large = (math.ceil(3032 / camera_factor), math.ceil(5320 / camera_factor), 3) # adjusted
        if img.shape == camera_small:
            img = img[:, math.ceil(img.shape[1] * crop_ratios[0] * 1.5):math.ceil(img.shape[1] * crop_ratios[1]), :] # h,w,c
            processed = np.zeros((camera_small[0], math.ceil(camera_large[1] * (1 - crop_ratios[0] * 2)), 3))
            processed[:,:img.shape[1],:] = img
        else:
            image = cv2.resize(img, (camera_large[1], camera_large[0]), interpolation=cv2.INTER_AREA)
            processed = image[math.ceil((camera_large[0] - camera_small[0]) / 2):math.ceil((camera_large[0] - camera_small[0]) / 2) + camera_small[0], math.ceil(image.shape[1] * crop_ratios[0]):math.ceil(image.shape[1] * crop_ratios[1]), :]
        return processed.astype(np.uint8)

    def _transform_images(self, image):
        # print(">>>>>>> before resize: {}".format(image.size)) # example result: (2866, 2168) --> w,h
        model_factor = 32 # otherwise size of encoder and decoder feature sets don't match up
        h, w = round((image.size[1] / self.resize_ratio) / model_factor) * model_factor, round((image.size[0] / self.resize_ratio) / model_factor) * model_factor
        resize = T.Resize(size=(h, w))
        image = resize(image)
        # print(">>>>>>> after resize: {}".format(image.size)) # example result: (704, 544) --> w,h

        image = T.ToTensor()(image) #F.to_tensor(image)

        normalise = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image = normalise(image)

        return {"image": image}

    def _auto_crop_image(self, img):  # TODO: some values are hardcoded, so if images are different sizes, will result in wrong crops

        resize_ratio = 0.25
        if img.shape[:2] != (2168, 4096):
            img = cv2.resize(img, (4096, 2168))
        # print('>>>>>>>>>>>>>>> Image shape: {}'.format(img.shape))  # result is: (2168, 3100, 3)
        img_resized = cv2.resize(img, (int(img.shape[1]*resize_ratio), int(img.shape[0]*resize_ratio)))
        # print('>>>>>>>>>>>>>>> Image shape: {}'.format(img_resized.shape))  # result is: (2168, 3100, 3)
        if img.shape[2] != 1:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized

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
