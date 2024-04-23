from __future__ import annotations

import os
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F
from torch import Tensor


def get_image_filenames(path: str | Path) -> list[Path]:

    image_filenames = sorted([Path(path, file_name) for file_name in os.listdir(path) if '.png' in file_name and '0_' in file_name]) 
    
    if isinstance(image_filenames, str):
        image_filenames = Path(image_filenames)

    if not image_filenames:
        raise ValueError(f"Found 0 images in {path}")

    return image_filenames


def generate_output_image_filename(input_path: str | Path, output_path: str | Path) -> Path:
    """Generate an output filename to save the inference image.

    This function generates an output filaname by checking the input and output filenames. Input path is
    the input to infer, and output path is the path to save the output predictions specified by the user.

    The function expects ``input_path`` to always be a file, not a directory. ``output_path`` could be a
    filename or directory. If it is a filename, the function checks if the specified filename exists on
    the file system. If yes, the function calls ``duplicate_filename`` to duplicate the filename to avoid
    overwriting the existing file. If ``output_path`` is a directory, this function adds the parent and
    filenames of ``input_path`` to ``output_path``.

    Args:
        input_path (str | Path): Path to the input image to infer.
        output_path (str | Path): Path to output to save the predictions.
            Could be a filename or a directory.

    Examples:
        >>> input_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> output_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> generate_output_image_filename(input_path, output_path)
        PosixPath('datasets/MVTec/bottle/test/broken_large/000_1.png')

        >>> input_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
        >>> output_path = Path("results/images")
        >>> generate_output_image_filename(input_path, output_path)
        PosixPath('results/images/broken_large/000.png')

    Raises:
        ValueError: When the ``input_path`` is not a file.

    Returns:
        Path: The output filename to save the output predictions from the inferencer.
    """

    if isinstance(input_path, str):
        input_path = Path(input_path)

    if isinstance(output_path, str):
        output_path = Path(output_path)

    # This function expects an ``input_path`` that is a file. This is to check if output_path
    if input_path.is_file() is False:
        raise ValueError("input_path is expected to be a file to generate a proper output filename.")

    file_path: Path
    if output_path.suffix == "":
        # If the output is a directory, then add parent directory name
        # and filename to the path. This is to ensure we do not overwrite
        # images and organize based on the categories.
        file_path = output_path / input_path.parent.name / input_path.name
    else:
        file_path = output_path

    # This new ``file_path`` might contain a directory path yet to be created.
    # Create the parent directory to avoid such cases.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    #if file_path.is_file():
    #    warnings.warn(f"{output_path} already exists. Renaming the file to avoid overwriting.")
    #    file_path = duplicate_filename(file_path)

    return file_path


def get_image_height_and_width(image_size: int | tuple[int, int]) -> tuple[int, int]:

    if isinstance(image_size, int):
        height_and_width = (image_size, image_size)
    elif isinstance(image_size, tuple):
        height_and_width = int(image_size[0]), int(image_size[1])
    else:
        raise ValueError("``image_size`` could be either int or tuple[int, int]")

    return height_and_width


def equalise_hist(img, adaptive=False): # HE is a statistical approach for spreading out intensity values

    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if not adaptive:
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # equalize the histogram of the Y channel (brightness)
    else:
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16,16))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return img_output

def read_image(path: str | Path, image_size: int | tuple[int, int] | None = None, crop=False) -> np.ndarray:

    path = path if isinstance(path, str) else str(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #image = equalise_hist(image, adaptive=True)

    if image_size:
        height, width = get_image_height_and_width(image_size)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)

    return image
