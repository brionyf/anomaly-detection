import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

def get_transforms(
    image_size: int | tuple[int, int] | None = None,
    center_crop: int | tuple[int, int] | None = None,
    normalization: str = 'imagenet',
    to_tensor: bool = True,
    training: bool = False,
    tiling: bool = False,
    infer: bool = False,
) -> A.Compose:

    transforms_list = []

    if isinstance(image_size, int):
        resize_height, resize_width = (image_size, image_size)
    elif isinstance(image_size, tuple):
        resize_height, resize_width = int(image_size[0]), int(image_size[1])

    if not tiling:  # CUSTOM - added to not resize image before tiling
        print("Adding A.Resize() transform to transforms list...")
        transforms_list.append(A.Resize(height=resize_height, width=resize_width, always_apply=True))

    # add center crop transform
    if center_crop is not None:
        if isinstance(center_crop, int):
            crop_height, crop_width = (center_crop, center_crop)
        elif isinstance(image_size, tuple):
            crop_height, crop_width = int(center_crop[0]), int(center_crop[1])
        if crop_height > resize_height or crop_width > resize_width:
            raise ValueError(f"Crop size may not be larger than image size. Found {image_size} and {center_crop}")
        transforms_list.append(A.CenterCrop(height=crop_height, width=crop_width, always_apply=True))

    # CUSTOM - add normalize transform, where p=0.5 means that the transform has a probability of 50% of occuring
    if training:
        transforms_list.append(A.VerticalFlip(p=0.5))
        transforms_list.append(A.HorizontalFlip(p=0.5))
        transforms_list.append(A.RandomBrightnessContrast(p=0.2))
    # CUSTOM - end

    # add normalize transform
    if normalization == 'imagenet':
        transforms_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    elif normalization == 'none':
        transforms_list.append(A.ToFloat(max_value=255))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    # add tensor conversion
    if to_tensor:
        transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)
