import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

# ---------------------------- transform for segmentation ---------------------------- #
class SegmentationTrainingTransform:
    """Transformation pipeline for training segmentation models.
    Note for some operations, we only transform the image, but not the mask.
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.min_image_size = 200
        self.max_image_size = 240

        basic_transforms = [
            RandomResizing(self.min_image_size, self.max_image_size), 
            RandomHorizontalFlipping(0.5), 
            RandomVerticalFlipping(0.5),
        ]

        advanced_transforms = [
            RandomCropping(),
            ImageToTensorConversion(),
            ImageDtypeConversion(torch.float),
            ImageNormalization(mean, std)
        ]
        self.transforms = TransformComposer(basic_transforms + advanced_transforms)

    def __call__(self, images, target):
        return self.transforms(images, target)


class SegmentationValidationTransform:
    """Transformation pipeline for validating segmentation models."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        validation_transforms = [
            ImageToTensorConversion(),
            ImageDtypeConversion(torch.float),
            ImageNormalization(mean, std)
        ]
        self.transforms = TransformComposer(validation_transforms)

    def __call__(self, images, target):
        return self.transforms(images, target)

# ---------------------------- transform for SSL ---------------------------- #
class SSLTrainingTransform:
    """Transformation pipeline for training segmentation models.
    Note here we transform both the image and the mask."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.min_image_size = 200
        self.max_image_size = 240

        basic_transforms = [
            RandomResizing_SSL(self.min_image_size, self.max_image_size), 
            RandomHorizontalFlipping_SSL(0.5), 
            RandomVerticalFlipping_SSL(0.5),
        ]

        advanced_transforms = [
            RandomCropping_SSL(),
            ImageToTensorConversion_SSL(),
            ImageDtypeConversion_SSL(torch.float),
            ImageNormalization_SSL(mean, std)
        ]
        self.transforms = TransformComposer(basic_transforms + advanced_transforms)

    def __call__(self, images, target):
        return self.transforms(images, target)

class SSLValidationTransform:
    """Transformation pipeline for validating segmentation models."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        validation_transforms = [
            ImageToTensorConversion_SSL(),
            ImageDtypeConversion_SSL(torch.float),
            ImageNormalization_SSL(mean, std)
        ]
        self.transforms = TransformComposer(validation_transforms)

    def __call__(self, images, target):
        return self.transforms(images, target)

# ---------------------------- auxiliary compose transformation functions ---------------------------- #
class TransformComposer:
    """Composes multiple image transformations."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, target):
        for transform in self.transforms:
            images, target = transform(images, target)
        return images, target

# ---------------------------- aux for segmentation ---------------------------- #

class RandomResizing:
    """Randomly resizes images."""
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = max_size if max_size else min_size

    def __call__(self, images, target):
        new_size = random.randint(self.min_size, self.max_size)
        resized_images = [F.resize(image, new_size) for image in images]
        resized_target = F.resize(target, new_size, interpolation=transforms.InterpolationMode.NEAREST)
        return resized_images, resized_target


class RandomHorizontalFlipping:
    """Randomly flips images horizontally."""
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images, target):
        if random.random() < self.flip_prob:
            images = [F.hflip(image) for image in images]
            target = F.hflip(target)
        return images, target


class RandomVerticalFlipping:
    """Randomly flips images vertically."""
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images, target):
        if random.random() < self.flip_prob:
            images = [F.vflip(image) for image in images]
            target = F.vflip(target)
        return images, target


class RandomCropping:
    """Applies random cropping to images."""
    def __init__(self, height=180, width=220):
        self.height = height
        self.width = width

    def __call__(self, images, target):
        crop_params = transforms.RandomCrop.get_params(images[0], (self.height, self.width))
        cropped_images = [F.crop(image, *crop_params) for image in images]
        cropped_target = F.crop(target, *crop_params)
        return cropped_images, cropped_target


class ImageToTensorConversion:
    """Converts PIL images to tensors."""
    def __call__(self, images, target):
        tensor_images = [F.pil_to_tensor(image) for image in images]
        tensor_target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return tensor_images, tensor_target


class ImageDtypeConversion:
    """Converts image data type."""
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, images, target):
        converted_images = [F.convert_image_dtype(image, self.dtype) for image in images]
        return converted_images, target


class ImageNormalization:
    """Normalizes images using mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, target):
        normalized_images = [torch.clamp(F.normalize(image, mean=self.mean, std=self.std), min=-1, max=1) for image in images]
        # normalized_images = [F.normalize(image, mean=self.mean, std=self.std) for image in images]
        return normalized_images, target

# ---------------------------- aux for SSL ---------------------------- #

class RandomResizing_SSL:
    """Randomly resizes images."""
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = max_size if max_size else min_size

    def __call__(self, images, target):
        new_size = random.randint(self.min_size, self.max_size)
        resized_images = [F.resize(image, new_size) for image in images]
        resized_target = F.resize(target, new_size)
        return resized_images, resized_target


class RandomHorizontalFlipping_SSL:
    """Randomly flips images horizontally."""
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images, target):
        if random.random() < self.flip_prob:
            images = [F.hflip(image) for image in images]
            target = F.hflip(target)
        return images, target


class RandomVerticalFlipping_SSL:
    """Randomly flips images vertically."""
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images, target):
        if random.random() < self.flip_prob:
            images = [F.vflip(image) for image in images]
            target = F.vflip(target)
        return images, target


class RandomCropping_SSL:
    """Applies random cropping to images."""
    def __init__(self, height=180, width=220):
        self.height = height
        self.width = width

    def __call__(self, images, target):
        crop_params = transforms.RandomCrop.get_params(images[0], (self.height, self.width))
        cropped_images = [F.crop(image, *crop_params) for image in images]
        cropped_target = F.crop(target, *crop_params)
        return cropped_images, cropped_target


class ImageToTensorConversion_SSL:
    """Converts PIL images to tensors."""
    def __call__(self, images, target):
        tensor_images = [F.pil_to_tensor(image) for image in images]
        tensor_target = F.pil_to_tensor(target)
        return tensor_images, tensor_target


class ImageDtypeConversion_SSL:
    """Converts image data type."""
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, images, target):
        converted_images = [F.convert_image_dtype(image, self.dtype) for image in images]
        converted_target = F.convert_image_dtype(target, self.dtype)
        return converted_images, converted_target


class ImageNormalization_SSL:
    """Normalizes images using mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, target):
        # normalized_images = [F.normalize(image, mean=self.mean, std=self.std) for image in images]
        # normalized_target = F.normalize(target, mean=self.mean, std=self.std)
        
        normalized_images = [torch.clamp(F.normalize(image, mean=self.mean, std=self.std), min=-1, max=1) for image in images]
        normalized_target = torch.clamp(F.normalize(target, mean=self.mean, std=self.std), min=-1, max=1)
    
        return normalized_images, normalized_target