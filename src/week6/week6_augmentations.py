"""Advanced Week 6 augmentations for satellite damage segmentation."""

from __future__ import annotations

import inspect
import random

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIX_CHANNEL_MEAN = IMAGENET_MEAN + IMAGENET_MEAN
SIX_CHANNEL_STD = IMAGENET_STD + IMAGENET_STD


class RandomCloudSmoke(A.ImageOnlyTransform):
    """Add soft gray-white haze blobs to both pre/post RGB halves."""

    def __init__(self, intensity: tuple[float, float] = (0.08, 0.22), always_apply: bool = False, p: float = 0.25) -> None:
        super().__init__(p=p)
        self.intensity = intensity

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        height, width = image.shape[:2]
        haze = np.zeros((height, width), dtype=np.float32)
        for _ in range(random.randint(3, 7)):
            center = (random.randint(0, width - 1), random.randint(0, height - 1))
            radius = random.randint(max(8, width // 16), max(16, width // 5))
            cv2.circle(haze, center, radius, random.uniform(*self.intensity), thickness=-1)
        haze = cv2.GaussianBlur(haze, (0, 0), sigmaX=max(width, height) / 18)
        output = image.astype(np.float32) / 255.0
        output = output * (1.0 - haze[..., None]) + haze[..., None]
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)


class RandomDisasterIntensityShift(A.ImageOnlyTransform):
    """Apply independent post-disaster brightness/contrast shifts."""

    def __init__(self, brightness_limit: float = 0.12, contrast_limit: float = 0.15, always_apply: bool = False, p: float = 0.35) -> None:
        super().__init__(p=p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        output = image.astype(np.float32)
        if output.shape[2] < 6:
            return image
        brightness = random.uniform(-self.brightness_limit, self.brightness_limit) * 255.0
        contrast = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
        output[:, :, 3:6] = np.clip(output[:, :, 3:6] * contrast + brightness, 0, 255)
        return output.astype(np.uint8)


def _optional_transform(factory):
    try:
        return factory()
    except Exception:
        return None


def build_coarse_dropout(p: float = 0.2):
    """Support Albumentations 1.x and 2.x CoarseDropout argument names."""
    parameters = inspect.signature(A.CoarseDropout).parameters
    if "num_holes_range" in parameters:
        return A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 48),
            hole_width_range=(8, 48),
            p=p,
        )
    return A.CoarseDropout(max_holes=8, max_height=48, max_width=48, p=p)


def get_week6_transforms(image_size: int = 512, train: bool = True, advanced: bool = True) -> A.Compose:
    """Build stronger Week 6 transforms while preserving 6-channel pre/post inputs."""
    transforms: list[A.BasicTransform] = [A.Resize(image_size, image_size)]
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.82, 1.18), translate_percent=(-0.06, 0.06), rotate=(-25, 25), p=0.5),
                A.RandomBrightnessContrast(p=0.35),
            ]
        )
        if advanced:
            optional = [
                _optional_transform(lambda: A.GaussNoise(p=0.25)),
                _optional_transform(lambda: A.MotionBlur(blur_limit=5, p=0.15)),
                _optional_transform(lambda: A.GridDistortion(num_steps=5, distort_limit=0.08, p=0.15)),
                _optional_transform(lambda: build_coarse_dropout(p=0.2)),
                RandomDisasterIntensityShift(p=0.35),
                RandomCloudSmoke(p=0.2),
            ]
            transforms.extend(transform for transform in optional if transform is not None)
    transforms.extend([A.Normalize(mean=SIX_CHANNEL_MEAN, std=SIX_CHANNEL_STD), ToTensorV2()])
    return A.Compose(transforms)
