"""PyTorch dataset for Week 11 building-level damage classification."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from week11_features import extract_feature_vector


CLASS_NAMES = ["no_damage", "minor_damage", "major_damage", "destroyed"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_rgb_tensor(path: Path, normalize: bool = True) -> torch.Tensor:
    """Load an RGB image as a CHW float tensor."""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def apply_train_augmentation(pre: torch.Tensor, post: torch.Tensor, diff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply lightweight geometry and intensity augmentation to aligned crops."""
    if random.random() < 0.5:
        pre = torch.flip(pre, dims=[2])
        post = torch.flip(post, dims=[2])
        diff = torch.flip(diff, dims=[2])
    if random.random() < 0.5:
        pre = torch.flip(pre, dims=[1])
        post = torch.flip(post, dims=[1])
        diff = torch.flip(diff, dims=[1])

    rotations = random.randint(0, 3)
    if rotations:
        pre = torch.rot90(pre, rotations, dims=[1, 2])
        post = torch.rot90(post, rotations, dims=[1, 2])
        diff = torch.rot90(diff, rotations, dims=[1, 2])

    if random.random() < 0.35:
        brightness = random.uniform(-0.08, 0.08)
        contrast = random.uniform(0.9, 1.1)
        pre = torch.clamp((pre - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        post = torch.clamp((post - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        diff = torch.abs(pre - post)
    return pre, post, diff


class BuildingDamageDataset(Dataset):
    """Load object-level xBD building crops saved by week11_extract_buildings.py."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        normalize: bool = True,
        augment: bool = False,
        include_features: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.normalize = normalize
        self.augment = augment
        self.include_features = include_features
        self.samples: list[dict[str, Path | int | str]] = []

        split_root = self.dataset_root / split
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_root = split_root / class_name
            if not class_root.exists():
                continue
            for sample_dir in sorted(path for path in class_root.iterdir() if path.is_dir()):
                required = [sample_dir / "pre.png", sample_dir / "post.png", sample_dir / "diff.png", sample_dir / "metadata.json"]
                if all(path.exists() for path in required):
                    self.samples.append(
                        {
                            "sample_dir": sample_dir,
                            "label": class_index,
                            "class_name": class_name,
                            "building_id": sample_dir.name,
                        }
                    )

        if not self.samples:
            raise ValueError(f"No Week 11 building samples found under {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        sample_dir = Path(sample["sample_dir"])
        with (sample_dir / "metadata.json").open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        pre = load_rgb_tensor(sample_dir / "pre.png", normalize=False)
        post = load_rgb_tensor(sample_dir / "post.png", normalize=False)
        diff = load_rgb_tensor(sample_dir / "diff.png", normalize=False)
        if self.augment:
            pre, post, diff = apply_train_augmentation(pre, post, diff)
        if self.normalize:
            pre = (pre - IMAGENET_MEAN) / IMAGENET_STD
            post = (post - IMAGENET_MEAN) / IMAGENET_STD
            diff = (diff - IMAGENET_MEAN) / IMAGENET_STD
        output = {
            "pre": pre,
            "post": post,
            "diff": diff,
            "label": int(sample["label"]),
            "class_name": str(sample["class_name"]),
            "building_id": str(sample["building_id"]),
            "sample_id": str(metadata.get("sample_id", "")),
            "disaster_type": str(metadata.get("disaster_type", "")),
            "metadata_path": str(sample_dir / "metadata.json"),
        }
        if self.include_features:
            output["features"] = torch.from_numpy(extract_feature_vector(sample_dir)).float()
        return output
