"""Week 9 multi-task xBD dataset for Siamese damage assessment."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from week1_preprocessing import create_damage_mask, image_pair_paths, load_image_rgb, parse_polygons
from week2_dataset import VALID_DAMAGE_CLASSES, read_label_features_quiet


def _valid_damage_features(label_path: Path) -> list[dict]:
    features = read_label_features_quiet(label_path)
    return [
        feature
        for feature in features
        if feature.get("properties", {}).get("subtype") in VALID_DAMAGE_CLASSES
    ]


def _read_features_or_empty(label_path: Path) -> list[dict]:
    if not label_path.exists():
        return []
    try:
        with label_path.open("r", encoding="utf-8") as handle:
            label_data = json.load(handle)
    except Exception:
        return []
    return label_data.get("features", {}).get("xy", [])


def _building_mask(image_shape: tuple[int, int, int], features: list[dict]) -> np.ndarray:
    valid_features = [
        feature
        for feature in features
        if feature.get("properties", {}).get("subtype") in VALID_DAMAGE_CLASSES
    ]
    polygons = parse_polygons(valid_features, warn_invalid=False)
    damage_mask = create_damage_mask(image_shape, polygons, warn_empty=False)
    return (damage_mask > 0).astype(np.uint8)


class XBDMultiTaskSampleDataset(Dataset):
    """Return pre/post images plus pre/post building masks and damage mask."""

    def __init__(
        self,
        data_dir: Path,
        sample_ids: list[str],
        split: str = "train",
        transform=None,
        filter_empty: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sample_ids = self._filter_sample_ids(sample_ids) if filter_empty else sample_ids
        if not self.sample_ids:
            raise ValueError(f"No valid Week 9 samples found in {data_dir}/{split}")

    def _filter_sample_ids(self, sample_ids: list[str]) -> list[str]:
        valid = []
        for sample_id in sample_ids:
            _, post_path, post_label_path = image_pair_paths(self.data_dir, sample_id, self.split)
            if not post_path.exists() or not post_label_path.exists():
                continue
            post_image = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
            if post_image is None:
                continue
            features = _valid_damage_features(post_label_path)
            if not features:
                continue
            polygons = parse_polygons(features, warn_invalid=False)
            if not polygons:
                continue
            damage_mask = create_damage_mask(post_image.shape, polygons, warn_empty=False)
            if int(damage_mask.sum()) == 0:
                continue
            valid.append(sample_id)
        return valid

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[index]
        pre_path, post_path, post_label_path = image_pair_paths(self.data_dir, sample_id, self.split)
        pre_label_path = self.data_dir / self.split / "labels" / f"{sample_id}_pre_disaster.json"

        pre_image = load_image_rgb(pre_path)
        post_image = load_image_rgb(post_path)
        post_features = _valid_damage_features(post_label_path)
        pre_features = _read_features_or_empty(pre_label_path)

        damage_polygons = parse_polygons(post_features, warn_invalid=False)
        damage_mask = create_damage_mask(post_image.shape, damage_polygons, warn_empty=False)
        post_building_mask = (damage_mask > 0).astype(np.uint8)
        pre_building_mask = _building_mask(pre_image.shape, pre_features)
        if int(pre_building_mask.sum()) == 0:
            pre_building_mask = post_building_mask.copy()

        image = np.concatenate([pre_image, post_image], axis=2)
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                masks=[pre_building_mask, post_building_mask, damage_mask],
            )
            image_tensor = transformed["image"].float()
            masks = transformed["masks"]
            pre_building_tensor = torch.as_tensor(masks[0]).long()
            post_building_tensor = torch.as_tensor(masks[1]).long()
            damage_tensor = torch.as_tensor(masks[2]).long()
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            pre_building_tensor = torch.from_numpy(pre_building_mask).long()
            post_building_tensor = torch.from_numpy(post_building_mask).long()
            damage_tensor = torch.from_numpy(damage_mask).long()

        return {
            "pre_image": image_tensor[:3],
            "post_image": image_tensor[3:6],
            "pre_building_mask": pre_building_tensor,
            "post_building_mask": post_building_tensor,
            "damage_mask": damage_tensor,
            "mask": damage_tensor,
            "sample_id": sample_id,
        }


class XBDMultiTaskCombinedDataset(Dataset):
    """Concatenate old training data with selected Week 8 extra samples."""

    def __init__(self, datasets: list[XBDMultiTaskSampleDataset]) -> None:
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative = np.cumsum(self.lengths).tolist()
        if not self.datasets:
            raise ValueError("No datasets were provided for Week 9 training.")

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        for dataset_index, end in enumerate(self.cumulative):
            start = 0 if dataset_index == 0 else self.cumulative[dataset_index - 1]
            if index < end:
                return self.datasets[dataset_index][index - start]
        raise IndexError(index)
