"""Week 2 data pipeline for xBD segmentation training.

Builds train/val/test splits, Albumentations transforms, a PyTorch Dataset, and
DataLoaders. The default target is binary building segmentation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from week1_preprocessing import (
    create_damage_mask,
    image_pair_paths,
    load_image_rgb,
    parse_polygons,
)


VALID_DAMAGE_CLASSES = {"no-damage", "minor-damage", "major-damage", "destroyed"}


def read_label_features_quiet(label_path: Path) -> list[dict]:
    """Read pixel-space label features without emitting per-file warnings."""
    with label_path.open("r", encoding="utf-8") as file:
        label_data = json.load(file)
    return label_data.get("features", {}).get("xy", [])


def sample_ids_from_labels(data_dir: Path, split: str = "train") -> list[str]:
    """Find sample ids with post-disaster labels."""
    labels_dir = data_dir / split / "labels"
    label_paths = sorted(labels_dir.glob("*_post_disaster.json"))
    return [path.stem.replace("_post_disaster", "") for path in label_paths]


def write_split_file(sample_ids: list[str], output_path: Path) -> None:
    """Save one sample id per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")


def read_split_file(split_path: Path) -> list[str]:
    """Read one sample id per line."""
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_train_val_test_splits(
    data_dir: Path,
    output_dir: Path = Path("splits"),
    split: str = "train",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, list[str]]:
    """Create deterministic train/val/test split files from available labels."""
    sample_ids = sample_ids_from_labels(data_dir, split)
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    if max_samples is not None and max_samples > 0:
        sample_ids = sample_ids[:max_samples]

    train_end = int(len(sample_ids) * train_ratio)
    val_end = train_end + int(len(sample_ids) * val_ratio)

    splits = {
        "train": sample_ids[:train_end],
        "val": sample_ids[train_end:val_end],
        "test": sample_ids[val_end:],
    }

    for split_name, ids in splits.items():
        write_split_file(ids, output_dir / f"{split_name}.txt")

    return splits


def get_transforms(image_size: int = 512, train: bool = True, use_week3_augmentation: bool = True) -> A.Compose:
    """Build Albumentations transforms for paired pre/post images and masks."""
    if train:
        transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.4),
        ]
        if use_week3_augmentation:
            transforms.append(
                A.OneOf(
                    [
                        A.Blur(blur_limit=3, p=1.0),
                        A.GaussNoise(p=1.0),
                    ],
                    p=0.25,
                )
            )
        transforms.extend(
            [
                A.Normalize(mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
                ToTensorV2(),
            ]
        )
        return A.Compose(transforms)

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
            ToTensorV2(),
        ]
    )


class XBDChangeDataset(Dataset):
    """xBD paired-image Dataset.

    Input tensor shape is 6 x H x W: pre RGB stacked with post RGB.
    For binary segmentation, target shape is 1 x H x W with building pixels = 1.
    """

    def __init__(
        self,
        data_dir: Path,
        sample_ids: list[str],
        split: str = "train",
        transform: A.Compose | None = None,
        target_mode: str = "binary",
        filter_empty: bool = True,
        verbose: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_mode = target_mode
        self.filter_empty = filter_empty
        self.filter_stats: Counter = Counter()

        if target_mode not in {"binary", "multiclass"}:
            raise ValueError("target_mode must be 'binary' or 'multiclass'")

        self.sample_ids = self._filter_sample_ids(sample_ids) if filter_empty else sample_ids
        if filter_empty and verbose:
            self._print_filter_summary(len(sample_ids))

        if not self.sample_ids:
            raise ValueError("No valid samples remain after filtering empty masks.")

    def _filter_sample_ids(self, sample_ids: list[str]) -> list[str]:
        """Keep only samples with valid labeled polygons that rasterize to mask pixels."""
        valid_sample_ids: list[str] = []

        for sample_id in sample_ids:
            _, post_path, label_path = image_pair_paths(self.data_dir, sample_id, self.split)
            if not post_path.exists() or not label_path.exists():
                self.filter_stats["missing_files"] += 1
                continue

            post_image = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
            if post_image is None:
                self.filter_stats["missing_files"] += 1
                continue

            try:
                features = read_label_features_quiet(label_path)
            except Exception:
                self.filter_stats["invalid_label_json"] += 1
                continue

            if not features:
                self.filter_stats["no_features"] += 1
                continue

            valid_features = [
                feature
                for feature in features
                if feature.get("properties", {}).get("subtype") in VALID_DAMAGE_CLASSES
            ]
            if not valid_features:
                self.filter_stats["no_valid_damage_class"] += 1
                continue

            polygons = parse_polygons(valid_features, warn_invalid=False)
            if not polygons:
                self.filter_stats["invalid_polygons"] += 1
                continue

            mask = create_damage_mask(post_image.shape, polygons, warn_empty=False)
            if int(mask.sum()) == 0:
                self.filter_stats["empty_mask"] += 1
                continue

            valid_sample_ids.append(sample_id)

        return valid_sample_ids

    def _print_filter_summary(self, original_count: int) -> None:
        """Print one compact filtering summary instead of repeated warnings."""
        skipped = original_count - len(self.sample_ids)
        print(
            f"{self.__class__.__name__}: kept {len(self.sample_ids)}/{original_count} samples "
            f"(skipped {skipped})"
        )
        for reason, count in sorted(self.filter_stats.items()):
            print(f"  skipped_{reason}: {count}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[index]
        pre_path, post_path, label_path = image_pair_paths(self.data_dir, sample_id, self.split)

        pre_image = load_image_rgb(pre_path)
        post_image = load_image_rgb(post_path)
        features = read_label_features_quiet(label_path)
        valid_features = [
            feature
            for feature in features
            if feature.get("properties", {}).get("subtype") in VALID_DAMAGE_CLASSES
        ]
        polygons = parse_polygons(valid_features, warn_invalid=False)
        mask = create_damage_mask(post_image.shape, polygons, warn_empty=False)

        if self.target_mode == "binary":
            mask = (mask > 0).astype(np.uint8)

        image = np.concatenate([pre_image, post_image], axis=2)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed["image"].float()
            mask_tensor = transformed["mask"].long()
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()

        if self.target_mode == "binary":
            mask_tensor = mask_tensor.unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample_id,
        }


def build_dataloaders(
    data_dir: Path,
    split_dir: Path = Path("splits"),
    data_split: str = "train",
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
    target_mode: str = "binary",
    filter_empty: bool = True,
    use_week3_augmentation: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    train_ids = read_split_file(split_dir / "train.txt")
    val_ids = read_split_file(split_dir / "val.txt")
    test_ids = read_split_file(split_dir / "test.txt")

    train_dataset = XBDChangeDataset(
        data_dir,
        train_ids,
        split=data_split,
        transform=get_transforms(image_size, train=True, use_week3_augmentation=use_week3_augmentation),
        target_mode=target_mode,
        filter_empty=filter_empty,
    )
    val_dataset = XBDChangeDataset(
        data_dir,
        val_ids,
        split=data_split,
        transform=get_transforms(image_size, train=False),
        target_mode=target_mode,
        filter_empty=filter_empty,
    )
    test_dataset = XBDChangeDataset(
        data_dir,
        test_ids,
        split=data_split,
        transform=get_transforms(image_size, train=False),
        target_mode=target_mode,
        filter_empty=filter_empty,
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Week 2 train/val/test splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Create splits from only this many samples.")
    args = parser.parse_args()

    splits = create_train_val_test_splits(
        args.data_dir,
        args.output_dir,
        args.split,
        args.train_ratio,
        args.val_ratio,
        args.seed,
        args.max_samples,
    )

    for split_name, ids in splits.items():
        print(f"{split_name}: {len(ids)} samples")


if __name__ == "__main__":
    main()
