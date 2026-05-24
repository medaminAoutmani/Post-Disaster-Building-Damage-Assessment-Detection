"""Week 7 dataset wrappers that expose pre/post images separately."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from week2_dataset import XBDChangeDataset


class XBDTemporalDamageDataset(Dataset):
    """Wrap the Week 2/6 xBD dataset and split the 6-channel tensor into pre/post tensors."""

    def __init__(
        self,
        data_dir: Path,
        sample_ids: list[str],
        split: str,
        transform,
        target_mode: str = "multiclass",
        filter_empty: bool = True,
    ) -> None:
        self.base_dataset = XBDChangeDataset(
            data_dir=data_dir,
            sample_ids=sample_ids,
            split=split,
            transform=transform,
            target_mode=target_mode,
            filter_empty=filter_empty,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base_dataset[index]
        image = sample["image"]
        return {
            "pre_image": image[:3],
            "post_image": image[3:6],
            "image": image,
            "mask": sample["mask"],
            "sample_id": sample["sample_id"],
        }

