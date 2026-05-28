"""Hierarchical label helpers for Week 12 two-stage damage classification."""

from __future__ import annotations

from torch.utils.data import Dataset

from week11_dataset import CLASS_NAMES, BuildingDamageDataset


STAGE1_CLASS_NAMES = ["no_damage", "damaged"]
STAGE2_CLASS_NAMES = ["minor_damage", "major_damage", "destroyed"]


class Stage1DamageDataset(Dataset):
    """Map four xBD classes to no_damage vs damaged."""

    def __init__(self, base_dataset: BuildingDamageDataset) -> None:
        self.base_dataset = base_dataset
        self.samples = base_dataset.samples

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.base_dataset[index]
        item["original_label"] = int(item["label"])
        item["label"] = 0 if int(item["label"]) == CLASS_NAMES.index("no_damage") else 1
        item["class_name"] = STAGE1_CLASS_NAMES[int(item["label"])]
        return item


class Stage2DamageDataset(Dataset):
    """Keep only damaged samples and remap minor/major/destroyed to 0/1/2."""

    def __init__(self, base_dataset: BuildingDamageDataset) -> None:
        self.base_dataset = base_dataset
        self.indices = [i for i, sample in enumerate(base_dataset.samples) if int(sample["label"]) != CLASS_NAMES.index("no_damage")]
        self.samples = [base_dataset.samples[i] for i in self.indices]
        if not self.indices:
            raise ValueError("Stage 2 dataset contains no damaged samples.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        item = self.base_dataset[self.indices[index]]
        original_label = int(item["label"])
        item["original_label"] = original_label
        item["label"] = original_label - 1
        item["class_name"] = STAGE2_CLASS_NAMES[int(item["label"])]
        return item
