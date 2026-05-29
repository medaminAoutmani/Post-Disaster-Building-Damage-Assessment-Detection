"""Isolate ConvNeXt no-damage/minor-damage mistakes for Week 13."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
WEEK12_DIR = CURRENT_DIR.parent / "week12"
for path in [WEEK11_DIR, WEEK12_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset
from week12_model_backbones import ObjectDamageRepresentationModel


def load_checkpoint(path: Path, device: torch.device) -> ObjectDamageRepresentationModel:
    checkpoint = torch.load(path, map_location="cpu")
    model = ObjectDamageRepresentationModel(
        backbone=checkpoint.get("backbone", "convnext_tiny"),
        fusion=checkpoint.get("fusion", "gated"),
        embedding_dim=int(checkpoint.get("embedding_dim", 256)),
        num_classes=len(CLASS_NAMES),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Week 13: isolate no_damage/minor_damage CNN mistakes.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings_week8_extra")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-csv", type=Path, default=Path("results") / "week13_topology" / "error_regions.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BuildingDamageDataset(args.dataset_root, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_checkpoint(args.checkpoint, device)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metadata_path", "sample_dir", "label", "class_name", "prediction", "prediction_name", "p_no_damage", "p_minor_damage", "is_target_pair_error"])
        target_pair = {0, 1}
        for batch in dataloader:
            logits = model(batch["pre"].to(device), batch["post"].to(device), batch["diff"].to(device))
            probs = F.softmax(logits, dim=1).cpu()
            predictions = torch.argmax(probs, dim=1)
            labels = batch["label"].long()
            for index in range(len(labels)):
                label = int(labels[index].item())
                prediction = int(predictions[index].item())
                if label not in target_pair and prediction not in target_pair:
                    continue
                sample_dir = str(Path(str(batch["metadata_path"][index])).parent)
                is_pair_error = label in target_pair and prediction in target_pair and label != prediction
                writer.writerow(
                    [
                        str(batch["metadata_path"][index]),
                        sample_dir,
                        label,
                        CLASS_NAMES[label],
                        prediction,
                        CLASS_NAMES[prediction],
                        float(probs[index, 0].item()),
                        float(probs[index, 1].item()),
                        int(is_pair_error),
                    ]
                )


if __name__ == "__main__":
    main()
