"""Create Week 11 qualitative error-analysis crop panels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset
from week11_model import SiameseBuildingClassifier


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to uint8 RGB."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image.cpu() * std + mean).clamp(0, 1)
    return (image.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)


def save_panel(path: Path, pre: torch.Tensor, post: torch.Tensor, diff: torch.Tensor) -> None:
    """Save a pre/post/diff crop panel."""
    panel = np.concatenate([denormalize(pre), denormalize(post), denormalize(diff)], axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 11: save classifier failure crop panels.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--checkpoint", type=Path, default=Path("results") / "week11" / "checkpoints" / "week11_siamese_resnet18_best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week11" / "error_analysis")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BuildingDamageDataset(args.dataset_root, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SiameseBuildingClassifier(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    saved = 0
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["pre"].to(device), batch["post"].to(device), batch["diff"].to(device))
            probabilities = torch.softmax(logits.cpu(), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            labels = batch["label"]

            for index, building_id in enumerate(batch["building_id"]):
                true_label = int(labels[index])
                predicted_label = int(predictions[index])
                confidence = float(probabilities[index, predicted_label].item())
                correct = true_label == predicted_label
                records.append(
                    {
                        "building_id": building_id,
                        "true_class": CLASS_NAMES[true_label],
                        "predicted_class": CLASS_NAMES[predicted_label],
                        "confidence": confidence,
                        "correct": correct,
                    }
                )
                if not correct and saved < args.max_examples:
                    failure_dir = args.output_dir / f"{CLASS_NAMES[true_label]}_as_{CLASS_NAMES[predicted_label]}"
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    save_panel(failure_dir / f"{building_id}.png", batch["pre"][index], batch["post"][index], batch["diff"][index])
                    with (failure_dir / f"{building_id}.json").open("w", encoding="utf-8") as file:
                        json.dump(records[-1], file, indent=2)
                    saved += 1

    with (args.output_dir / "prediction_records.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["building_id", "true_class", "predicted_class", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(records)
    print(f"saved_failures={saved} records={len(records)} output={args.output_dir}")


if __name__ == "__main__":
    main()
