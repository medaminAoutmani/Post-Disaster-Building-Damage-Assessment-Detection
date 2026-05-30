"""Export predicted xBD building damage counts from the preferred Week 12 model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
for path in [CURRENT_DIR, WEEK11_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset
from week12_model_backbones import ArcMarginProduct, ObjectDamageRepresentationModel


PREFERRED_WEEK12_CHECKPOINT = (
    Path("results")
    / "week12"
    / "week12_convnext_tiny_gated_effective_no_sampler"
    / "checkpoints"
    / "week12_convnext_tiny_gated_ce_best.pt"
)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ObjectDamageRepresentationModel, ArcMarginProduct | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone = checkpoint.get("backbone", "convnext_tiny")
    fusion = checkpoint.get("fusion", "gated")
    embedding_dim = int(checkpoint.get("embedding_dim", 256))
    model = ObjectDamageRepresentationModel(backbone=backbone, fusion=fusion, embedding_dim=embedding_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    arc_head = None
    if checkpoint.get("arc_head_state_dict") is not None:
        arc_head = ArcMarginProduct(
            embedding_dim,
            len(CLASS_NAMES),
            scale=float(checkpoint.get("arcface_scale", 30.0)),
            margin=float(checkpoint.get("arcface_margin", 0.3)),
        ).to(device)
        arc_head.load_state_dict(checkpoint["arc_head_state_dict"])
        arc_head.eval()
    return model, arc_head


def keep_sample(batch: dict[str, Any], index: int, sample_filter: str | None) -> bool:
    if sample_filter is None:
        return True
    fields = [
        str(batch.get("sample_id", [""])[index]),
        str(batch.get("disaster_type", [""])[index]),
        str(batch.get("metadata_path", [""])[index]),
    ]
    return any(sample_filter.lower() in field.lower() for field in fields)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Week 12: export satellite damage counts for Week 15 fusion.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings_week8_extra")
    parser.add_argument("--checkpoint", type=Path, default=PREFERRED_WEEK12_CHECKPOINT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--sample-filter", help="Optional substring filter over sample_id, disaster_type, or metadata path.")
    parser.add_argument("--output-json", type=Path, default=Path("results") / "week15_inputs" / "satellite.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, arc_head = load_model(args.checkpoint, device)
    dataset = BuildingDamageDataset(args.dataset_root, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    counts: Counter[str] = Counter()
    confidence_sum: defaultdict[str, float] = defaultdict(float)
    total_confidence = 0.0
    total = 0
    for batch in dataloader:
        logits, embeddings = model(batch["pre"].to(device), batch["post"].to(device), batch["diff"].to(device), return_embedding=True)
        if arc_head is not None:
            logits = arc_head.cosine_logits(embeddings)
        probabilities = F.softmax(logits.cpu(), dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        max_probabilities = torch.max(probabilities, dim=1).values
        for index, prediction in enumerate(predictions.tolist()):
            if not keep_sample(batch, index, args.sample_filter):
                continue
            class_name = CLASS_NAMES[prediction]
            confidence = float(max_probabilities[index].item())
            counts[class_name] += 1
            confidence_sum[class_name] += confidence
            total_confidence += confidence
            total += 1

    damaged_total = counts["minor_damage"] + counts["major_damage"] + counts["destroyed"]
    output = {
        "source": "week12_preferred_convnext_tiny_gated",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "sample_filter": args.sample_filter,
        "total_buildings": total,
        "no_damage": counts["no_damage"],
        "minor": counts["minor_damage"],
        "major": counts["major_damage"],
        "destroyed": counts["destroyed"],
        "total_damaged": damaged_total,
        "confidence": total_confidence / max(total, 1),
        "class_confidence": {
            class_name: confidence_sum[class_name] / max(counts[class_name], 1)
            for class_name in CLASS_NAMES
            if counts[class_name] > 0
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
