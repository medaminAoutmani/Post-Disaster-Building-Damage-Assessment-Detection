"""Train/evaluate a transformer text classifier for Week 14 CrisisMMD tasks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CrisisTextDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["tweet_text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }


class FocalLoss(nn.Module):
    def __init__(self, weights: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weights = weights
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = None if self.weights is None else self.weights.to(device=logits.device, dtype=logits.dtype)
        ce = nn.functional.cross_entropy(logits, labels, weight=weights, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def read_rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [row for row in csv.DictReader(file) if row["split"] == split]


def confusion_matrix(predictions: list[int], labels: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for pred, label in zip(predictions, labels):
        matrix[label][pred] += 1
    return matrix


def metrics_from_confusion(matrix: list[list[int]]) -> dict[str, float]:
    num_classes = len(matrix)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(num_classes))
    f1_values = []
    weighted_f1 = 0.0
    for index in range(num_classes):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(num_classes) if row != index)
        fn = sum(matrix[index][col] for col in range(num_classes) if col != index)
        support = sum(matrix[index])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        weighted_f1 += f1 * support
    return {
        "accuracy": correct / max(total, 1),
        "macro_f1": sum(f1_values) / max(num_classes, 1),
        "weighted_f1": weighted_f1 / max(total, 1),
    }


def class_weights(rows: list[dict[str, str]], num_classes: int) -> torch.Tensor:
    counts = Counter(int(row["label_id"]) for row in rows)
    total = sum(counts.values())
    return torch.tensor([total / max(num_classes * counts.get(index, 1), 1) for index in range(num_classes)], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device, num_classes: int) -> tuple[dict[str, float], list[list[int]]]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    for batch in dataloader:
        outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        predictions.extend(torch.argmax(outputs.logits.cpu(), dim=1).tolist())
        labels.extend(batch["labels"].tolist())
    matrix = confusion_matrix(predictions, labels, num_classes)
    return metrics_from_confusion(matrix), matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14: train transformer text classifier.")
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week14_crisismmd" / "model")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--use-fast-tokenizer", action="store_true", help="Use the fast tokenizer when the local environment supports conversion.")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers and sentencepiece to train DeBERTa: pip install transformers sentencepiece") from exc

    train_rows = read_rows(args.task_csv, "train")
    val_rows = read_rows(args.task_csv, "val")
    if not train_rows or not val_rows:
        raise ValueError("Training requires non-empty train and val splits.")
    num_classes = max(int(row["label_id"]) for row in train_rows + val_rows) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.use_fast_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_classes).to(device).float()
    train_loader = DataLoader(CrisisTextDataset(train_rows, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CrisisTextDataset(val_rows, tokenizer, args.max_length), batch_size=args.batch_size)
    weights = class_weights(train_rows, num_classes).to(device) if args.loss in {"weighted_ce", "focal"} else None
    criterion: nn.Module = FocalLoss(weights) if args.loss == "focal" else nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            loss = criterion(outputs.logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        metrics, _ = evaluate(model, val_loader, device, num_classes)
        history.append({"epoch": epoch, "train_loss": total_loss / max(len(train_loader), 1), **metrics})
        print(history[-1])

    val_metrics, val_confusion = evaluate(model, val_loader, device, num_classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "checkpoint")
    tokenizer.save_pretrained(args.output_dir / "checkpoint")
    (args.output_dir / "metrics.json").write_text(json.dumps({"val": val_metrics, "history": history}, indent=2), encoding="utf-8")
    (args.output_dir / "confusion_matrix.json").write_text(json.dumps(val_confusion, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
