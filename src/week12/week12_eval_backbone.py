"""Evaluate Week 12 checkpoints and export embedding visualizations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
for path in [CURRENT_DIR, WEEK11_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset
from week11_train_classifier import confusion_matrix, metrics_from_confusion, save_confusion_csv, save_confusion_plot, save_json
from week12_model_backbones import ArcMarginProduct, ObjectDamageRepresentationModel


@torch.no_grad()
def collect_outputs(
    model: ObjectDamageRepresentationModel,
    dataloader: DataLoader,
    device: torch.device,
    arc_head: ArcMarginProduct | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    all_embeddings = []
    all_logits = []
    all_labels = []
    all_paths = []
    for batch in dataloader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        diff = batch["diff"].to(device)
        logits, embeddings = model(pre, post, diff, return_embedding=True)
        if arc_head is not None:
            logits = arc_head.cosine_logits(embeddings)
        all_logits.append(logits.cpu())
        all_embeddings.append(embeddings.cpu())
        all_labels.append(batch["label"].long())
        all_paths.extend(str(path) for path in batch["metadata_path"])
    return torch.cat(all_logits), torch.cat(all_embeddings), torch.cat(all_labels), all_paths


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def maybe_tsne_2d(embeddings: np.ndarray, seed: int) -> np.ndarray | None:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None
    perplexity = min(30, max(5, (len(embeddings) - 1) // 3))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(embeddings)


def maybe_umap_2d(embeddings: np.ndarray, seed: int) -> np.ndarray | None:
    try:
        import umap
    except Exception:
        return None
    return umap.UMAP(n_components=2, random_state=seed).fit_transform(embeddings)


def save_projection(points: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for index, class_name in enumerate(CLASS_NAMES):
        mask = labels == index
        if np.any(mask):
            plt.scatter(points[mask, 0], points[mask, 1], s=9, alpha=0.65, label=class_name, color=colors[index])
    plt.title(title)
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.legend(markerscale=2)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_embedding_csv(embeddings: torch.Tensor, labels: torch.Tensor, metadata_paths: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metadata_path", "label", "class_name", *[f"embedding_{i}" for i in range(embeddings.shape[1])]])
        for vector, label, metadata_path in zip(embeddings.tolist(), labels.tolist(), metadata_paths):
            writer.writerow([metadata_path, label, CLASS_NAMES[label], *vector])


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 12: evaluate object-level representation checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week12_eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-plot-samples", type=int, default=4000)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    backbone = checkpoint.get("backbone", "resnet34")
    fusion = checkpoint.get("fusion", "concat")
    embedding_dim = int(checkpoint.get("embedding_dim", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ObjectDamageRepresentationModel(backbone=backbone, fusion=fusion, embedding_dim=embedding_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
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

    dataset = BuildingDamageDataset(args.dataset_root, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logits, embeddings, labels, metadata_paths = collect_outputs(model, dataloader, device, arc_head)
    predictions = torch.argmax(logits, dim=1)
    confusion = confusion_matrix(predictions, labels, len(CLASS_NAMES))
    metrics = metrics_from_confusion(confusion)

    save_json({"checkpoint": str(args.checkpoint), "split": args.split, **metrics}, args.output_dir / "metrics" / "metrics.json")
    save_confusion_csv(confusion, args.output_dir / "confusion_matrices" / "confusion_matrix.csv")
    save_confusion_plot(confusion, args.output_dir / "confusion_matrices" / "confusion_matrix.png")
    save_embedding_csv(embeddings, labels, metadata_paths, args.output_dir / "embeddings" / f"{args.split}_embeddings.csv")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(labels))
    if len(indices) > args.max_plot_samples:
        indices = rng.choice(indices, size=args.max_plot_samples, replace=False)
    embedding_np = embeddings.numpy()[indices]
    label_np = labels.numpy()[indices]
    save_projection(pca_2d(embedding_np), label_np, "Week 12 embedding PCA", args.output_dir / "embedding_plots" / "pca.png")
    tsne = maybe_tsne_2d(embedding_np, args.seed)
    if tsne is not None:
        save_projection(tsne, label_np, "Week 12 embedding t-SNE", args.output_dir / "embedding_plots" / "tsne.png")
    umap_points = maybe_umap_2d(embedding_np, args.seed)
    if umap_points is not None:
        save_projection(umap_points, label_np, "Week 12 embedding UMAP", args.output_dir / "embedding_plots" / "umap.png")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
