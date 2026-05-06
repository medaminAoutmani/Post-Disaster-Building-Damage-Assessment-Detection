"""Train the Week 2 binary U-Net baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from week2_dataset import build_dataloaders
from week2_model import UNet


def limit_dataloader(dataloader: DataLoader, max_samples: int | None, shuffle: bool) -> DataLoader:
    """Return a DataLoader backed by only the first max_samples items."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataloader.dataset):
        return dataloader

    subset = Subset(dataloader.dataset, list(range(max_samples)))
    return DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=shuffle,
        num_workers=dataloader.num_workers,
        pin_memory=getattr(dataloader, "pin_memory", False),
    )


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    """Compute binary Dice score from raw logits."""
    predictions = (torch.sigmoid(logits) > 0.5).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.mean().item())


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and Dice."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_dice += dice_score_from_logits(logits.detach(), masks)

    return total_loss / len(dataloader), total_dice / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate and return average loss and Dice."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += float(loss.item())
        total_dice += dice_score_from_logits(logits, masks)

    return total_loss / len(dataloader), total_dice / len(dataloader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Week 2 binary U-Net baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Use only this many training samples.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Use only this many validation samples.")
    parser.add_argument("--small-model", action="store_true", help="Use a smaller U-Net for faster CPU experiments.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs") / "checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_mode="binary",
        use_week3_augmentation=False,
    )
    train_loader = limit_dataloader(train_loader, args.max_train_samples, shuffle=True)
    val_loader = limit_dataloader(val_loader, args.max_val_samples, shuffle=False)

    features = (16, 32, 64) if args.small_model else (32, 64, 128, 256)
    model = UNet(in_channels=6, out_channels=1, features=features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0
    print(f"device={device}")
    print(f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                },
                args.checkpoint_dir / "week2_unet_binary_best.pt",
            )


if __name__ == "__main__":
    main()
