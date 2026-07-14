"""Training entrypoint for Tasks A (baseline fine-tune), B (two-stage
training), and C (self-attention). The same loop handles all three; the
differences are which flags/config fields are set.

Example:
    python src/train.py --config configs/config.yaml --task a --freeze_backbone
    python src/train.py --config configs/config.yaml --task b --stage2_data deepdrid --unfreeze_all
    python src/train.py --config configs/config.yaml --task c --use_attention
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RetinopathyDataset, get_transforms
from models import build_model
from utils import get_device, load_config, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--task", type=str, choices=["a", "b", "c"], required=True)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_all", action="store_true")
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--stage1_data", type=str, default=None, help="Auxiliary dataset root for task B stage 1")
    parser.add_argument("--stage2_data", type=str, default=None, help="Target dataset root for task B stage 2")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Warm-start weights (e.g. stage-1 output)")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    return parser.parse_args()


def build_dataloaders(config: dict, data_root: str) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_ds = RetinopathyDataset(
        csv_path=os.path.join(data_root, data_cfg["train_csv"]),
        image_root=data_root,
        transform=get_transforms(data_cfg["image_size"], train=True),
        dual_image=data_cfg["dual_image"],
    )
    val_ds = RetinopathyDataset(
        csv_path=os.path.join(data_root, data_cfg["val_csv"]),
        image_root=data_root,
        transform=get_transforms(data_cfg["image_size"], train=False),
        dual_image=data_cfg["dual_image"],
    )
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            if isinstance(images, (list, tuple)):
                images = [img.to(device) for img in images]
                outputs = model(*images)
            else:
                images = images.to(device)
                outputs = model(images)
            labels = labels.to(device)

            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, config, device, checkpoint_path: str):
    train_cfg = config["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    if train_cfg["lr_scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["num_epochs"])
    elif train_cfg["lr_scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        scheduler = None

    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(train_cfg["num_epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{train_cfg['num_epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for "
                      f"{train_cfg['early_stopping_patience']} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, checkpoint_path, extra={"best_val_acc": best_val_acc})
    print(f"Saved best checkpoint (val_acc={best_val_acc:.4f}) to {checkpoint_path}")
    return model


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device()

    # Apply CLI overrides on top of the config file.
    if args.use_attention:
        config["model"]["use_attention"] = True
    if args.freeze_backbone:
        config["model"]["freeze_backbone"] = True
    if args.unfreeze_all:
        config["model"]["freeze_backbone"] = False
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs

    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)

    if args.task == "b" and args.stage1_data:
        # Stage 1: pretrain on the auxiliary dataset.
        print(f"=== Task B, Stage 1: training on {args.stage1_data} ===")
        stage1_config = copy.deepcopy(config)
        model = build_model(stage1_config).to(device)
        train_loader, val_loader = build_dataloaders(stage1_config, args.stage1_data)
        stage1_ckpt = os.path.join(config["paths"]["checkpoint_dir"], "stage1.pth")
        model = train_model(model, train_loader, val_loader, stage1_config, device, stage1_ckpt)
        args.init_checkpoint = stage1_ckpt

        print(f"=== Task B, Stage 2: fine-tuning on {args.stage2_data or config['data']['data_root']} ===")
        data_root = args.stage2_data or config["data"]["data_root"]
    else:
        data_root = config["data"]["data_root"]

    model = build_model(config).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Warm-started from {args.init_checkpoint}")

    train_loader, val_loader = build_dataloaders(config, data_root)
    final_ckpt = os.path.join(config["paths"]["checkpoint_dir"], f"model_task_{args.task}.pth")
    train_model(model, train_loader, val_loader, config, device, final_ckpt)


if __name__ == "__main__":
    main()
