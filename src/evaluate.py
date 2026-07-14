"""Computes accuracy/precision/recall/F1/kappa on val or test, and writes
a Kaggle-style `ID,TARGET` prediction CSV for the test split.

Example:
    python src/evaluate.py --config configs/config.yaml \
        --checkpoint ckpts/model_task_b.pth --split test --out outputs/test_predictions_b.csv
"""

import argparse
import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RetinopathyDataset, get_transforms
from models import build_model
from utils import get_device, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--out", type=str, default=None, help="Path to write prediction CSV (test split only)")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, has_labels: bool):
    model.eval()
    all_preds, all_labels, all_ids = [], [], []

    for images, meta in tqdm(loader, desc="evaluate"):
        if isinstance(images, (list, tuple)):
            images = [img.to(device) for img in images]
            outputs = model(*images)
        else:
            images = images.to(device)
            outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)

        if has_labels:
            all_labels.extend(meta.tolist())
        else:
            all_ids.extend(meta)

    return all_preds, all_labels, all_ids


def compute_metrics(preds, labels) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(labels, preds, weights="quadratic"),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    model = build_model(config)
    model = load_checkpoint(model, args.checkpoint, device)

    data_cfg = config["data"]
    csv_name = data_cfg["val_csv"] if args.split == "val" else data_cfg["test_csv"]
    dataset = RetinopathyDataset(
        csv_path=os.path.join(data_cfg["data_root"], csv_name),
        image_root=data_cfg["data_root"],
        transform=get_transforms(data_cfg["image_size"], train=False),
        dual_image=data_cfg["dual_image"],
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    preds, labels, ids = evaluate(model, loader, device, has_labels=dataset.has_labels)

    if dataset.has_labels:
        metrics = compute_metrics(preds, labels)
        print("Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
    else:
        out_path = args.out or os.path.join(config["paths"]["output_dir"], "test_predictions.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame({"ID": ids, "TARGET": preds}).to_csv(out_path, index=False)
        print(f"Wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()
