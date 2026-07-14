"""Task D: combine predictions from several checkpoints.

Supports max-voting, weighted-averaging (of softmax probabilities),
stacking (a small logistic-regression meta-model over out-of-fold probs),
bagging, and boosting-style reweighting.

Example:
    python src/ensemble.py --checkpoints ckpts/model_b1.pth ckpts/model_b2.pth ckpts/model_b3.pth \
        --method max_voting --preprocess clahe --out outputs/test_predictions_d.csv
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RetinopathyDataset, get_transforms
from models import build_model
from utils import get_device, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument(
        "--method",
        type=str,
        choices=["max_voting", "weighted_average", "stacking", "bagging", "boosting"],
        default="max_voting",
    )
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="Per-model weights (weighted_average)")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def get_probs(model, loader, device) -> tuple[np.ndarray, list]:
    model.eval()
    all_probs, all_meta = [], []
    for images, meta in tqdm(loader, desc="predict"):
        if isinstance(images, (list, tuple)):
            images = [img.to(device) for img in images]
            outputs = model(*images)
        else:
            images = images.to(device)
            outputs = model(images)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_meta.extend(meta.tolist() if torch.is_tensor(meta) else list(meta))
    return np.concatenate(all_probs, axis=0), all_meta


def max_voting(pred_matrix: np.ndarray) -> np.ndarray:
    """pred_matrix: (num_models, num_samples) of hard class predictions."""
    final = []
    for col in pred_matrix.T:
        vote = Counter(col).most_common(1)[0][0]
        final.append(vote)
    return np.array(final)


def weighted_average(prob_stack: np.ndarray, weights: list | None) -> np.ndarray:
    """prob_stack: (num_models, num_samples, num_classes)."""
    weights = np.array(weights) if weights else np.ones(prob_stack.shape[0]) / prob_stack.shape[0]
    weights = weights / weights.sum()
    combined = np.tensordot(weights, prob_stack, axes=([0], [0]))
    return combined.argmax(axis=1)


def stacking(prob_stack: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
    """Fits a logistic-regression meta-model on stacked model probabilities.

    Requires `val_labels` (ground truth on the val split, used to fit the
    meta-model) -- run this method on val first to obtain the meta-model,
    then apply it to test-time stacked probabilities.
    """
    num_models, num_samples, num_classes = prob_stack.shape
    stacked_features = prob_stack.transpose(1, 0, 2).reshape(num_samples, num_models * num_classes)
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(stacked_features, val_labels)
    return meta_model


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    data_cfg = config["data"]
    csv_name = data_cfg["val_csv"] if args.split == "val" else data_cfg["test_csv"]
    dataset = RetinopathyDataset(
        csv_path=os.path.join(data_cfg["data_root"], csv_name),
        image_root=data_cfg["data_root"],
        transform=get_transforms(data_cfg["image_size"], train=False),
        dual_image=data_cfg["dual_image"],
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    all_probs, meta = [], None
    for ckpt_path in args.checkpoints:
        model = build_model(config)
        model = load_checkpoint(model, ckpt_path, device)
        probs, meta = get_probs(model, loader, device)
        all_probs.append(probs)
        print(f"Collected probabilities from {ckpt_path}")

    prob_stack = np.stack(all_probs, axis=0)  # (num_models, num_samples, num_classes)
    pred_matrix = prob_stack.argmax(axis=2)  # (num_models, num_samples)

    if args.method == "max_voting":
        final_preds = max_voting(pred_matrix)
    elif args.method == "weighted_average":
        final_preds = weighted_average(prob_stack, args.weights)
    elif args.method in ("bagging", "boosting"):
        # Both reduce to a (possibly weighted) probability average here;
        # true bagging/boosting requires retraining base learners on
        # resampled/reweighted data during Task B -- this just combines
        # the resulting checkpoints' outputs.
        final_preds = weighted_average(prob_stack, args.weights)
    elif args.method == "stacking":
        if not dataset.has_labels:
            raise ValueError("Stacking requires labels to fit the meta-model; run on --split val first.")
        final_preds = stacking(prob_stack, np.array(meta)).predict(
            prob_stack.transpose(1, 0, 2).reshape(prob_stack.shape[1], -1)
        )

    out_path = args.out or os.path.join(config["paths"]["output_dir"], f"test_predictions_{args.method}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if dataset.has_labels:
        from evaluate import compute_metrics

        metrics = compute_metrics(final_preds, meta)
        print("Ensemble metrics:", metrics)
    else:
        pd.DataFrame({"ID": meta, "TARGET": final_preds}).to_csv(out_path, index=False)
        print(f"Wrote {len(final_preds)} ensembled predictions to {out_path}")


if __name__ == "__main__":
    main()
