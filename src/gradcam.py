"""Task E: training curve plots + Grad-CAM heatmaps for explainability.

Example:
    python src/gradcam.py --config configs/config.yaml \
        --checkpoint ckpts/model_task_c.pth --num_samples 10 --out_dir assets/gradcam
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader

from dataset import RetinopathyDataset, get_transforms
from models import build_model
from utils import get_device, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="assets/gradcam")
    return parser.parse_args()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)


def plot_training_curves(history: dict, out_dir: str) -> None:
    """history expects keys: train_loss, val_loss, train_acc, val_acc (lists per epoch).

    If you logged history during training (see train.py), pass it in here;
    otherwise this function can be skipped and curves plotted directly from
    saved logs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="Training Loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Training Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_gradcam(model, dataset, device, num_samples: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Target the last conv block of the backbone for CAM computation.
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (image, meta) in enumerate(loader):
        if i >= num_samples:
            break
        image = image.to(device)
        grayscale_cam = cam(input_tensor=image)[0]

        rgb_img = denormalize(image[0])
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(visualization)
        axes[1].set_title("Grad-CAM Overlay")
        axes[1].axis("off")

        sample_id = meta if isinstance(meta, str) else str(meta)
        fig.savefig(os.path.join(out_dir, f"gradcam_{i}_{sample_id}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {num_samples} Grad-CAM overlays to {out_dir}")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    model = build_model(config)
    model = load_checkpoint(model, args.checkpoint, device)

    data_cfg = config["data"]
    dataset = RetinopathyDataset(
        csv_path=os.path.join(data_cfg["data_root"], data_cfg["test_csv"]),
        image_root=data_cfg["data_root"],
        transform=get_transforms(data_cfg["image_size"], train=False),
        dual_image=False,  # Grad-CAM here assumes single-image mode
    )

    run_gradcam(model, dataset, device, args.num_samples, args.out_dir)


if __name__ == "__main__":
    main()
