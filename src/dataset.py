"""RetinopathyDataset: loads DeepDRiD fundus images and labels from a CSV.

Supports two modes:
  - single-image: one fundus photo per sample.
  - dual-image: two photos per sample (e.g. left/right eye or two fields),
    concatenated channel-wise before being passed to a dual-input model.
"""

import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Return the standard train/eval transform pipeline.

    Training gets augmentation (random crop/flip/jitter); eval gets only
    deterministic resize + normalize so results are reproducible.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class RetinopathyDataset(Dataset):
    """Reads (image_path[, image_path_2], label) rows from a CSV.

    Expected CSV columns:
      - single-image mode: img_path, label
      - dual-image mode:   img_path, img_path_2, label
    `label` may be absent for the test split (predictions only).
    """

    def __init__(self, csv_path: str, image_root: str, transform=None, dual_image: bool = False):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.dual_image = dual_image
        self.has_labels = "label" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, rel_path: str) -> Image.Image:
        return Image.open(os.path.join(self.image_root, rel_path)).convert("RGB")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self._load(row["img_path"])
        if self.transform:
            image = self.transform(image)

        if self.dual_image:
            image2 = self._load(row["img_path_2"])
            if self.transform:
                image2 = self.transform(image2)
            sample = (image, image2)
        else:
            sample = image

        if self.has_labels:
            label = int(row["label"])
            return sample, label
        # test-time: also return an identifier so predictions can be written back out
        identifier = row.get("image_id", row.get("img_path"))
        return sample, identifier
