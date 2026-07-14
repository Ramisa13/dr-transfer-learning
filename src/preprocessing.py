"""Fundus-image preprocessing techniques used in Task D's comparison.

Each function takes and returns an RGB numpy array (H, W, 3), uint8, so
they can be composed and dropped into a torchvision-style pipeline via
`transforms.Lambda`.
"""

import cv2
import numpy as np


def circular_crop(img: np.ndarray) -> np.ndarray:
    """Crop to the circular fundus region and zero out the black background."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return img[y : y + h, x : x + w]


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on the L channel (LAB space)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe_op.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def ben_graham(img: np.ndarray, sigma_x: int = 10) -> np.ndarray:
    """Ben Graham's preprocessing: subtract local average color to boost lesion contrast.

    Popularized by the winning solution to the original Kaggle DR competition.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma_x)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def gaussian_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


PREPROCESSORS = {
    "none": lambda img: img,
    "circular_crop": circular_crop,
    "clahe": clahe,
    "ben_graham": ben_graham,
    "gaussian_blur": gaussian_blur,
    "sharpen": sharpen,
}


def apply_preprocessing(img: np.ndarray, method: str) -> np.ndarray:
    if method not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessing method '{method}'. Options: {list(PREPROCESSORS)}")
    return PREPROCESSORS[method](img)
