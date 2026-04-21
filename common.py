import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


VALID_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


@dataclass
class ComplexPoint:
    x: float = 0.0
    y: float = 0.0


@dataclass
class NoiseBlob:
    mask: np.ndarray
    name: str


def arg_complex(p: ComplexPoint) -> float:
    return math.atan2(p.y, p.x)


def numeric_stem_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if digits:
        return (0, int(digits), stem)
    return (1, stem.lower())


def sorted_numeric_paths(paths: List[Path]) -> List[Path]:
    return sorted(paths, key=numeric_stem_key)


def read_image_any(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f'Unsupported image shape: {img.shape}')


def load_gray_image(
    path: str | Path,
    size: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    img = to_gray_u8(read_image_any(path, cv2.IMREAD_UNCHANGED))
    if size is not None:
        img = cv2.resize(img, size, interpolation=interpolation)
    return img.astype(np.float32) / 255.0


def normalize_0_100_float(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return 100.0 * (a - mn) / (mx - mn)


def normalize_int_like_anguli(a: np.ndarray) -> np.ndarray:
    """Emulate normalize2Df_print() from Anguli on integer mats.

    - work in integer space as long as possible
    - subtract min if min >= 0, else add abs(min)
    - divide by (max/100.0)
    - truncate toward zero when writing back to int mat
    """
    out = np.asarray(a, dtype=np.int32).copy()
    minval = int(out.min())
    if minval < 0:
        out = out + abs(minval)
    else:
        out = out - minval

    maxval = int(out.max())
    if maxval <= 0:
        return np.zeros_like(out, dtype=np.int32)

    scale = maxval / 100.0
    out = np.trunc(out.astype(np.float64) / scale).astype(np.int32)
    return out


def binarize_int_like_anguli(a: np.ndarray, thresh: int) -> np.ndarray:
    out = np.asarray(a, dtype=np.int32).copy()
    out = np.where(out > thresh, 100, 0).astype(np.int32)
    return out
