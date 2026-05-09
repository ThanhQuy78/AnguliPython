from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from common import NoiseBlob, load_gray_image, read_image_any, sorted_numeric_paths, to_gray_u8


FILTER_ZERO_POINT = 46


def require_dir(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f'{label} is required')
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f'{label} directory not found: {path}')
    return path


def index_densitymaps(density_dir: Path | None, strict_assets: bool) -> List[Path]:
    density_dir = require_dir(density_dir, 'density_dir')
    files = sorted_numeric_paths([p for p in density_dir.iterdir() if p.is_file() and p.suffix.lower() == '.jpeg'])
    if strict_assets and len(files) != 2000:
        raise RuntimeError(f'Expected exactly 2000 density maps (*.jpeg) in {density_dir}, found {len(files)}')
    if len(files) < 2:
        raise RuntimeError(f'Need at least 2 density maps in {density_dir}, found {len(files)}')
    return files


def decode_kernel_image_like_anguli(img_u8: np.ndarray, zero_point: int = FILTER_ZERO_POINT) -> np.ndarray:
    """Best-effort decoding for Anguli filterbank BMP assets.

    Anguli uses integer-valued filters and in the optimized path indexes them as
    filter_value + 46, which strongly suggests the stored kernel values are centered
    around 46. We therefore decode BMP grayscale values by subtracting 46 rather than
    normalizing them like a generic Gabor kernel.
    """
    img100 = np.rint(img_u8.astype(np.float32) * (100.0 / 255.0)).astype(np.int32)
    ker = img100 - zero_point
    ker = np.clip(ker, -46, 54)
    return ker


def load_filterbank_assets(
    filterbank_dir: Path | None,
    strict_assets: bool,
    distnct_o: int,
    filter_zero_point: int = FILTER_ZERO_POINT,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if filterbank_dir is not None and filterbank_dir.is_file() and filterbank_dir.suffix.lower() == '.npz':
        data = np.load(str(filterbank_dir))
        return data['filterbank'], data['filter_sizes'], int(data['max_size'])

    filterbank_dir = require_dir(filterbank_dir, 'filterbank_dir')
    folders = sorted_numeric_paths([p for p in filterbank_dir.iterdir() if p.is_dir()])
    if strict_assets and len(folders) != 100:
        raise RuntimeError(f'Expected 100 filterbank folders in {filterbank_dir}, found {len(folders)}')

    kernels: dict[tuple[int, int], np.ndarray] = {}
    max_size = 0
    for folder in folders:
        if not folder.name.isdigit():
            raise RuntimeError(f'Invalid filterbank folder name: {folder.name}')
        freq_idx = int(folder.name)
        if not (1 <= freq_idx <= 100):
            raise RuntimeError(f'Filterbank folder index out of range 1..100: {folder}')

        files = sorted_numeric_paths([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.bmp'])
        if strict_assets and len(files) != 180:
            raise RuntimeError(f'Expected 180 .bmp files in {folder}, found {len(files)}')

        for file_path in files:
            if not file_path.stem.isdigit():
                raise RuntimeError(f'Invalid filter file name: {file_path.name}')
            orient_idx = int(file_path.stem)
            if not (1 <= orient_idx <= 180):
                raise RuntimeError(f'Filter image index out of range 1..180: {file_path}')
            img_u8 = to_gray_u8(read_image_any(file_path, cv2.IMREAD_UNCHANGED))
            if img_u8.shape[0] != img_u8.shape[1]:
                raise RuntimeError(f'Filter kernel is not square: {file_path} shape={img_u8.shape}')
            size = int(img_u8.shape[0])
            max_size = max(max_size, size)
            kernels[(freq_idx, orient_idx)] = decode_kernel_image_like_anguli(img_u8, zero_point=filter_zero_point)

    filter_sizes = np.zeros((101, distnct_o), dtype=np.int32)
    filterbank = np.zeros((101, distnct_o, max_size, max_size), dtype=np.int32)

    for freq_idx in range(1, 101):
        for orient_idx in range(1, 181):
            key = (freq_idx, orient_idx)
            if key not in kernels:
                raise RuntimeError(f'Missing filterbank kernel for freq={freq_idx}, orient={orient_idx}')
            ker = kernels[key]
            size = int(ker.shape[0])
            filter_sizes[freq_idx, orient_idx - 1] = size
            start = (max_size - size) // 2
            filterbank[freq_idx, orient_idx - 1, start:start + size, start:start + size] = ker

    filter_sizes[0, :] = filter_sizes[1, :]
    filterbank[0, :, :, :] = filterbank[1, :, :, :]
    return filterbank, filter_sizes, max_size


def load_noise_blobs(noise_blob_dir: Path | None, strict_assets: bool) -> List[NoiseBlob]:
    noise_dir = require_dir(noise_blob_dir, 'noise_blob_dir')
    bmp_files = sorted_numeric_paths([p for p in noise_dir.iterdir() if p.is_file() and p.suffix.lower() == '.bmp'])
    png_files = sorted_numeric_paths([p for p in noise_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png'])

    if strict_assets and len(bmp_files) != 5:
        raise RuntimeError(f'Expected exactly 5 .bmp files in {noise_dir}, found {len(bmp_files)}')
    if strict_assets and png_files and len(png_files) not in (5, 10):
        raise RuntimeError(f'Expected 5 or 10 .png files in {noise_dir}, found {len(png_files)}')

    blobs: List[NoiseBlob] = [NoiseBlob(mask=np.zeros((1, 1), dtype=np.uint8), name='dummy0')]
    for bmp_path in bmp_files:
        bmp = to_gray_u8(read_image_any(bmp_path, cv2.IMREAD_UNCHANGED))
        blobs.append(NoiseBlob(mask=bmp, name=bmp_path.stem))
    return blobs


def select_and_merge_densitymap(density_files: List[Path], py_rng, width: int, height: int) -> np.ndarray:
    # Faithful to Anguli: draw freq1 and freq2 independently, then freq3 with 50% chance.
    freq1 = py_rng.choice(density_files)
    freq2 = py_rng.choice(density_files)
    mats = [
        load_gray_image(freq1),
        load_gray_image(freq2),
    ]
    if py_rng.random() > 0.5:
        mats.append(load_gray_image(py_rng.choice(density_files)))

    merged = np.mean(mats, axis=0).astype(np.float32)
    merged = cv2.resize(merged, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(merged, 0.0, 1.0)
