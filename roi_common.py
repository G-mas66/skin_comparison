"""Shared ROI utilities for the MR ROI benchmark routes (Step 11-19).

Implements the fixed rules of ``TRAINING_PLAN.md`` ("MR ROI 实验扩展路线"):

- polygon coordinate conversion from the original 3448 x 4600 labelme JSON
  space to the 384 x 384 MR input space (x*288/3448 + 48, y*384/4600);
- pixel-center rasterization of all ``red`` polygons into a binary mask;
- the joint bounding box of all ``red`` polygons and its 20 % context box;
- per-fold training Mean Fill (computed on 0-255 MR pixels before normalize);
- dataset variants: ROI Mask, Background Only, ROI Crop, ROI + 20 % Context.

Geometric augmentation (flip / rotation) is applied to image and mask with the
same random decision; brightness / contrast are applied to the image only.
Evaluation folds never use random augmentation.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from dataset import (
    DEFAULT_AUGMENTATION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    ResizePad,
    _augmentation_values,
)

ORIG_WIDTH = 3448
ORIG_HEIGHT = 4600
RESIZE_W = 288
RESIZE_H = 384
PAD_X = 48
CONTEXT_RATIO = 0.2
ROI_LABEL = "red"

SOFT_LABEL_MATRIX = np.array(
    [
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.8, 0.1, 0.0, 0.0],
        [0.0, 0.1, 0.8, 0.1, 0.0],
        [0.0, 0.0, 0.1, 0.8, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.9],
    ],
    dtype=np.float32,
)


def to_384(x: float, y: float) -> tuple[float, float]:
    """Map one original-space vertex to the 384 x 384 MR input space."""
    return x * RESIZE_W / ORIG_WIDTH + PAD_X, y * RESIZE_H / ORIG_HEIGHT


@dataclass
class ROIRecord:
    sample_id: str
    image_path: Path
    polygons: list[list[tuple[float, float]]]
    n_red_polygons: int
    out_of_range_vertices: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive-exclusive style)


def load_roi_polygons(json_path: str | Path) -> tuple[list[list[tuple[float, float]]], str, int]:
    """Return (red polygons in 384 space, imagePath, out-of-range vertex count)."""

    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    polygons: list[list[tuple[float, float]]] = []
    out_of_range = 0
    for shape in data.get("shapes", []):
        if str(shape.get("label", "")).strip().lower() != ROI_LABEL:
            continue
        converted: list[tuple[float, float]] = []
        for x, y in shape["points"]:
            nx, ny = to_384(float(x), float(y))
            if not (0.0 <= nx <= INPUT_SIZE and 0.0 <= ny <= INPUT_SIZE):
                out_of_range += 1
            nx = min(max(nx, 0.0), float(INPUT_SIZE))
            ny = min(max(ny, 0.0), float(INPUT_SIZE))
            converted.append((nx, ny))
        if converted:
            polygons.append(converted)
    return polygons, str(data.get("imagePath", "")), out_of_range


def joint_bbox(polygons: list[list[tuple[float, float]]]) -> tuple[int, int, int, int]:
    """Integer joint bounding box of all polygons, clamped to the image."""

    xs = [p[0] for poly in polygons for p in poly]
    ys = [p[1] for poly in polygons for p in poly]
    x0 = max(0, int(math.floor(min(xs))))
    y0 = max(0, int(math.floor(min(ys))))
    x1 = min(INPUT_SIZE, int(math.ceil(max(xs))))
    y1 = min(INPUT_SIZE, int(math.ceil(max(ys))))
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return x0, y0, x1, y1


def context_bbox(bbox: tuple[int, int, int, int], ratio: float = CONTEXT_RATIO) -> tuple[int, int, int, int]:
    """Expand the box by ``ratio`` of its own width / height on all four sides."""

    x0, y0, x1, y1 = bbox
    dx = (x1 - x0) * ratio
    dy = (y1 - y0) * ratio
    cx0 = max(0, int(math.floor(x0 - dx)))
    cy0 = max(0, int(math.floor(y0 - dy)))
    cx1 = min(INPUT_SIZE, int(math.ceil(x1 + dx)))
    cy1 = min(INPUT_SIZE, int(math.ceil(y1 + dy)))
    cx1 = max(cx1, cx0 + 1)
    cy1 = max(cy1, cy0 + 1)
    return cx0, cy0, cx1, cy1


def rasterize_mask(polygons: list[list[tuple[float, float]]]) -> np.ndarray:
    """Pixel-center-in-polygon rasterization of the polygon union."""

    from matplotlib.path import Path as MplPath

    centers_x = np.arange(INPUT_SIZE) + 0.5
    centers_y = np.arange(INPUT_SIZE) + 0.5
    grid_x, grid_y = np.meshgrid(centers_x, centers_y)  # (H, W)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    mask = np.zeros(INPUT_SIZE * INPUT_SIZE, dtype=bool)
    for poly in polygons:
        vertices = np.asarray(poly, dtype=float)
        if len(vertices) < 3:
            continue
        closed = np.vstack([vertices, vertices[0]])
        path = MplPath(closed)
        mask |= path.contains_points(points)
    return mask.reshape(INPUT_SIZE, INPUT_SIZE)


def build_roi_record(json_path: str | Path, image_path: str | Path) -> ROIRecord:
    polygons, _image_path, out_of_range = load_roi_polygons(json_path)
    sample_id = Path(json_path).stem
    for suffix in ("_1", "_2"):
        if sample_id.endswith(suffix):
            sample_id = sample_id[: -len(suffix)]
    return ROIRecord(
        sample_id=sample_id,
        image_path=Path(image_path),
        polygons=polygons,
        n_red_polygons=len(polygons),
        out_of_range_vertices=out_of_range,
        bbox=joint_bbox(polygons) if polygons else (0, 0, 0, 0),
    )


def load_mr_image(image_path: str | Path) -> Image.Image:
    with Image.open(image_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def compute_mean_fill(image_paths: list[str | Path]) -> np.ndarray:
    """Per-channel mean image (3, 384, 384) in the 0-255 domain."""

    accumulator = np.zeros((3, INPUT_SIZE, INPUT_SIZE), dtype=np.float64)
    for path in image_paths:
        array = np.asarray(load_mr_image(path), dtype=np.float64)
        accumulator += array.transpose(2, 0, 1)
    return (accumulator / len(image_paths)).astype(np.float32)


def augment_pair(
    image: Image.Image,
    mask: Image.Image | None,
    training: bool,
    augmentation: dict[str, float],
) -> tuple[Image.Image, Image.Image | None]:
    """Apply the shared geometric augmentation decision to image and mask."""

    flip, rotation, brightness, contrast = _augmentation_values(training, augmentation)
    if flip:
        image = ImageOps.mirror(image)
        if mask is not None:
            mask = ImageOps.mirror(mask)
    if rotation:
        image = image.rotate(rotation, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        if mask is not None:
            mask = mask.rotate(rotation, resample=Image.Resampling.NEAREST, fillcolor=0)
    if training:
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
    return image, mask


def normalize_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


INPUT_VARIANTS = ("mask", "background", "crop", "context")


def load_cached_mask(mask_path: str | Path) -> np.ndarray:
    """Load a precomputed 384 x 384 binary ROI mask saved by Step 11."""

    with Image.open(mask_path) as handle:
        mask = handle.convert("L")
    if mask.size != (INPUT_SIZE, INPUT_SIZE):
        raise ValueError(f"cached ROI mask has wrong size: {mask_path}")
    return np.asarray(mask, dtype=np.float32) / 255.0 > 0.5


class ROIDataset(Dataset):
    """MR ROI benchmark dataset.

    ``variant`` selects the input construction:

    - ``mask``:       MR x Mask + MeanFill x (1 - Mask)
    - ``background``: MR x (1 - Mask) + MeanFill x Mask
    - ``crop``:       joint-bounding-box crop, restored to 384 x 384 via ResizePad
    - ``context``:    crop of the 20 %-expanded box, restored via ResizePad

    ``frame`` is a manifest frame with columns sample_id / capture_id /
    patient_id / label / fold; ``roi_lookup`` maps sample_id -> ROIRecord.
    ``mask_dir`` points at the precomputed Step 11 mask cache; when omitted the
    mask is rasterized on the fly (identical output, slower).
    """

    def __init__(
        self,
        frame,
        roi_lookup: dict[str, ROIRecord],
        training: bool,
        variant: str,
        mean_fill: np.ndarray | None = None,
        augmentation: dict[str, float] | None = None,
        mask_dir: str | Path | None = None,
    ) -> None:
        if variant not in INPUT_VARIANTS:
            raise ValueError(f"unknown ROI variant: {variant}")
        if variant in ("mask", "background") and mean_fill is None:
            raise ValueError(f"{variant} variant requires a training Mean Fill image")
        self.frame = frame.reset_index(drop=True)
        self.roi_lookup = roi_lookup
        self.training = bool(training)
        self.variant = variant
        self.mean_fill = mean_fill
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.resize_pad = ResizePad(INPUT_SIZE)
        self.augmentation = {**DEFAULT_AUGMENTATION, **(augmentation or {})}

    def __len__(self) -> int:
        return len(self.frame)

    def _load_mask(self, sample_id: str, record: ROIRecord) -> np.ndarray:
        if self.mask_dir is not None:
            return load_cached_mask(self.mask_dir / f"{sample_id}.png")
        return rasterize_mask(record.polygons)

    def _compose_fill(self, image: Image.Image, mask_array: np.ndarray) -> Image.Image:
        array = np.asarray(image, dtype=np.float32)
        mask = mask_array.astype(np.float32)
        if self.variant == "mask":
            composed = array * mask[..., None] + self.mean_fill.transpose(1, 2, 0) * (1.0 - mask[..., None])
        else:  # background
            composed = array * (1.0 - mask[..., None]) + self.mean_fill.transpose(1, 2, 0) * mask[..., None]
        return Image.fromarray(np.clip(np.round(composed), 0, 255).astype(np.uint8))

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        sample_id = str(row["sample_id"])
        record = self.roi_lookup[sample_id]
        image = load_mr_image(record.image_path)

        if self.variant in ("mask", "background"):
            mask_array = self._load_mask(sample_id, record)
            mask_image = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
            image = self.resize_pad(image)
            image, mask_image = augment_pair(image, mask_image, self.training, self.augmentation)
            mask_array = np.asarray(mask_image, dtype=np.float32) / 255.0 > 0.5
            image = self._compose_fill(image, mask_array)
        else:
            box = record.bbox if self.variant == "crop" else context_bbox(record.bbox)
            image = image.crop((box[0], box[1], box[2], box[3]))
            image = self.resize_pad(image)
            image, _ = augment_pair(image, None, self.training, self.augmentation)

        return {
            "image": normalize_tensor(image),
            "target": int(row["label"]),
            "sample_id": sample_id,
            "capture_id": int(row["capture_id"]),
            "patient_id": str(row["patient_id"]),
        }


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker an independent augmentation RNG stream."""

    seed = torch.initial_seed() + worker_id
    random.seed(seed % (2**32))
    np.random.seed(seed % (2**32))


__all__ = [
    "CONTEXT_RATIO",
    "INPUT_VARIANTS",
    "ROIRecord",
    "ROIDataset",
    "SOFT_LABEL_MATRIX",
    "build_roi_record",
    "compute_mean_fill",
    "context_bbox",
    "joint_bbox",
    "load_cached_mask",
    "load_mr_image",
    "load_roi_polygons",
    "rasterize_mask",
    "worker_init_fn",
]
