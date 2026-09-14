"""Common 384x384 dataset and five-fold splitting utilities."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


INPUT_SIZE = 384
N_FOLDS = 5
SEED = 42
EXCLUDED_IDS = frozenset({69, 296, 769, 770})
MODALITIES = ("M", "MB", "MP", "MR", "MUV")
IMAGE_PATTERN = re.compile(
    r"^(MUV|MB|MP|MR|M)(\d+)\.(?:jpe?g|png|bmp)$", re.IGNORECASE
)
ID_PATTERN = re.compile(r"(\d+)$")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_AUGMENTATION = {
    "horizontal_flip_p": 0.5,
    "rotation_degrees": 7.0,
    "brightness": 0.10,
    "contrast": 0.10,
}


def _capture_id(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    match = ID_PATTERN.search(str(value).strip())
    if match is None:
        raise ValueError(f"cannot extract numeric sample ID from {value!r}")
    return int(match.group(1))


def _read_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        csv_paths = sorted(manifest_path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"no CSV manifest found in {manifest_path}")
        return pd.concat(
            [pd.read_csv(csv_path) for csv_path in csv_paths], ignore_index=True
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def load_manifest(
    manifest: str | Path | pd.DataFrame,
    excluded_ids: Iterable[int] = EXCLUDED_IDS,
) -> pd.DataFrame:
    """Load and normalize one combined metadata table.

    The table must contain ``label``, ``patient_id`` or ``patient_group``, and
    one of ``sample_id``, ``capture_id``, ``source_id`` or ``split_unit``.
    """

    frame = (
        _read_manifest(manifest)
        if not isinstance(manifest, pd.DataFrame)
        else manifest.copy()
    )
    if "label" not in frame.columns:
        raise ValueError("manifest must contain a 'label' column")

    patient_column = next(
        (name for name in ("patient_id", "patient_group") if name in frame.columns),
        None,
    )
    if patient_column is None:
        raise ValueError("manifest must contain 'patient_id' or 'patient_group'")

    id_column = next(
        (
            name
            for name in ("sample_id", "capture_id", "source_id", "split_unit")
            if name in frame.columns
        ),
        None,
    )
    if id_column is None:
        raise ValueError(
            "manifest must contain 'sample_id', 'capture_id', 'source_id', or 'split_unit'"
        )

    frame = frame.reset_index(drop=True)
    frame["sample_id"] = frame[id_column].map(
        lambda value: str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer()
        else str(value).strip()
    )
    frame["capture_id"] = frame[id_column].map(_capture_id).astype(int)
    frame["patient_id"] = frame[patient_column].map(lambda value: str(value).strip())
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)

    invalid_labels = sorted(set(frame["label"]) - set(range(5)))
    if invalid_labels:
        raise ValueError(f"manifest contains labels outside 0-4: {invalid_labels}")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"manifest contains duplicate sample IDs: {duplicates[:5]}")

    excluded = {int(value) for value in excluded_ids}
    frame = frame.loc[~frame["capture_id"].isin(excluded)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("no samples remain after applying excluded IDs")
    return frame


def discover_images(
    data_root: str | Path,
    excluded_ids: Iterable[int] = EXCLUDED_IDS,
) -> dict[str, dict[int, Path]]:
    """Index modality files below ``data_root`` without hard-coded paths."""

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")
    excluded = {int(value) for value in excluded_ids}
    lookup: dict[str, dict[int, Path]] = {modality: {} for modality in MODALITIES}

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        match = IMAGE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        modality = match.group(1).upper()
        capture_id = int(match.group(2))
        if capture_id in excluded:
            continue
        if capture_id in lookup[modality]:
            raise ValueError(
                f"duplicate {modality} image for sample {capture_id}: "
                f"{lookup[modality][capture_id]} and {path}"
            )
        lookup[modality][capture_id] = path
    return lookup


def _validate_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).upper() for value in modalities)
    if not normalized:
        raise ValueError("at least one modality is required")
    unknown = sorted(set(normalized) - set(MODALITIES))
    if unknown:
        raise ValueError(f"unknown modalities: {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("modalities must not contain duplicates")
    return normalized


def filter_available_samples(
    frame: pd.DataFrame,
    image_lookup: dict[str, dict[int, Path]],
    modalities: Sequence[str],
) -> pd.DataFrame:
    """Keep only samples for which every requested modality is present."""

    normalized = _validate_modalities(modalities)
    available_ids = set(image_lookup[normalized[0]])
    for modality in normalized[1:]:
        available_ids &= set(image_lookup[modality])
    filtered = frame.loc[frame["capture_id"].isin(available_ids)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("no samples remain after modality matching")
    return filtered


def _assign_sample_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    assignments: dict[str, int] = {}
    for _, group in frame.groupby("label", sort=True):
        sample_ids = group["sample_id"].tolist()
        rng.shuffle(sample_ids)
        for position, sample_id in enumerate(sample_ids):
            assignments[str(sample_id)] = position % n_folds
    return assignments


def _assign_patient_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    groups: list[tuple[str, pd.DataFrame]] = [
        (str(patient_id), group.copy())
        for patient_id, group in frame.groupby("patient_id", sort=True)
    ]
    if len(groups) < n_folds:
        raise ValueError(
            f"patient isolation requires at least {n_folds} patients; found {len(groups)}"
        )

    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    labels = sorted(frame["label"].unique())
    target_size = len(frame) / n_folds
    target_labels = {
        label: int((frame["label"] == label).sum()) / n_folds for label in labels
    }
    fold_sizes = [0] * n_folds
    fold_labels = [{label: 0 for label in labels} for _ in range(n_folds)]
    assignments: dict[str, int] = {}

    for group_index, (_, group) in enumerate(groups):
        group_size = len(group)
        group_labels = group["label"].value_counts().to_dict()

        def score(fold: int) -> tuple[float, int]:
            size_error = abs((fold_sizes[fold] + group_size) - target_size)
            label_error = sum(
                abs(
                    fold_labels[fold].get(label, 0)
                    + int(group_labels.get(label, 0))
                    - target_labels[label]
                )
                for label in labels
            )
            return size_error + label_error, fold_sizes[fold]

        selected_fold = (
            group_index
            if group_index < n_folds
            else min(range(n_folds), key=score)
        )
        fold_sizes[selected_fold] += group_size
        for label in labels:
            fold_labels[selected_fold][label] += int(group_labels.get(label, 0))
        for sample_id in group["sample_id"]:
            assignments[str(sample_id)] = selected_fold
    return assignments


def make_five_folds(
    manifest: str | Path | pd.DataFrame,
    patient_isolation: bool,
    seed: int = SEED,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """Return a normalized manifest with a deterministic ``fold`` column."""

    if n_folds != N_FOLDS:
        raise ValueError(f"this protocol requires exactly {N_FOLDS} folds")
    frame = load_manifest(manifest)
    assignments = (
        _assign_patient_folds(frame, n_folds, seed)
        if patient_isolation
        else _assign_sample_folds(frame, n_folds, seed)
    )
    frame["fold"] = frame["sample_id"].map(assignments)
    if frame["fold"].isna().any():
        raise RuntimeError("fold assignment is incomplete")
    frame["fold"] = frame["fold"].astype(int)
    validate_folds(frame, patient_isolation=patient_isolation, n_folds=n_folds)
    return frame


def iter_folds(
    folded_frame: pd.DataFrame,
    patient_isolation: bool,
    n_folds: int = N_FOLDS,
) -> Iterable[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(fold_number, train_frame, evaluation_frame)`` for each Fold."""

    validate_folds(folded_frame, patient_isolation, n_folds)
    for fold in range(n_folds):
        evaluation = folded_frame.loc[folded_frame["fold"] == fold].reset_index(drop=True)
        training = folded_frame.loc[folded_frame["fold"] != fold].reset_index(drop=True)
        yield fold + 1, training, evaluation


def validate_folds(
    folded_frame: pd.DataFrame,
    patient_isolation: bool,
    n_folds: int = N_FOLDS,
) -> None:
    """Check sample coverage and the selected patient-isolation rule."""

    required = {"sample_id", "patient_id", "fold"}
    missing = required - set(folded_frame.columns)
    if missing:
        raise ValueError(f"folded manifest is missing columns: {sorted(missing)}")
    if set(folded_frame["fold"].unique()) != set(range(n_folds)):
        raise ValueError("fold IDs must be exactly 0, 1, 2, 3, and 4")
    if folded_frame["sample_id"].duplicated().any():
        raise ValueError("folded manifest contains duplicate sample IDs")

    all_samples = set(folded_frame["sample_id"].astype(str))
    for fold in range(n_folds):
        evaluation = folded_frame.loc[folded_frame["fold"] == fold]
        training = folded_frame.loc[folded_frame["fold"] != fold]
        if evaluation.empty or training.empty:
            raise ValueError(f"Fold {fold + 1} has an empty training or evaluation set")
        if set(evaluation["sample_id"]) & set(training["sample_id"]):
            raise ValueError(f"sample leakage detected in Fold {fold + 1}")
        if patient_isolation and set(evaluation["patient_id"]) & set(training["patient_id"]):
            raise ValueError(f"patient leakage detected in Fold {fold + 1}")

    evaluated_samples = set(
        folded_frame.loc[folded_frame["fold"].isin(range(n_folds)), "sample_id"]
    )
    if evaluated_samples != all_samples:
        raise ValueError("not every sample has exactly one evaluation Fold")


class ResizePad:
    """Resize without distortion and center-pad to a square image."""

    def __init__(self, size: int = INPUT_SIZE) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = self.size / max(image.size)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        resized = image.resize(
            (width, height), Image.Resampling.BILINEAR, reducing_gap=2.0
        )
        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        canvas.paste(resized, ((self.size - width) // 2, (self.size - height) // 2))
        return canvas


def _augmentation_values(
    training: bool, augmentation: dict[str, float]
) -> tuple[bool, float, float, float]:
    if not training:
        return False, 0.0, 1.0, 1.0
    flip = random.random() < float(augmentation["horizontal_flip_p"])
    rotation = random.uniform(
        -float(augmentation["rotation_degrees"]),
        float(augmentation["rotation_degrees"]),
    )
    brightness = random.uniform(
        max(0.0, 1.0 - float(augmentation["brightness"])),
        1.0 + float(augmentation["brightness"]),
    )
    contrast = random.uniform(
        max(0.0, 1.0 - float(augmentation["contrast"])),
        1.0 + float(augmentation["contrast"]),
    )
    return flip, rotation, brightness, contrast


class SkinDataset(Dataset):
    """Shared single- and multi-modality dataset.

    One modality returns three RGB channels. Multiple modalities are loaded in
    the declared order and concatenated into ``[3 * n_modalities, H, W]``.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        data_root: str | Path,
        modalities: str | Sequence[str] = ("M",),
        training: bool = False,
        input_size: int = INPUT_SIZE,
        augmentation: dict[str, float] | None = None,
    ) -> None:
        if input_size != INPUT_SIZE:
            raise ValueError(f"this protocol requires input_size={INPUT_SIZE}")
        self.modalities = _validate_modalities(
            (modalities,) if isinstance(modalities, str) else modalities
        )
        self.training = bool(training)
        self.resize_pad = ResizePad(input_size)
        self.augmentation = {**DEFAULT_AUGMENTATION, **(augmentation or {})}
        self.frame = load_manifest(manifest)
        self.image_lookup = discover_images(data_root)
        self.frame = filter_available_samples(
            self.frame, self.image_lookup, self.modalities
        )

    @property
    def input_channels(self) -> int:
        return 3 * len(self.modalities)

    def __len__(self) -> int:
        return len(self.frame)

    def _load_images(self, capture_id: int) -> list[Image.Image]:
        images: list[Image.Image] = []
        for modality in self.modalities:
            path = self.image_lookup[modality][capture_id]
            with Image.open(path) as source:
                images.append(ImageOps.exif_transpose(source).convert("RGB"))
        return images

    def _to_tensor(
        self,
        image: Image.Image,
        flip: bool,
        rotation: float,
        brightness: float,
        contrast: float,
        apply_color: bool,
    ) -> Tensor:
        image = self.resize_pad(image)
        if flip:
            image = ImageOps.mirror(image)
        if rotation:
            image = image.rotate(
                rotation,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
        if apply_color:
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        capture_id = int(row["capture_id"])
        images = self._load_images(capture_id)
        flip, rotation, brightness, contrast = _augmentation_values(
            self.training, self.augmentation
        )
        tensors = [
            self._to_tensor(
                image,
                flip,
                rotation,
                brightness,
                contrast,
                apply_color=(position == 0 and self.modalities[position] == "M"),
            )
            for position, image in enumerate(images)
        ]
        return {
            "image": torch.cat(tensors, dim=0),
            "target": int(row["label"]),
            "sample_id": str(row["sample_id"]),
            "capture_id": capture_id,
            "patient_id": str(row["patient_id"]),
            "image_paths": [
                str(self.image_lookup[modality][capture_id])
                for modality in self.modalities
            ],
        }


# Compatibility aliases for step scripts that prefer the CEA naming.
CEADataset = SkinDataset
build_five_folds = make_five_folds


__all__ = [
    "CEADataset",
    "DEFAULT_AUGMENTATION",
    "EXCLUDED_IDS",
    "INPUT_SIZE",
    "MODALITIES",
    "N_FOLDS",
    "SEED",
    "SkinDataset",
    "build_five_folds",
    "discover_images",
    "filter_available_samples",
    "iter_folds",
    "load_manifest",
    "make_five_folds",
    "validate_folds",
]
