"""Step 06_N: five-class hard-label multi-channel benchmark without patient isolation."""

from __future__ import annotations

import argparse
import base64
import html
import json
import random
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    DEFAULT_AUGMENTATION,
    EXCLUDED_IDS,
    IMAGE_PATTERN,
    INPUT_SIZE,
    MODALITIES,
    SEED as FOLD_SEED,
    SkinDataset,
    discover_images,
    filter_available_samples,
    iter_folds,
    make_five_folds,
)
from model import build_classifier, count_parameters  # noqa: E402
from non_isolated.report_style import (  # noqa: E402
    REPORT_CSS,
    BAR_EDGE_COLOR,
    combination_axis_label,
    combination_color,
    modality_color,
    style_axis,
    style_bars,
)


STEP_NAME = "step_06_N"
TRAINING_SEEDS = (42, 3407, 2026)
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
MODEL_NAME = "resnet50"
LOSS_NAME = "CrossEntropyLoss"
NUM_WORKERS = 4
COMBINATIONS = (
    ("M", "MR"),
    ("M", "MB"),
    ("M", "MB", "MR"),
    ("MR", "MB"),
    ("M", "MB", "MP", "MR", "MUV"),
)
METRICS = (
    "accuracy",
    "macro_f1",
    "mae",
    "sensitivity",
    "specificity",
    "class_0_precision",
    "class_0_recall",
    "class_0_f1",
    "class_1_precision",
    "class_1_recall",
    "class_1_f1",
    "class_2_precision",
    "class_2_recall",
    "class_2_f1",
    "class_3_precision",
    "class_3_recall",
    "class_3_f1",
    "class_4_precision",
    "class_4_recall",
    "class_4_f1",
)


def combination_name(modalities: tuple[str, ...]) -> str:
    return "+".join(modalities)


def combination_slug(modalities: tuple[str, ...]) -> str:
    return "_".join(modalities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-data-root")
    parser.add_argument("--reference-split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_output_root(output_root: Path) -> None:
    if output_root.name != STEP_NAME or output_root.parent.name != "non_isolated":
        raise ValueError(
            "output-root must end with non_isolated/step_06_N; "
            "pass an explicit result directory on the data disk"
        )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_label(label: int) -> int:
    return int(label)


class BinaryTargetDataset(Dataset):
    """Map labels dynamically while retaining the original manifest unchanged."""

    def __init__(self, dataset: SkinDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = dict(self.dataset[index])
        item["target"] = binary_label(int(item["target"]))
        return item


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def make_loader(
    frame: pd.DataFrame,
    modalities: tuple[str, ...],
    image_lookup: dict[str, dict[int, Path]],
    data_root: Path,
    training: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = BinaryTargetDataset(
        SkinDataset(
            frame,
            data_root,
            modalities=modalities,
            training=training,
            input_size=INPUT_SIZE,
            augmentation=DEFAULT_AUGMENTATION,
            image_lookup=image_lookup,
            include_metadata=False,
        )
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def raw_manifest_frame(manifest: str | Path) -> pd.DataFrame:
    path = Path(manifest).expanduser().resolve()
    if path.is_dir():
        paths = sorted(path.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"no CSV manifest found in {path}")
        return pd.concat([pd.read_csv(item) for item in paths], ignore_index=True)
    return pd.read_csv(path)


def excluded_image_counts(data_root: Path) -> dict[str, dict[str, int]]:
    counts = {
        modality: {str(item): 0 for item in sorted(EXCLUDED_IDS)}
        for modality in MODALITIES
    }
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        match = IMAGE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        modality = match.group(1).upper()
        capture_id = int(match.group(2))
        if capture_id in EXCLUDED_IDS:
            counts[modality][str(capture_id)] += 1
    return counts


def image_size_counts(image_lookup: dict[str, dict[int, Path]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for modality in MODALITIES:
        modality_counts: dict[str, int] = {}
        for path in image_lookup[modality].values():
            with Image.open(path) as image:
                size = f"{image.width}x{image.height}"
            modality_counts[size] = modality_counts.get(size, 0) + 1
        counts[modality] = modality_counts
    return counts


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    mapped = frame["label"].map(binary_label)
    return {str(label): int((mapped == label).sum()) for label in range(NUM_CLASSES)}


def reference_split_matches(folded: pd.DataFrame, reference_split: str | Path) -> tuple[bool, str]:
    path = Path(reference_split).expanduser().resolve()
    if not path.is_file():
        return False, f"reference split not found: {path}"
    reference = pd.read_csv(path)
    if not {"sample_id", "fold"}.issubset(reference.columns):
        return False, "reference split must contain sample_id and fold columns"
    current_map = {
        str(sample_id): int(fold)
        for sample_id, fold in zip(folded["sample_id"], folded["fold"])
    }
    reference_map = {
        str(sample_id): int(fold)
        for sample_id, fold in zip(reference["sample_id"], reference["fold"])
    }
    if current_map != reference_map:
        return False, "step 06 N fold assignments differ from the fixed reference split"
    return True, "matched_step_02_N_fixed_sample_split"


def run_integrity_checks(
    manifest: str,
    data_root: Path,
    reference_data_root: Path | None,
    reference_split: str,
) -> tuple[pd.DataFrame, dict[str, dict[int, Path]], dict[str, object]]:
    raw = raw_manifest_frame(manifest)
    folded = make_five_folds(manifest, patient_isolation=False, seed=FOLD_SEED)
    image_lookup = discover_images(data_root)
    errors: list[str] = []

    if "label" not in raw.columns:
        errors.append("manifest has no label column")
    if set(folded["capture_id"]) & EXCLUDED_IDS:
        errors.append("excluded capture IDs remain after manifest filtering")
    if set(folded["fold"].unique()) != set(range(5)):
        errors.append("fold IDs are not exactly 0-4")
    if folded["sample_id"].duplicated().any():
        errors.append("duplicate sample IDs remain")
    if set(raw["label"].dropna().astype(int)) - set(range(5)):
        errors.append("manifest contains labels outside 0-4")

    split_matches, split_status = reference_split_matches(folded, reference_split)
    if not split_matches:
        errors.append(split_status)

    cache_excluded_counts = excluded_image_counts(data_root)
    reference_root = reference_data_root or data_root
    reference_excluded_counts = excluded_image_counts(reference_root)
    if reference_excluded_counts["MR"]["69"] != 2:
        errors.append(
            "expected two excluded MR ID=69 images in reference data root, "
            f"found {reference_excluded_counts['MR']['69']}"
        )
    if any(
        count
        for modality_counts in cache_excluded_counts.values()
        for count in modality_counts.values()
    ):
        errors.append("excluded IDs remain in the 384x384 cache")
    size_counts = image_size_counts(image_lookup)
    if any(
        size != f"{INPUT_SIZE}x{INPUT_SIZE}"
        for modality_sizes in size_counts.values()
        for size in modality_sizes
    ):
        errors.append("cache contains images that are not exactly 384x384")
    if any(not image_lookup[modality] for modality in MODALITIES):
        errors.append("one or more required modalities have no cached images")

    combination_stats: dict[str, list[dict[str, object]]] = {}
    all_sample_ids = set(folded["sample_id"].astype(str))
    for modalities in COMBINATIONS:
        name = combination_name(modalities)
        stats: list[dict[str, object]] = []
        for fold, train_raw, eval_raw in iter_folds(folded, patient_isolation=False):
            try:
                train_frame = filter_available_samples(train_raw, image_lookup, modalities)
                eval_frame = filter_available_samples(eval_raw, image_lookup, modalities)
            except ValueError as exc:
                errors.append(f"{name} Fold {fold} modality matching failed: {exc}")
                continue
            train_ids = set(train_frame["sample_id"].astype(str))
            eval_ids = set(eval_frame["sample_id"].astype(str))
            if train_ids & eval_ids:
                errors.append(f"{name} Fold {fold} has sample overlap")
            if train_ids | eval_ids != all_sample_ids:
                errors.append(f"{name} Fold {fold} does not cover the fixed sample set")
            if train_frame.empty or eval_frame.empty:
                errors.append(f"{name} Fold {fold} has empty train/evaluation data")
            stats.append(
                {
                    "fold": fold,
                    "train_count": len(train_frame),
                    "evaluation_count": len(eval_frame),
                    "train_class_counts": class_counts(train_frame),
                    "evaluation_class_counts": class_counts(eval_frame),
                }
            )
        combination_stats[name] = stats

    report = {
        "passed": not errors,
        "errors": errors,
        "manifest_rows_before_exclusion": len(raw),
        "manifest_rows_after_exclusion": len(folded),
        "excluded_ids": sorted(EXCLUDED_IDS),
        "reference_data_root": str(reference_root),
        "reference_excluded_image_counts": reference_excluded_counts,
        "cache_excluded_image_counts": cache_excluded_counts,
        "cache_image_size_counts": size_counts,
        "patient_isolation": False,
        "patient_overlap_check": "not_required_for_patient_isolation_false",
        "sample_coverage": len(all_sample_ids) == len(folded),
        "fixed_split_reference": str(Path(reference_split).expanduser().resolve()),
        "fixed_split_status": split_status,
        "fold_counts": {
            str(fold): int((folded["fold"] == fold).sum()) for fold in range(5)
        },
        "global_class_counts": class_counts(folded),
        "combination_stats": combination_stats,
    }
    return folded, image_lookup, report


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal RTX 5090 D run")
    device = torch.device(requested)
    if device.type == "cuda":
        device = torch.device(f"cuda:{device.index or 0}")
        torch.cuda.set_device(device.index)
    return device


def build_model(device: torch.device, modalities: tuple[str, ...]) -> nn.Module:
    return build_classifier(
        name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        pretrained=True,
        dropout=DROPOUT,
        input_channels=3 * len(modalities),
    ).to(device)


def metric_values(targets: Iterable[int], predictions: Iterable[int]) -> tuple[dict[str, float], list[list[int]]]:
    targets_array = np.asarray(list(targets), dtype=np.int64)
    predictions_array = np.asarray(list(predictions), dtype=np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(targets_array, predictions_array):
        confusion[target, prediction] += 1
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for label in range(NUM_CLASSES):
        true_positive = confusion[label, label]
        predicted = confusion[:, label].sum()
        actual = confusion[label, :].sum()
        p = float(true_positive / predicted) if predicted else 0.0
        r = float(true_positive / actual) if actual else 0.0
        score = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)
    metrics = {
        "accuracy": float((targets_array == predictions_array).mean()),
        "macro_f1": float(np.mean(f1)),
        "mae": float(np.abs(targets_array - predictions_array).mean()),
        "sensitivity": recall[1],
        "specificity": recall[0],
    }
    for label in range(NUM_CLASSES):
        metrics[f"class_{label}_precision"] = precision[label]
        metrics[f"class_{label}_recall"] = recall[label]
        metrics[f"class_{label}_f1"] = f1[label]
    return metrics, confusion.tolist()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[list[int]]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(batch["target"].tolist())
    return metric_values(targets, predictions)


def run_smoke(
    folded: pd.DataFrame,
    image_lookup: dict[str, dict[int, Path]],
    data_root: Path,
    device: torch.device,
    num_workers: int,
    output_root: Path,
) -> None:
    smoke_rows = []
    _, train_raw, eval_raw = next(iter_folds(folded, patient_isolation=False))
    for modalities in COMBINATIONS:
        name = combination_name(modalities)
        train_frame = filter_available_samples(train_raw, image_lookup, modalities)
        eval_frame = filter_available_samples(eval_raw, image_lookup, modalities)
        loader = make_loader(
            train_frame.head(2), modalities, image_lookup, data_root, False, 42, num_workers
        )
        batch = next(iter(loader))
        expected_shape = (3 * len(modalities), INPUT_SIZE, INPUT_SIZE)
        if tuple(batch["image"].shape[1:]) != expected_shape:
            raise RuntimeError(f"{name} smoke input shape is invalid: {batch['image'].shape}")
        target = batch["target"]
        if int(target.min()) < 0 or int(target.max()) >= NUM_CLASSES:
            raise RuntimeError(f"{name} smoke target mapping is invalid")
        model = build_model(device, modalities)
        with torch.no_grad():
            logits = model(batch["image"].to(device, non_blocking=True))
            loss = nn.CrossEntropyLoss()(logits, target.to(device))
        if tuple(logits.shape) != (len(target), NUM_CLASSES):
            raise RuntimeError(f"{name} smoke logits shape is invalid: {logits.shape}")
        smoke_rows.append(
            {
                "combination": name,
                "modalities": list(modalities),
                "input_channels": 3 * len(modalities),
                "train_count_fold_01": len(train_frame),
                "evaluation_count_fold_01": len(eval_frame),
                "input_shape": list(batch["image"].shape),
                "logits_shape": list(logits.shape),
                "loss": float(loss.item()),
                "parameter_count": count_parameters(model),
            }
        )
        del model, batch, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(output_root / "logs" / "smoke_test.json", {"passed": True, "rows": smoke_rows})


def train_one(
    modalities: tuple[str, ...],
    fold: int,
    training_seed: int,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    image_lookup: dict[str, dict[int, Path]],
    data_root: Path,
    output_root: Path,
    device: torch.device,
    num_workers: int,
) -> dict[str, object]:
    set_seed(training_seed)
    name = combination_name(modalities)
    run_root = output_root / f"fold_{fold:02d}" / combination_slug(modalities) / f"seed_{training_seed}"
    checkpoint_path = run_root / "checkpoints" / "last.pth"
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_log = run_root / "train.log"
    eval_log = run_root / "eval.log"

    train_loader = make_loader(
        train_frame, modalities, image_lookup, data_root, True, training_seed, num_workers
    )
    eval_loader = make_loader(
        eval_frame, modalities, image_lookup, data_root, False, training_seed, num_workers
    )
    model = build_model(device, modalities)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    parameter_count = count_parameters(model)

    with train_log.open("w", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "combination": name,
                    "modalities": list(modalities),
                    "fold": fold,
                    "training_seed": training_seed,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_total = 0.0
            sample_total = 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.item()) * len(targets)
                sample_total += len(targets)
            epoch_loss = loss_total / sample_total
            log.write(f"epoch={epoch},train_loss={epoch_loss:.8f}\n")
            log.flush()

    torch.save(
        {
            "step": STEP_NAME,
            "combination": name,
            "modalities": list(modalities),
            "fold": fold,
            "training_seed": training_seed,
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    metrics, confusion = evaluate(model, eval_loader, device)
    with eval_log.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"metrics": metrics, "confusion_matrix": confusion}) + "\n")
    row: dict[str, object] = {
        "step": STEP_NAME,
        "combination": name,
        "modalities": list(modalities),
        "input_channels": 3 * len(modalities),
        "fold": fold,
        "training_seed": training_seed,
        "train_count": len(train_frame),
        "evaluation_count": len(eval_frame),
        "train_class_counts": class_counts(train_frame),
        "evaluation_class_counts": class_counts(eval_frame),
        "parameter_count": parameter_count,
        **metrics,
        "confusion_matrix": confusion,
    }
    write_json(
        run_root / "config.json",
        {
            "step": STEP_NAME,
            "combination": name,
            "modalities": list(modalities),
            "fold": fold,
            "training_seed": training_seed,
            "patient_isolation": False,
            "fold_seed": FOLD_SEED,
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "input_channels": 3 * len(modalities),
            "model": MODEL_NAME,
            "pretrained": True,
            "num_classes": NUM_CLASSES,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": None,
            "early_stopping": False,
            "loss": LOSS_NAME,
            "augmentation": DEFAULT_AUGMENTATION,
            "evaluation_checkpoint": str(checkpoint_path.relative_to(output_root)),
        },
    )
    del model, train_loader, eval_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def aggregate(values: list[float]) -> dict[str, float]:
    return {"mean": float(mean(values)), "std": float(stdev(values)) if len(values) > 1 else 0.0}


def summarize(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seed_summaries: list[dict[str, object]] = []
    for modalities in COMBINATIONS:
        name = combination_name(modalities)
        for training_seed in TRAINING_SEEDS:
            selected = [
                row for row in rows
                if row["combination"] == name and row["training_seed"] == training_seed
            ]
            if len(selected) != 5:
                raise RuntimeError(f"{name} Seed {training_seed} has {len(selected)} Fold rows")
            seed_summaries.append(
                {
                    "combination": name,
                    "modalities": list(modalities),
                    "training_seed": training_seed,
                    "fold_count": len(selected),
                    "metrics": {
                        metric: aggregate([float(row[metric]) for row in selected])
                        for metric in METRICS
                    },
                }
            )
    final_summaries: list[dict[str, object]] = []
    for modalities in COMBINATIONS:
        name = combination_name(modalities)
        selected = [item for item in seed_summaries if item["combination"] == name]
        final_summaries.append(
            {
                "combination": name,
                "modalities": list(modalities),
                "training_seed_count": len(selected),
                "metrics": {
                    metric: aggregate(
                        [float(item["metrics"][metric]["mean"]) for item in selected]
                    )
                    for metric in METRICS
                },
            }
        )
    return seed_summaries, final_summaries


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    flat_rows = []
    for row in rows:
        item = dict(row)
        item["modalities"] = "+".join(item["modalities"])
        item["confusion_matrix"] = json.dumps(item["confusion_matrix"])
        flat_rows.append(item)
    pd.DataFrame(flat_rows).to_csv(path, index=False)


def figure_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_figures(
    final_summaries: list[dict[str, object]],
    rows: list[dict[str, object]],
    output_root: Path,
    baseline_summary: dict[str, object] | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    plot_summaries = list(final_summaries)
    if baseline_summary is not None:
        baseline_item = dict(baseline_summary)
        baseline_item["combination"] = "M (baseline)"
        plot_summaries.insert(0, baseline_item)
    labels = [item["combination"] for item in plot_summaries]
    colors = [
        modality_color("M") if label == "M (baseline)" else combination_color(label)
        for label in labels
    ]
    generated: list[Path] = []

    for metric, title, ylabel, formatter, filename in (
        ("accuracy", "step 06 N Accuracy", "accuracy", lambda value: f"{value:.2%}", "accuracy.png"),
        ("macro_f1", "step 06 N Macro-F1", "macro_f1", lambda value: f"{value:.4f}", "macro_f1.png"),
        ("mae", "step 06 N MAE", "mae", lambda value: f"{value:.4f}", "mae.png"),
    ):
        values = [item["metrics"][metric]["mean"] for item in plot_summaries]
        errors = [item["metrics"][metric]["std"] for item in plot_summaries]
        figure, axis = plt.subplots(figsize=(12.8, 8))
        bars = axis.bar(
            range(len(labels)),
            values,
            yerr=errors,
            capsize=6,
            color=colors,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.35,
            error_kw={"ecolor": BAR_EDGE_COLOR, "elinewidth": 1.1, "capthick": 1.1},
        )
        if baseline_summary is not None:
            bars[0].set_hatch("///")
        style_bars(bars)
        axis.set_title(title, loc="left", fontsize=17, pad=16, fontweight="bold")
        axis.set_ylabel(ylabel, fontsize=13)
        axis.set_xticks(range(len(labels)), [combination_axis_label(label) for label in labels])
        axis.tick_params(axis="x", labelsize=11)
        style_axis(axis)
        offset = max(max(values) * 0.018, 0.008)
        for bar, value, error in zip(bars, values, errors):
            axis.text(bar.get_x() + bar.get_width() / 2, value + error + offset, formatter(value), ha="center", va="bottom", fontsize=11, fontweight="bold")
        axis.set_ylim(0, max(value + error for value, error in zip(values, errors)) * 1.20)
        figure.tight_layout()
        path = figures_root / filename
        figure.savefig(path, dpi=120, facecolor="white")
        plt.close(figure)
        generated.append(path)

    figure, axis = plt.subplots(figsize=(12.8, 8))
    x = np.arange(NUM_CLASSES)
    width = 0.13
    all_tops = []
    for index, (label, color) in enumerate(zip(labels, colors)):
        summary = next(item for item in plot_summaries if item["combination"] == label)
        values = [summary["metrics"][f"class_{class_index}_f1"]["mean"] for class_index in range(NUM_CLASSES)]
        errors = [summary["metrics"][f"class_{class_index}_f1"]["std"] for class_index in range(NUM_CLASSES)]
        bars = axis.bar(
            x + (index - (len(plot_summaries) - 1) / 2) * width,
            values,
            width,
            yerr=errors,
            capsize=4,
            color=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.15,
            label=label,
            error_kw={"ecolor": BAR_EDGE_COLOR, "elinewidth": 1.0, "capthick": 1.0},
        )
        if label == "M (baseline)":
            for bar in bars:
                bar.set_hatch("///")
        style_bars(bars)
        for bar, value, error in zip(bars, values, errors):
            all_tops.append(value + error)
            axis.text(bar.get_x() + bar.get_width() / 2, value + error + 0.012, f"{value:.4f}", ha="center", va="bottom", fontsize=8, rotation=90, fontweight="bold")
    axis.set_title("step 06 N Per-class F1", loc="left", fontsize=17, pad=16, fontweight="bold")
    axis.set_ylabel("F1", fontsize=13)
    axis.set_xticks(x, [f"Class {index}" for index in x], fontsize=12)
    axis.set_ylim(0, max(all_tops) * 1.22)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9, frameon=False)
    style_axis(axis)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    path = figures_root / "per_class_f1.png"
    figure.savefig(path, dpi=120, facecolor="white")
    plt.close(figure)
    generated.append(path)

    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.5), squeeze=False)
    axes = axes.ravel()
    image = None
    for axis, modalities in zip(axes, COMBINATIONS):
        name = combination_name(modalities)
        selected = [row for row in rows if row["combination"] == name]
        counts = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
        for row in selected:
            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    counts[i][j] += row["confusion_matrix"][i][j]
        normalized = []
        for count_row in counts:
            total = sum(count_row)
            normalized.append([value / total if total else 0 for value in count_row])
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                color = "white" if normalized[i][j] >= 0.55 else "#123a63"
                axis.text(j, i, f"{counts[i][j]}\n{normalized[i][j]:.1%}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
        axis.set_title(combination_axis_label(name), fontsize=12, pad=8, fontweight="bold")
        class_ticks = list(range(NUM_CLASSES))
        class_labels = [str(index) for index in class_ticks]
        axis.set_xticks(class_ticks, class_labels, fontsize=8)
        axis.set_yticks(class_ticks, class_labels, fontsize=8)
        axis.set_xlabel("Predicted class", fontsize=9)
        axis.set_ylabel("True class", fontsize=9)
    for axis in axes[len(COMBINATIONS):]:
        axis.axis("off")
    figure.suptitle("step 06 N Normalized Confusion Matrices", fontsize=20, y=0.98, fontweight="bold")
    figure.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.88, wspace=0.38, hspace=0.38)
    if image is not None:
        colorbar_axis = figure.add_axes([0.94, 0.22, 0.012, 0.50])
        figure.colorbar(image, cax=colorbar_axis, label="row proportion")
    path = figures_root / "confusion_matrices.png"
    figure.savefig(path, dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    generated.insert(0, path)
    return generated


def html_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            if isinstance(value, list):
                value = "+".join(value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_report(
    output_root: Path,
    config: dict[str, object],
    integrity: dict[str, object],
    rows: list[dict[str, object]],
    seed_summaries: list[dict[str, object]],
    final_summaries: list[dict[str, object]],
    figures: list[Path],
    baseline_summary: dict[str, object] | None = None,
) -> None:
    final_rows = []
    if baseline_summary is not None:
        final_rows.append(
            {
                "combination": "M (baseline)",
                "runs": baseline_summary["training_seed_count"] * 5,
                "accuracy_mean": baseline_summary["metrics"]["accuracy"]["mean"],
                "accuracy_std": baseline_summary["metrics"]["accuracy"]["std"],
                "macro_f1_mean": baseline_summary["metrics"]["macro_f1"]["mean"],
                "macro_f1_std": baseline_summary["metrics"]["macro_f1"]["std"],
                "mae_mean": baseline_summary["metrics"]["mae"]["mean"],
                "mae_std": baseline_summary["metrics"]["mae"]["std"],
            }
        )
    for item in final_summaries:
        final_rows.append(
            {
                "combination": item["combination"],
                "runs": item["training_seed_count"] * 5,
                "accuracy_mean": item["metrics"]["accuracy"]["mean"],
                "accuracy_std": item["metrics"]["accuracy"]["std"],
                "macro_f1_mean": item["metrics"]["macro_f1"]["mean"],
                "macro_f1_std": item["metrics"]["macro_f1"]["std"],
                "mae_mean": item["metrics"]["mae"]["mean"],
                "mae_std": item["metrics"]["mae"]["std"],
            }
        )
    seed_rows = []
    for item in seed_summaries:
        seed_rows.append(
            {
                "combination": item["combination"],
                "seed": item["training_seed"],
                "folds": item["fold_count"],
                "accuracy_mean": item["metrics"]["accuracy"]["mean"],
                "accuracy_std": item["metrics"]["accuracy"]["std"],
                "macro_f1_mean": item["metrics"]["macro_f1"]["mean"],
                "macro_f1_std": item["metrics"]["macro_f1"]["std"],
            }
        )
    fold_rows = []
    for row in rows:
        matrix = row["confusion_matrix"]
        fold_rows.append(
            {
                "combination": row["combination"],
                "seed": row["training_seed"],
                "fold": row["fold"],
                "eval_n": row["evaluation_count"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "mae": row["mae"],
                "confusion_matrix": json.dumps(matrix, ensure_ascii=False),
            }
        )
    images = "".join(
        f"<h2>{html.escape(path.stem)}</h2><img src=\"{figure_data_uri(path)}\" alt=\"{html.escape(path.stem)}\" />"
        for path in figures
    )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{STEP_NAME}_report</title>
<style>{REPORT_CSS}</style></head><body>
<h1>Step 06_N Five-class Hard-label Multi-channel Benchmark</h1>
<p>Patient Isolation = False；样本级五折交叉验证；原始标签 0–4 保持不变；Input 384×384；ResNet50 ImageNet pretrained；Batch Size 32；Epoch 50；AdamW；Learning Rate 1e-4；Weight Decay 1e-4；无 Scheduler、无 Early Stopping。</p>
<p class="note">正式完成度：75/75 multi-channel runs。Step 05_N 的 M 结果作为既有 Baseline；本 Step 只改变输入通道组合。每个组合包含 3 个 Training Seed × 5 个 Fold。</p>
<section><h2>Final Mean ± Std</h2>{html_table(final_rows, list(final_rows[0]))}</section>
<section><h2>Training Seed Mean ± Std</h2>{html_table(seed_rows, list(seed_rows[0]))}</section>
<section><h2>Fold-level results</h2>{html_table(fold_rows, list(fold_rows[0]))}</section>
<section><h2>Configuration</h2><pre>{html.escape(json.dumps(config, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Integrity checks</h2><pre>{html.escape(json.dumps(integrity, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Figures</h2>{images}</section>
</body></html>"""
    (output_root / "report_step_06_N.html").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    validate_output_root(output_root)
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        stale_markers = [
            output_root / "metrics_step_06_N.json",
            output_root / "metrics_step_06_N.csv",
            output_root / "report_step_06_N.html",
        ]
        existing_runs = [
            path
            for fold_root in output_root.glob("fold_*")
            if fold_root.is_dir()
            for path in fold_root.iterdir()
            if path.is_dir() and path.name not in {".gitkeep"}
        ]
        if any(path.exists() for path in stale_markers) or existing_runs:
            raise RuntimeError("step 06 N contains prior formal results; refusing to mix runs")

    data_root = Path(args.data_root).expanduser().resolve()
    reference_data_root = (
        Path(args.reference_data_root).expanduser().resolve()
        if args.reference_data_root
        else None
    )
    folded, image_lookup, integrity = run_integrity_checks(
        args.manifest, data_root, reference_data_root, args.reference_split
    )
    write_json(output_root / "integrity_report.json", integrity)
    if not integrity["passed"]:
        raise RuntimeError("pre-training integrity checks failed; see integrity_report.json")

    device = resolve_device(args.device)
    config = {
        "experiment_name": STEP_NAME,
        "benchmark_type": "multi_modality_channel_combination",
        "task": "five_class",
        "label_strategy": "Hard Label",
        "patient_isolation": False,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "data_root": str(data_root),
        "reference_data_root": str(reference_data_root) if reference_data_root else None,
        "reference_split": str(Path(args.reference_split).expanduser().resolve()),
        "preprocessed_cache": "384x384_jpeg",
        "output_root": str(output_root),
        "fold_seed": FOLD_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "combinations": [list(modalities) for modalities in COMBINATIONS],
        "baseline_reference": "non_isolated/step_05_N M",
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "model": MODEL_NAME,
        "pretrained": True,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": None,
        "early_stopping": False,
        "loss": LOSS_NAME,
        "dropout": DROPOUT,
        "augmentation": DEFAULT_AUGMENTATION,
        "device": str(device),
        "num_workers": args.num_workers,
        "evaluation_checkpoint": "Epoch 50 last.pth",
    }
    write_json(output_root / "config_step_06_N.json", config)
    folded[["sample_id", "fold"]].to_csv(output_root / "folds_step_06_N.csv", index=False)
    run_smoke(folded, image_lookup, data_root, device, args.num_workers, output_root)
    if args.mode == "smoke":
        print("SMOKE_TEST_OK", flush=True)
        return

    rows: list[dict[str, object]] = []
    for modalities in COMBINATIONS:
        name = combination_name(modalities)
        for training_seed in TRAINING_SEEDS:
            for fold, train_raw, eval_raw in iter_folds(folded, patient_isolation=False):
                train_frame = filter_available_samples(train_raw, image_lookup, modalities)
                eval_frame = filter_available_samples(eval_raw, image_lookup, modalities)
                print(
                    f"START combination={name} seed={training_seed} fold={fold}",
                    flush=True,
                )
                row = train_one(
                    modalities,
                    fold,
                    training_seed,
                    train_frame,
                    eval_frame,
                    image_lookup,
                    data_root,
                    output_root,
                    device,
                    args.num_workers,
                )
                rows.append(row)
                write_json(output_root / "metrics_step_06_N.json", {"fold_results": rows})
                save_csv(output_root / "metrics_step_06_N.csv", rows)
                print(
                    f"DONE combination={name} seed={training_seed} fold={fold} "
                    f"macro_f1={row['macro_f1']:.6f}",
                    flush=True,
                )

    seed_summaries, final_summaries = summarize(rows)
    write_json(
        output_root / "metrics_step_06_N.json",
        {
            "integrity": integrity,
            "fold_results": rows,
            "seed_summaries": seed_summaries,
            "final_summaries": final_summaries,
        },
    )
    save_csv(output_root / "metrics_step_06_N.csv", rows)
    write_json(output_root / "seed_summaries_step_06_N.json", seed_summaries)
    write_json(output_root / "final_summary_step_06_N.json", final_summaries)
    baseline_path = output_root.parent / "step_05_N" / "final_summary_step_05_N.json"
    baseline_summaries = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_summary = next(item for item in baseline_summaries if item["modality"] == "M")
    figures = make_figures(final_summaries, rows, output_root, baseline_summary)
    write_report(
        output_root,
        config,
        integrity,
        rows,
        seed_summaries,
        final_summaries,
        figures,
        baseline_summary,
    )
    print("STEP_06_N_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
