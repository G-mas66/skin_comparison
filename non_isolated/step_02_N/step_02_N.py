"""Step 02_N: binary single-modality benchmark without patient isolation."""

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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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
    load_manifest,
    make_five_folds,
)
from model import build_classifier, count_parameters  # noqa: E402
from non_isolated.report_style import (  # noqa: E402
    REPORT_CSS,
    BAR_EDGE_COLOR,
    combination_color,
    modality_color,
    style_axis,
    style_bars,
)


STEP_NAME = "step_02_N"
TRAINING_SEEDS = (42, 3407, 2026)
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
MODEL_NAME = "resnet50"
LOSS_NAME = "CrossEntropyLoss"
NUM_WORKERS = 4
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-data-root")
    parser.add_argument("--output-root", default=str(SCRIPT_DIR))
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
    return 0 if int(label) <= 2 else 1


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
    modality: str,
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
            modalities=(modality,),
            training=training,
            input_size=INPUT_SIZE,
            augmentation=DEFAULT_AUGMENTATION,
            image_lookup=image_lookup,
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
    counts = {modality: {str(item): 0 for item in sorted(EXCLUDED_IDS)} for modality in MODALITIES}
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


def image_size_counts(
    image_lookup: dict[str, dict[int, Path]],
) -> dict[str, dict[str, int]]:
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


def run_integrity_checks(
    manifest: str,
    data_root: Path,
    reference_data_root: Path | None = None,
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

    cache_excluded_counts = excluded_image_counts(data_root)
    reference_root = reference_data_root or data_root
    reference_excluded_counts = excluded_image_counts(reference_root)
    if reference_excluded_counts["MR"]["69"] != 2:
        errors.append(
            "expected two excluded MR ID=69 images in the reference data root, "
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

    modality_stats: dict[str, list[dict[str, object]]] = {}
    for modality in MODALITIES:
        if not image_lookup[modality]:
            errors.append(f"no images found for modality {modality}")
            modality_stats[modality] = []
            continue
        stats: list[dict[str, object]] = []
        for fold, train_raw, eval_raw in iter_folds(folded, patient_isolation=False):
            try:
                train_frame = filter_available_samples(train_raw, image_lookup, (modality,))
                eval_frame = filter_available_samples(eval_raw, image_lookup, (modality,))
            except ValueError as exc:
                errors.append(f"{modality} Fold {fold} modality matching failed: {exc}")
                continue
            if train_frame.empty or eval_frame.empty:
                errors.append(f"{modality} Fold {fold} has empty train/evaluation data")
            stats.append(
                {
                    "fold": fold,
                    "train_count": len(train_frame),
                    "evaluation_count": len(eval_frame),
                    "train_class_counts": class_counts(train_frame),
                    "evaluation_class_counts": class_counts(eval_frame),
                }
            )
        modality_stats[modality] = stats

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
        "sample_coverage": len(set(folded["sample_id"])) == len(folded),
        "fold_counts": {
            str(fold): int((folded["fold"] == fold).sum()) for fold in range(5)
        },
        "global_binary_class_counts": class_counts(folded),
        "modality_stats": modality_stats,
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


def build_model(device: torch.device) -> nn.Module:
    model = build_classifier(
        name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        pretrained=True,
        dropout=DROPOUT,
        input_channels=3,
    )
    return model.to(device)


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
    with torch.no_grad():
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
    for modality in MODALITIES:
        _, train_raw, eval_raw = next(iter_folds(folded, patient_isolation=False))
        train_frame = filter_available_samples(train_raw, image_lookup, (modality,))
        eval_frame = filter_available_samples(eval_raw, image_lookup, (modality,))
        loader = make_loader(
            train_frame.head(1), modality, image_lookup, data_root, False, 42, num_workers
        )
        batch = next(iter(loader))
        if tuple(batch["image"].shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
            raise RuntimeError(f"{modality} smoke input shape is invalid: {batch['image'].shape}")
        target = batch["target"]
        if int(target.min()) not in {0, 1} or int(target.max()) not in {0, 1}:
            raise RuntimeError(f"{modality} smoke target mapping is invalid")
        model = build_model(device)
        with torch.no_grad():
            logits = model(batch["image"].to(device, non_blocking=True))
            loss = nn.CrossEntropyLoss()(logits, target.to(device))
        if tuple(logits.shape) != (1, NUM_CLASSES):
            raise RuntimeError(f"{modality} smoke logits shape is invalid: {logits.shape}")
        smoke_rows.append(
            {
                "modality": modality,
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
    modality: str,
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
    run_root = output_root / f"fold_{fold:02d}" / f"seed_{training_seed}"
    checkpoint_path = run_root / "checkpoints" / "last.pth"
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_log = run_root / "train.log"
    eval_log = run_root / "eval.log"

    train_loader = make_loader(
        train_frame, modality, image_lookup, data_root, True, training_seed, num_workers
    )
    eval_loader = make_loader(
        eval_frame, modality, image_lookup, data_root, False, training_seed, num_workers
    )
    model = build_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    parameter_count = count_parameters(model)

    with train_log.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"modality": modality, "fold": fold, "training_seed": training_seed}) + "\n")
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
            "modality": modality,
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
        "modality": modality,
        "fold": fold,
        "training_seed": training_seed,
        "train_count": len(train_frame),
        "evaluation_count": len(eval_frame),
        "train_class_0": int(sum(binary_label(value) == 0 for value in train_frame["label"])),
        "train_class_1": int(sum(binary_label(value) == 1 for value in train_frame["label"])),
        "evaluation_class_0": int(sum(binary_label(value) == 0 for value in eval_frame["label"])),
        "evaluation_class_1": int(sum(binary_label(value) == 1 for value in eval_frame["label"])),
        "parameter_count": parameter_count,
        **metrics,
        "confusion_matrix": confusion,
    }
    write_json(
        run_root / "config.json",
        {
            "step": STEP_NAME,
            "modality": modality,
            "fold": fold,
            "training_seed": training_seed,
            "patient_isolation": False,
            "fold_seed": FOLD_SEED,
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "model": MODEL_NAME,
            "pretrained": True,
            "input_channels": 3,
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
    return row


def aggregate(values: list[float]) -> dict[str, float]:
    return {"mean": float(mean(values)), "std": float(stdev(values)) if len(values) > 1 else 0.0}


def summarize(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seed_summaries: list[dict[str, object]] = []
    for modality in MODALITIES:
        for training_seed in TRAINING_SEEDS:
            selected = [
                row for row in rows
                if row["modality"] == modality and row["training_seed"] == training_seed
            ]
            seed_summaries.append(
                {
                    "modality": modality,
                    "training_seed": training_seed,
                    "fold_count": len(selected),
                    "metrics": {
                        metric: aggregate([float(row[metric]) for row in selected])
                        for metric in METRICS
                    },
                }
            )
    final_summaries: list[dict[str, object]] = []
    for modality in MODALITIES:
        selected = [item for item in seed_summaries if item["modality"] == modality]
        final_summaries.append(
            {
                "modality": modality,
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
        item["confusion_matrix"] = json.dumps(item["confusion_matrix"])
        flat_rows.append(item)
    pd.DataFrame(flat_rows).to_csv(path, index=False)


def figure_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_figures(
    final_summaries: list[dict[str, object]],
    output_root: Path,
    rows: list[dict[str, object]] | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    labels = [item["modality"] for item in final_summaries]
    colors = [modality_color(label) for label in labels]
    for metric, title in (
        ("accuracy", "Step 02_N Accuracy"),
        ("macro_f1", "Step 02_N Macro-F1"),
        ("mae", "Step 02_N MAE"),
    ):
        values = [item["metrics"][metric]["mean"] for item in final_summaries]
        errors = [item["metrics"][metric]["std"] for item in final_summaries]
        figure, axis = plt.subplots(figsize=(8.5, 5.4))
        bars = axis.bar(
            labels,
            values,
            yerr=errors,
            capsize=4,
            color=colors,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.35,
            error_kw={"ecolor": BAR_EDGE_COLOR, "elinewidth": 1.1, "capthick": 1.1},
        )
        style_bars(bars)
        axis.set_title(title, loc="left", pad=14, fontsize=15, fontweight="bold")
        axis.set_ylabel(metric)
        style_axis(axis)
        offset = max(max(values) * 0.018, 0.008)
        for bar, value, error in zip(bars, values, errors):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#172033",
            )
        axis.set_ylim(0, max(value + error for value, error in zip(values, errors)) * 1.2)
        figure.tight_layout()
        path = figures_root / f"{metric}.png"
        figure.savefig(path, dpi=160, facecolor="#F7F9FC")
        plt.close(figure)
        generated.append(path)

    figure, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(NUM_CLASSES)
    width = 0.15
    for index, item in enumerate(final_summaries):
        means = [item["metrics"][f"class_{label}_f1"]["mean"] for label in range(NUM_CLASSES)]
        errors = [item["metrics"][f"class_{label}_f1"]["std"] for label in range(NUM_CLASSES)]
        bars = axis.bar(
            x + (index - (len(final_summaries) - 1) / 2) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=colors[index],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.2,
            label=item["modality"],
            error_kw={"ecolor": BAR_EDGE_COLOR, "elinewidth": 1.0, "capthick": 1.0},
        )
        style_bars(bars)
    axis.set_title("Step 02_N Per-class F1", loc="left", pad=14, fontsize=15, fontweight="bold")
    axis.set_ylabel("F1")
    axis.set_xticks(x, ["Class 0", "Class 1"])
    axis.legend(frameon=False, ncol=3)
    style_axis(axis)
    figure.tight_layout()
    path = figures_root / "per_class_f1.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(path)
    if rows is not None:
        figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.5), squeeze=False)
        axes = axes.ravel()
        image = None
        for axis, item in zip(axes, final_summaries):
            selected = [row for row in rows if row["modality"] == item["modality"]]
            counts = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
            for row in selected:
                for true_label in range(NUM_CLASSES):
                    for predicted_label in range(NUM_CLASSES):
                        counts[true_label][predicted_label] += row["confusion_matrix"][true_label][predicted_label]
            normalized = []
            for count_row in counts:
                total = sum(count_row)
                normalized.append([value / total if total else 0 for value in count_row])
            image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for true_label in range(NUM_CLASSES):
                for predicted_label in range(NUM_CLASSES):
                    text_color = "white" if normalized[true_label][predicted_label] >= 0.55 else "#123a63"
                    axis.text(
                        predicted_label,
                        true_label,
                        f"{counts[true_label][predicted_label]}\n{normalized[true_label][predicted_label]:.1%}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=10,
                        fontweight="bold",
                    )
            axis.set_title(item["modality"], fontsize=12, pad=8, fontweight="bold")
            axis.set_xticks(range(NUM_CLASSES), ["Class 0", "Class 1"], fontsize=8)
            axis.set_yticks(range(NUM_CLASSES), ["Class 0", "Class 1"], fontsize=8)
            axis.set_xlabel("Predicted", fontsize=9)
            axis.set_ylabel("True", fontsize=9)
        for axis in axes[len(final_summaries):]:
            axis.axis("off")
        figure.suptitle("Step 02_N Normalized Confusion Matrices", fontsize=20, y=0.98, fontweight="bold")
        figure.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.88, wspace=0.38, hspace=0.38)
        if image is not None:
            colorbar_axis = figure.add_axes([0.92, 0.22, 0.016, 0.48])
            figure.colorbar(image, cax=colorbar_axis, label="row proportion")
        path = figures_root / "confusion_matrices.png"
        figure.savefig(path, dpi=120, facecolor="#F7F9FC", bbox_inches="tight")
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
) -> None:
    final_rows = []
    for item in final_summaries:
        final_rows.append(
            {
                "modality": item["modality"],
                "accuracy_mean": item["metrics"]["accuracy"]["mean"],
                "accuracy_std": item["metrics"]["accuracy"]["std"],
                "macro_f1_mean": item["metrics"]["macro_f1"]["mean"],
                "macro_f1_std": item["metrics"]["macro_f1"]["std"],
                "mae_mean": item["metrics"]["mae"]["mean"],
                "mae_std": item["metrics"]["mae"]["std"],
            }
        )
    images = "".join(
        f'<h3>{html.escape(path.stem)}</h3><img src="{figure_data_uri(path)}" />'
        for path in figures
    )
    fold_columns = ["modality", "fold", "training_seed", "train_count", "evaluation_count", *METRICS]
    report = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{STEP_NAME}</title>
<style>{REPORT_CSS}</style>
</head><body><h1>{STEP_NAME}</h1>
<p>Patient Isolation = False; binary labels 0/1/2 → Class 0 and 3/4 → Class 1.</p>
<h2>Configuration</h2><pre>{html.escape(json.dumps(config, ensure_ascii=False, indent=2))}</pre>
<h2>Integrity checks</h2><pre>{html.escape(json.dumps(integrity, ensure_ascii=False, indent=2))}</pre>
<h2>Final Mean ± Std</h2>{html_table(final_rows, list(final_rows[0]))}
<h2>Fold-level results</h2>{html_table(rows, fold_columns)}
<h2>Figures</h2>{images}
</body></html>"""
    (output_root / "report_step_02_N.html").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root != SCRIPT_DIR:
        raise ValueError(f"output-root must be the Step 02_N directory: {SCRIPT_DIR}")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve()

    reference_data_root = (
        Path(args.reference_data_root).expanduser().resolve()
        if args.reference_data_root
        else None
    )
    folded, image_lookup, integrity = run_integrity_checks(
        args.manifest, data_root, reference_data_root
    )
    write_json(output_root / "integrity_report.json", integrity)
    if not integrity["passed"]:
        raise RuntimeError("pre-training integrity checks failed; see integrity_report.json")

    device = resolve_device(args.device)
    run_smoke(folded, image_lookup, data_root, device, args.num_workers, output_root)
    config = {
        "experiment_name": STEP_NAME,
        "benchmark_type": "single_modality_ablation",
        "task": "binary",
        "label_strategy": "dynamic_0_1_2_to_0_3_4_to_1",
        "patient_isolation": False,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "data_root": str(data_root),
        "reference_data_root": str(reference_data_root) if reference_data_root else None,
        "preprocessed_cache": "384x384_jpeg",
        "output_root": str(output_root),
        "fold_seed": FOLD_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "modalities": list(MODALITIES),
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "model": MODEL_NAME,
        "pretrained": True,
        "input_channels": 3,
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
    write_json(output_root / "config_step_02_N.json", config)
    split_frame = folded[["sample_id", "fold"]].copy()
    split_frame.to_csv(output_root / "folds_step_02_N.csv", index=False)
    if args.mode == "smoke":
        print("SMOKE_TEST_OK")
        return

    rows: list[dict[str, object]] = []
    for modality in MODALITIES:
        for training_seed in TRAINING_SEEDS:
            for fold, train_raw, eval_raw in iter_folds(folded, patient_isolation=False):
                train_frame = filter_available_samples(train_raw, image_lookup, (modality,))
                eval_frame = filter_available_samples(eval_raw, image_lookup, (modality,))
                print(f"START modality={modality} seed={training_seed} fold={fold}", flush=True)
                row = train_one(
                    modality,
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
                write_json(output_root / "metrics_step_02_N.json", {"fold_results": rows})
                save_csv(output_root / "metrics_step_02_N.csv", rows)
                print(
                    f"DONE modality={modality} seed={training_seed} fold={fold} "
                    f"macro_f1={row['macro_f1']:.6f}",
                    flush=True,
                )

    seed_summaries, final_summaries = summarize(rows)
    metrics_payload = {
        "integrity": integrity,
        "fold_results": rows,
        "seed_summaries": seed_summaries,
        "final_summaries": final_summaries,
    }
    write_json(output_root / "metrics_step_02_N.json", metrics_payload)
    save_csv(output_root / "metrics_step_02_N.csv", rows)
    write_json(output_root / "seed_summaries_step_02_N.json", seed_summaries)
    write_json(output_root / "final_summary_step_02_N.json", final_summaries)
    figures = make_figures(final_summaries, output_root, rows)
    write_report(output_root, config, integrity, rows, seed_summaries, final_summaries, figures)
    print("STEP_02_N_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
