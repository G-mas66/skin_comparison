"""Shared training / evaluation / reporting pipeline for the MR ROI Steps 12-19.

Every step is 3 Training Seeds x 5 Folds = 15 independent runs with the fixed
protocol configuration (ResNet50, AdamW, lr 1e-4, wd 1e-4, batch 32, 50
epochs, no scheduler, no early stopping, CrossEntropyLoss; Soft Label steps
use the fixed soft-target cross entropy). Aggregation follows PROTOCOL.md
S12: per-seed five-fold Mean +- Std first, then Mean +- Std over the three
seed means. Figures follow S13 (Mean +- Std bars, per-class bars, confusion
matrix) and every step writes a self-contained HTML report (S13.6).
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import model as model_lib
from dataset import (
    DEFAULT_AUGMENTATION,
    EXCLUDED_IDS,
    INPUT_SIZE,
    N_FOLDS,
    SEED,
    load_manifest,
    make_five_folds,
    iter_folds,
)
from roi_common import (
    INPUT_VARIANTS,
    ROIDataset,
    SOFT_LABEL_MATRIX,
    build_roi_record,
    compute_mean_fill,
    worker_init_fn,
)

TRAINING_SEEDS = [42, 3407, 2026]
NUM_CLASSES = 5
SCHEME_COLORS = {
    "Whole Image": "#264653",
    "ROI Mask": "#2A9D8F",
    "ROI Crop": "#E9C46A",
    "ROI + 20% Context": "#E76F51",
    "Background Only": "#6D597A",
}

FIXED_TRAIN_CONFIG = {
    "model": "resnet50",
    "pretrained": True,
    "input_channels": 3,
    "num_classes": NUM_CLASSES,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": None,
    "early_stopping": False,
    "loss": "CrossEntropyLoss",
    "dropout": 0.3,
    "fold_seed": SEED,
    "training_seeds": TRAINING_SEEDS,
    "augmentation": DEFAULT_AUGMENTATION,
    "evaluation_checkpoint": "Epoch 50 last.pth",
}


# --------------------------------------------------------------------------- #
# Seeding                                                                     #
# --------------------------------------------------------------------------- #

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Accuracy / Macro-F1 / MAE / per-class P-R-F1 / confusion matrix."""

    assert len(y_true) == len(y_pred) and y_true, "empty predictions"
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    supports = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tps = np.diag(matrix)
    metrics: dict = {}
    f1_scores = []
    for class_id in range(NUM_CLASSES):
        precision = float(tps[class_id] / predicted[class_id]) if predicted[class_id] else 0.0
        recall = float(tps[class_id] / supports[class_id]) if supports[class_id] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
        metrics[f"class_{class_id}_precision"] = precision
        metrics[f"class_{class_id}_recall"] = recall
        metrics[f"class_{class_id}_f1"] = f1
    metrics["accuracy"] = float(tps.sum() / matrix.sum())
    metrics["macro_f1"] = float(np.mean(f1_scores))
    metrics["mae"] = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["support"] = supports.tolist()
    return metrics


METRIC_KEYS = ["accuracy", "macro_f1", "mae"] + [
    f"class_{class_id}_{kind}"
    for class_id in range(NUM_CLASSES)
    for kind in ("precision", "recall", "f1")
]


def aggregate_records(fold_records: list[dict]) -> dict:
    """Per-seed 5-fold summaries plus the final 3-seed summary."""

    seed_summaries = []
    for seed in TRAINING_SEEDS:
        seed_records = [r for r in fold_records if r["training_seed"] == seed]
        if len(seed_records) != N_FOLDS:
            raise ValueError(f"seed {seed} has {len(seed_records)} fold records")
        summary = {"training_seed": seed, "fold_count": N_FOLDS, "metrics": {}}
        for key in METRIC_KEYS:
            values = [float(r[key]) for r in seed_records]
            summary["metrics"][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        seed_summaries.append(summary)

    seed_means = {key: [s["metrics"][key]["mean"] for s in seed_summaries] for key in METRIC_KEYS}
    final_summary = {
        "training_seed_count": len(TRAINING_SEEDS),
        "scheme": fold_records[0].get("scheme", fold_records[0].get("step")),
        "label_strategy": fold_records[0]["label_strategy"],
        "metrics": {
            key: {"mean": float(np.mean(seed_means[key])), "std": float(np.std(seed_means[key]))}
            for key in METRIC_KEYS
        },
    }
    return {"seed_summaries": seed_summaries, "final_summary": final_summary}


# --------------------------------------------------------------------------- #
# References                                                                  #
# --------------------------------------------------------------------------- #

def load_reference_records(path: str | Path, modality: str = "MR") -> dict:
    """Load a whole-image baseline (step_05_N / step_08_N) MR reference."""

    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = [r for r in payload["fold_results"] if r.get("modality") == modality]
    if len(records) != 15:
        raise ValueError(f"{path}: expected 15 MR fold records, found {len(records)}")
    return {
        "label_strategy": records[0]["label_strategy"],
        "fold_records": records,
        "integrity": payload.get("integrity", {}),
        "final_summary": next(
            (f for f in payload["final_summaries"] if f.get("modality") == modality), None
        ),
    }


def load_step_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["fold_results"]
    if len(records) != 15:
        raise ValueError(f"{path}: expected 15 fold records, found {len(records)}")
    return records


# --------------------------------------------------------------------------- #
# Integrity checks (PROTOCOL.md S17 + ROI additions)                          #
# --------------------------------------------------------------------------- #

def run_integrity_checks(
    folded: pd.DataFrame,
    roi_lookup: dict,
    reference_fold_structure: list[dict] | None,
    variant: str,
    roi_stats_path: str | Path | None = None,
) -> dict:
    errors: list[str] = []

    frame = load_manifest(folded) if "fold" not in folded.columns else folded
    excluded_present = sorted(EXCLUDED_IDS & set(frame["capture_id"].astype(int)))
    if excluded_present:
        errors.append(f"excluded IDs still present: {excluded_present}")
    if frame["sample_id"].duplicated().any():
        errors.append("duplicate sample IDs in manifest")

    # the ROI-valid sample set must equal the mainline MR sample set exactly
    roi_ids = set(roi_lookup)
    manifest_ids = set(frame["sample_id"].astype(str))
    if roi_ids != manifest_ids:
        errors.append(
            f"ROI sample set differs from mainline: missing={sorted(manifest_ids - roi_ids)[:5]} "
            f"extra={sorted(roi_ids - manifest_ids)[:5]}"
        )

    # per-fold: no train/eval sample overlap; class/count bookkeeping
    fold_stats = []
    for fold_number, train_frame, eval_frame in iter_folds(frame, patient_isolation=False):
        overlap = set(train_frame["sample_id"]) & set(eval_frame["sample_id"])
        if overlap:
            errors.append(f"fold {fold_number}: train/eval sample overlap {len(overlap)}")
        if set(frame["label"].unique()) != set(range(5)):
            errors.append("labels are not exactly 0-4")
        fold_stats.append(
            {
                "fold": fold_number,
                "train_count": int(len(train_frame)),
                "evaluation_count": int(len(eval_frame)),
                "train_class_counts": {str(k): int(v) for k, v in train_frame["label"].value_counts().sort_index().items()},
                "evaluation_class_counts": {str(k): int(v) for k, v in eval_frame["label"].value_counts().sort_index().items()},
                "sample_overlap": 0,
            }
        )

    # fold structure must match the mainline N-route reference exactly
    if reference_fold_structure is not None:
        for got, want in zip(fold_stats, reference_fold_structure):
            if got["evaluation_count"] != want["evaluation_count"]:
                errors.append(f"fold {got['fold']}: eval count {got['evaluation_count']} != reference {want['evaluation_count']}")
            if got["evaluation_class_counts"] != want["evaluation_class_counts"]:
                errors.append(f"fold {got['fold']}: eval class counts differ from mainline reference")

    # ROI-specific checks: record legality from the Step 11 ROI manifest
    empty_roi, empty_bbox = [], []
    for sample_id, record in roi_lookup.items():
        if record.n_red_polygons < 1:
            empty_roi.append(sample_id)
            continue
        x0, y0, x1, y1 = record.bbox
        if x1 <= x0 or y1 <= y0:
            empty_bbox.append(sample_id)
    if empty_roi:
        errors.append(f"samples without a valid red polygon: {empty_roi[:5]}")
    if empty_bbox:
        errors.append(f"samples with an empty bounding box: {empty_bbox[:5]}")
    if variant not in INPUT_VARIANTS:
        errors.append(f"unknown input variant {variant}")

    roi_fold_stats: list | None = None
    if roi_stats_path is not None and Path(roi_stats_path).exists():
        stats = pd.read_csv(roi_stats_path)
        total_pixels = INPUT_SIZE * INPUT_SIZE
        illegal = stats[(stats["mask_area"] <= 0) | (stats["mask_area"] >= total_pixels)]
        if len(illegal):
            errors.append(f"{len(illegal)} samples with all-zero / all-full ROI mask: {illegal['sample_id'].head(5).tolist()}")
        stats = stats.set_index("sample_id")
        roi_fold_stats = []
        for fold_number, train_frame, eval_frame in iter_folds(frame, patient_isolation=False):
            train_stats = stats.loc[stats.index.isin(train_frame["sample_id"])]
            eval_stats = stats.loc[stats.index.isin(eval_frame["sample_id"])]
            roi_fold_stats.append({
                "fold": fold_number,
                "train_polygon_count_mean": float(train_stats["n_red_polygons"].mean()),
                "eval_polygon_count_mean": float(eval_stats["n_red_polygons"].mean()),
                "train_area_ratio_median": float(train_stats["area_ratio"].median()),
                "eval_area_ratio_median": float(eval_stats["area_ratio"].median()),
                "train_area_ratio_min": float(train_stats["area_ratio"].min()),
                "train_area_ratio_max": float(train_stats["area_ratio"].max()),
            })

    if errors:
        raise RuntimeError("DATA INTEGRITY CHECK FAILED: " + " | ".join(errors))
    return {
        "passed": True,
        "errors": errors,
        "patient_isolation": False,
        "patient_overlap_check": "not_required_for_patient_isolation_false",
        "fold_stats": fold_stats,
        "roi_sample_count": len(roi_lookup),
        "roi_fold_stats": roi_fold_stats,
        "task": "five_class",
        "num_classes": NUM_CLASSES,
    }


# --------------------------------------------------------------------------- #
# One training run (one seed x one fold)                                      #
# --------------------------------------------------------------------------- #

def train_one_run(
    config: dict,
    run: dict,
) -> dict:
    """Train and evaluate one (training seed, fold) run; returns the record."""

    seed = int(run["training_seed"])
    fold_number = int(run["fold"])
    set_all_seeds(seed)
    device = torch.device(config["device"])

    train_dataset = ROIDataset(
        frame=run["train_frame"],
        roi_lookup=run["roi_lookup"],
        training=True,
        variant=config["input_variant"],
        mean_fill=run["mean_fill"],
        mask_dir=run.get("mask_dir"),
    )
    eval_dataset = ROIDataset(
        frame=run["eval_frame"],
        roi_lookup=run["roi_lookup"],
        training=False,
        variant=config["input_variant"],
        mean_fill=run["mean_fill"],
        mask_dir=run.get("mask_dir"),
    )
    if config.get("smoke"):
        train_dataset.frame = train_dataset.frame.head(min(96, len(train_dataset))).reset_index(drop=True)
        eval_dataset.frame = eval_dataset.frame.head(min(96, len(eval_dataset))).reset_index(drop=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    torch_module = model_lib.ResNet50Classifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=config["dropout"],
        input_channels=3,
    )
    torch_module = torch_module.to(device)
    parameter_count = model_lib.count_parameters(torch_module)
    optimizer = torch.optim.AdamW(
        torch_module.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    soft_label = config["label_strategy"] == "Soft Label"
    soft_matrix = torch.tensor(SOFT_LABEL_MATRIX, device=device)

    epochs = 1 if config.get("smoke") else config["epochs"]
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / ("smoke_train.log" if config.get("smoke") else "train.log")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"run: seed={seed} fold={fold_number} variant={config['input_variant']} label={config['label_strategy']}\n")
        log.write(f"train samples={len(train_dataset)} eval samples={len(eval_dataset)} epochs={epochs}\n")
        for epoch in range(1, epochs + 1):
            torch_module.train()
            started = time.time()
            losses = []
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad()
                logits = torch_module(images)
                if soft_label:
                    q = soft_matrix[targets]
                    log_probs = torch.log_softmax(logits, dim=1)
                    loss = -(q * log_probs).sum(dim=1).mean()
                else:
                    loss = nn.functional.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            log.write(
                f"epoch {epoch:02d}/{epochs} mean_loss={np.mean(losses):.4f} "
                f"lr={config['learning_rate']} elapsed={time.time() - started:.1f}s\n"
            )
            log.flush()

        # evaluation on the held-out fold with the epoch-50 weights
        torch_module.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in eval_loader:
                images = batch["image"].to(device, non_blocking=True)
                logits = torch_module(images)
                y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
                y_true.extend(batch["target"].tolist())
        metrics = compute_metrics(y_true, y_pred)
        with open(run_dir / ("smoke_eval.log" if config.get("smoke") else "eval.log"), "w", encoding="utf-8") as elog:
            elog.write(f"eval samples={len(y_true)}\n")
            elog.write(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=1) + "\n")

    if not config.get("smoke"):
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(torch_module.state_dict(), checkpoint_dir / "last.pth")

    record = {
        "step": config["experiment_name"],
        "task": "five_class",
        "label_strategy": config["label_strategy"],
        "modality": "MR",
        "input_region": config["input_variant"],
        "fold": fold_number,
        "training_seed": seed,
        "train_count": int(len(train_dataset)),
        "evaluation_count": int(len(eval_dataset)),
        "train_class_counts": {str(k): int(v) for k, v in run["train_frame"]["label"].value_counts().sort_index().items()},
        "evaluation_class_counts": {str(k): int(v) for k, v in run["eval_frame"]["label"].value_counts().sort_index().items()},
        "parameter_count": parameter_count,
    }
    record.update({k: v for k, v in metrics.items() if k != "support"})
    with open(run_dir / ("smoke_metrics.json" if config.get("smoke") else "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, ensure_ascii=False)
    return record


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #

def _scheme_color(name: str) -> str:
    for key, color in SCHEME_COLORS.items():
        if name.startswith(key):
            return color
    return "#8D99AE"


def _value_label(metric: str) -> str:
    return {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "mae": "MAE"}.get(metric, metric)


def fig_metric_bars(step_dir: Path, schemes: list[dict], metric: str, suffix: str) -> None:
    """Main PROTOCOL S13.2 horizontal bar figure for one metric."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s["name"] for s in schemes]
    means = [s["final"]["metrics"][metric]["mean"] for s in schemes]
    stds = [s["final"]["metrics"][metric]["std"] for s in schemes]
    colors = [_scheme_color(n) for n in names]
    fig, axis = plt.subplots(figsize=(8.6, 0.72 * len(schemes) + 1.7))
    bars = axis.barh(names, means, xerr=stds, color=colors, edgecolor="#111827", height=0.58, capsize=5)
    axis.set_xlabel(f"{_value_label(metric)} (Mean ± Std over 3 Training Seed means)", fontsize=10)
    axis.set_title(f"{schemes[0]['title']} — {_value_label(metric)} (5-fold CV, 3 seeds)", fontsize=12)
    axis.grid(axis="x", color="#D8DEE8", linewidth=0.8)
    axis.set_axisbelow(True)
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    pad = max(0.02, (hi - lo) * 0.35)
    axis.set_xlim(max(0.0, lo - pad), min(1.0 if metric != "mae" else hi + pad, hi + pad))
    for bar, mean, std in zip(bars, means, stds):
        axis.text(bar.get_width() + pad * 0.08, bar.get_y() + bar.get_height() / 2,
                  f"{mean:.4f} ± {std:.4f}", va="center", fontsize=9)
    axis.invert_yaxis()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(step_dir / "figures" / f"{suffix}_{metric}.{extension}", dpi=170)
    plt.close(fig)


def fig_seed_detail(step_dir: Path, schemes: list[dict], suffix: str) -> None:
    """Per-seed grouped bars: height = seed's 5-fold mean, error = 5-fold std."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "macro_f1", "mae"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(TRAINING_SEEDS))
    width = 0.8 / len(schemes)
    for axis, metric in zip(axes, metrics):
        for scheme_index, scheme in enumerate(schemes):
            means = [s["metrics"][metric]["mean"] for s in scheme["seed_summaries"]]
            stds = [s["metrics"][metric]["std"] for s in scheme["seed_summaries"]]
            offset = (scheme_index - len(schemes) / 2 + 0.5) * width
            axis.bar(
                x + offset, means, width, yerr=stds, capsize=3,
                label=scheme["name"], color=_scheme_color(scheme["name"]), edgecolor="#111827",
            )
        axis.set_xticks(x)
        axis.set_xticklabels([f"Seed {seed}" for seed in TRAINING_SEEDS])
        axis.set_ylabel(_value_label(metric))
        axis.set_title(f"{_value_label(metric)} per Training Seed (Mean ± Std over 5 folds)")
        axis.grid(axis="y", color="#D8DEE8", linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(schemes[0]["title"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(step_dir / "figures" / f"{suffix}_seed_detail.{extension}", dpi=170)
    plt.close(fig)


def fig_per_class(step_dir: Path, schemes: list[dict], suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(NUM_CLASSES)
    width = 0.8 / len(schemes)
    for axis, (kind, kind_label) in zip(axes, kinds):
        for scheme_index, scheme in enumerate(schemes):
            means = [scheme["final"]["metrics"][f"class_{class_id}_{kind}"]["mean"] for class_id in range(NUM_CLASSES)]
            stds = [scheme["final"]["metrics"][f"class_{class_id}_{kind}"]["std"] for class_id in range(NUM_CLASSES)]
            offset = (scheme_index - len(schemes) / 2 + 0.5) * width
            axis.bar(
                x + offset, means, width, yerr=stds, capsize=3,
                label=scheme["name"], color=_scheme_color(scheme["name"]), edgecolor="#111827",
            )
        axis.set_xticks(x)
        axis.set_xticklabels([f"Class {class_id}" for class_id in range(NUM_CLASSES)])
        axis.set_ylim(0, 1)
        axis.set_title(f"Per-class {kind_label} (Mean ± Std, 3 seeds)")
        axis.grid(axis="y", color="#D8DEE8", linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    fig.suptitle(schemes[0]["title"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(step_dir / "figures" / f"{suffix}_per_class.{extension}", dpi=170)
    plt.close(fig)


def fig_confusion(step_dir: Path, records: list[dict], scheme_name: str, suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = np.asarray([r["confusion_matrix"] for r in records], dtype=float)
    mean_matrix = matrices.mean(axis=0)
    fig, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(mean_matrix, cmap="Blues", vmin=0)
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            axis.text(column, row, f"{mean_matrix[row, column]:.1f}", ha="center", va="center",
                      color="white" if mean_matrix[row, column] > mean_matrix.max() * 0.6 else "#172033",
                      fontsize=9)
    axis.set_xticks(range(NUM_CLASSES))
    axis.set_yticks(range(NUM_CLASSES))
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"{scheme_name}: mean confusion matrix over 15 runs")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(step_dir / "figures" / f"{suffix}_confusion.{extension}", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# HTML report                                                                 #
# --------------------------------------------------------------------------- #

def _figure_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{path.stem}"/>'


def _html_table(rows: list[dict], columns: list[tuple[str, str]], float_format: str = "{:.4f}") -> str:
    header = "".join(f"<th>{title}</th>" for _, title in columns)
    body_rows = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"<td>{float_format.format(value)}</td>")
            else:
                cells.append(f"<td>{value}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_html_report(
    step_dir: Path,
    step_id: str,
    config: dict,
    integrity: dict,
    records: list[dict],
    seed_summaries: list[dict],
    final_summary: dict,
    schemes: list[dict],
    extra_notes: list[str],
) -> None:
    from report_style import REPORT_CSS

    figure_dir = step_dir / "figures"
    config_rows = [{"item": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
                   for key, value in config.items()]
    fold_columns = [("fold", "Fold"), ("training_seed", "Seed"), ("train_count", "Train n"),
                    ("evaluation_count", "Eval n"), ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
                    ("mae", "MAE")]
    summary_columns = [("training_seed", "Seed"), ("accuracy_mean", "Acc Mean"), ("accuracy_std", "Acc Std"),
                       ("macro_f1_mean", "F1 Mean"), ("macro_f1_std", "F1 Std"),
                       ("mae_mean", "MAE Mean"), ("mae_std", "MAE Std")]
    seed_rows = []
    for summary in seed_summaries:
        seed_rows.append({
            "training_seed": summary["training_seed"],
            "accuracy_mean": summary["metrics"]["accuracy"]["mean"],
            "accuracy_std": summary["metrics"]["accuracy"]["std"],
            "macro_f1_mean": summary["metrics"]["macro_f1"]["mean"],
            "macro_f1_std": summary["metrics"]["macro_f1"]["std"],
            "mae_mean": summary["metrics"]["mae"]["mean"],
            "mae_std": summary["metrics"]["mae"]["std"],
        })
    final_rows = []
    for scheme in schemes:
        final = scheme["final"]["metrics"]
        final_rows.append({
            "scheme": scheme["name"],
            "accuracy": final["accuracy"]["mean"], "accuracy_std": final["accuracy"]["std"],
            "macro_f1": final["macro_f1"]["mean"], "macro_f1_std": final["macro_f1"]["std"],
            "mae": final["mae"]["mean"], "mae_std": final["mae"]["std"],
        })
    comparison_columns = [("scheme", "Scheme"), ("accuracy", "Acc Mean"), ("accuracy_std", "Acc Std"),
                          ("macro_f1", "F1 Mean"), ("macro_f1_std", "F1 Std"),
                          ("mae", "MAE Mean"), ("mae_std", "MAE Std")]
    figures_html = "".join(
        _figure_tag(figure_dir / f"{name}.png")
        for name in sorted(path.stem for path in figure_dir.glob("*.png"))
    )
    notes_html = "".join(f"<li>{note}</li>" for note in extra_notes)
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>{step_id} report</title><style>{REPORT_CSS}</style></head>
<body>
<h1>{step_id} — MR ROI Benchmark Report (Patient Isolation = False)</h1>
<h2>Experiment Configuration</h2>
{_html_table(config_rows, [("item", "Item"), ("value", "Value")], "{:.6g}")}
<h2>Data Integrity Checks</h2>
<p><b>passed:</b> {integrity['passed']} | errors: {len(integrity['errors'])} | ROI samples: {integrity['roi_sample_count']}</p>
{_html_table(integrity['fold_stats'], [("fold", "Fold"), ("train_count", "Train n"), ("evaluation_count", "Eval n"), ("train_class_counts", "Train class counts"), ("evaluation_class_counts", "Eval class counts")], "{:.0f}")}
<h2>Fold-level Records (15 runs)</h2>
{_html_table(records, fold_columns)}
<h2>Per-seed Five-fold Summaries</h2>
{_html_table(seed_rows, summary_columns)}
<h2>Final Comparison (Mean ± Std over 3 seed means)</h2>
{_html_table(final_rows, comparison_columns)}
<h2>Figures</h2>
{figures_html}
<h2>Notes</h2>
<ul>{notes_html}</ul>
</body></html>"""
    (step_dir / f"report_{step_id}.html").write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Step driver                                                                 #
# --------------------------------------------------------------------------- #

def parse_step_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--manifest", default="/root/autodl-tmp/data_all_384/manifest.csv")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data_all_384")
    parser.add_argument("--roi-cache", default="/root/autodl-tmp/data_all_384/roi_cache")
    parser.add_argument("--repo-root", default="/root/autodl-tmp/skin_comparison")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def prepare_roi(args: argparse.Namespace, data_root: Path) -> tuple[pd.DataFrame, dict]:
    """Load manifest, reproduce the fixed N-route folds, build ROI lookup."""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    folded = make_five_folds(args.manifest, patient_isolation=False, seed=SEED)
    roi_lookup: dict = {}
    cache_dir = Path(args.roi_cache)
    mask_dir = cache_dir / "masks"
    for path in sorted(data_root.glob("MR*.json")):
        stem = path.stem
        sample_id = stem
        for suffix in ("_1", "_2"):
            if sample_id.endswith(suffix):
                sample_id = sample_id[: -len(suffix)]
        image_path = data_root / f"{sample_id}.png"
        if not image_path.exists():
            continue
        record = build_roi_record(path, image_path)
        roi_lookup[record.sample_id] = record
    return folded, roi_lookup


def run_step(config: dict) -> None:
    """Full driver for one ROI step (smoke mode or the 15-run benchmark)."""

    step_id = config["experiment_name"]
    # Output dir must be the *step script's* directory (train_common.py itself
    # lives in the repo root), never the repo root.
    raw_dir = config.get("step_dir")
    step_dir = Path(raw_dir).resolve() if raw_dir else Path(sys.argv[0]).resolve().parent
    args = parse_step_args(config["description"])
    data_root = Path(args.data_root)
    repo_root = Path(args.repo_root)

    full_config = copy.deepcopy({**FIXED_TRAIN_CONFIG, **config})
    full_config.update(
        {
            "manifest": args.manifest,
            "data_root": args.data_root,
            "roi_cache": args.roi_cache,
            "repo_root": str(repo_root),
            "output_root": str(step_dir),
            "device": args.device,
            "num_workers": args.num_workers,
            "patient_isolation": False,
            "input_resolution": "384x384",
            "smoke": bool(args.smoke),
        }
    )
    (step_dir / "logs").mkdir(parents=True, exist_ok=True)
    (step_dir / "figures").mkdir(parents=True, exist_ok=True)
    for fold in range(1, N_FOLDS + 1):
        (step_dir / f"fold_{fold:02d}").mkdir(exist_ok=True)

    folded, roi_lookup = prepare_roi(args, data_root)

    reference = load_reference_records(repo_root / "non_isolated" / "baseline_refs" / "metrics_step_05_N.json")
    integrity = run_integrity_checks(
        folded,
        roi_lookup,
        reference["integrity"]["modality_stats"]["MR"],
        config["input_variant"],
        roi_stats_path=Path(args.roi_cache) / "roi_manifest.csv",
    )
    print(f"[{step_id}] integrity checks passed: {len(roi_lookup)} ROI samples, 5 folds")

    cache_dir = Path(args.roi_cache)
    mask_dir = cache_dir / "masks"
    needs_fill = config["input_variant"] in ("mask", "background")
    mean_fills = {}
    if needs_fill:
        for fold_number, train_frame, _ in iter_folds(folded, patient_isolation=False):
            path = cache_dir / f"mean_fill_fold_{fold_number:02d}.npy"
            if not path.exists():
                raise FileNotFoundError(f"mean fill cache missing: {path}; run step_11_N first")
            mean_fills[fold_number] = np.load(path)

    if args.smoke:
        full_config["smoke_note"] = "1 epoch, capped 96 train/96 eval samples, no checkpoints"
        run = next(iter_folds(folded, patient_isolation=False))
        fold_number, train_frame, eval_frame = run
        smoke_dir = step_dir / "logs" / "smoke"
        record = train_one_run(
            full_config,
            {
                "training_seed": 42,
                "fold": 1,
                "train_frame": train_frame,
                "eval_frame": eval_frame,
                "roi_lookup": roi_lookup,
                "mean_fill": mean_fills.get(1),
                "mask_dir": mask_dir,
                "run_dir": smoke_dir,
            },
        )
        print(f"[{step_id}] SMOKE OK acc={record['accuracy']:.4f} macro_f1={record['macro_f1']:.4f} mae={record['mae']:.4f}")
        with open(step_dir / "logs" / "smoke_summary.json", "w", encoding="utf-8") as handle:
            json.dump({"passed": True, "config": full_config, "record": record}, handle, indent=1, ensure_ascii=False)
        return

    with open(step_dir / f"config_{step_id}.json", "w", encoding="utf-8") as handle:
        json.dump(full_config, handle, indent=2, ensure_ascii=False)
    with open(step_dir / "integrity_report.json", "w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2, ensure_ascii=False)

    records: list[dict] = []
    for seed in TRAINING_SEEDS:
        for fold_number, train_frame, eval_frame in iter_folds(folded, patient_isolation=False):
            run_dir = step_dir / f"fold_{fold_number:02d}" / f"seed_{seed}"
            record = train_one_run(
                full_config,
                {
                    "training_seed": seed,
                    "fold": fold_number,
                    "train_frame": train_frame,
                    "eval_frame": eval_frame,
                    "roi_lookup": roi_lookup,
                    "mean_fill": mean_fills.get(fold_number),
                    "mask_dir": mask_dir,
                    "run_dir": run_dir,
                },
            )
            records.append(record)
            print(
                f"[{step_id}] seed={seed} fold={fold_number} acc={record['accuracy']:.4f} "
                f"macro_f1={record['macro_f1']:.4f} mae={record['mae']:.4f}"
            )

    aggregation = aggregate_records(records)
    metrics_payload = {
        "integrity": integrity,
        "fold_results": records,
        "seed_summaries": aggregation["seed_summaries"],
        "final_summaries": [aggregation["final_summary"]],
    }
    with open(step_dir / f"metrics_{step_id}.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=1, ensure_ascii=False)
    frame_csv = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"} for r in records])
    frame_csv.to_csv(step_dir / f"metrics_{step_id}.csv", index=False)
    with open(step_dir / f"seed_summaries_{step_id}.json", "w", encoding="utf-8") as handle:
        json.dump(aggregation["seed_summaries"], handle, indent=1, ensure_ascii=False)
    with open(step_dir / f"final_summary_{step_id}.json", "w", encoding="utf-8") as handle:
        json.dump(aggregation["final_summary"], handle, indent=1, ensure_ascii=False)

    print(f"[{step_id}] all 15 runs complete; building figures and report")
    return_dict = {
        "records": records,
        "aggregation": aggregation,
        "step_dir": step_dir,
        "full_config": full_config,
        "integrity": integrity,
    }
    globals()["_LAST_STEP_RESULT"] = return_dict
    finalize_step(return_dict)


def finalize_step(result: dict) -> None:
    """Build comparison schemes, figures and the HTML report for a step."""

    step_id = result["full_config"]["experiment_name"]
    step_dir = result["step_dir"]
    records = result["records"]
    aggregation = result["aggregation"]
    config = result["full_config"]

    schemes = build_schemes(config, records, aggregation)
    scheme_own = next(s for s in schemes if s.get("own"))
    for metric in ("accuracy", "macro_f1", "mae"):
        fig_metric_bars(step_dir, schemes, metric, "benchmark")
    fig_seed_detail(step_dir, schemes, "benchmark")
    fig_per_class(step_dir, schemes, "benchmark")
    fig_confusion(step_dir, records, scheme_own["name"], "own")

    write_html_report(
        step_dir,
        step_id,
        config,
        result["integrity"],
        records,
        aggregation["seed_summaries"],
        aggregation["final_summary"],
        schemes,
        config.get("report_notes", []),
    )
    print(f"[{step_id}] report written: report_{step_id}.html")


def build_schemes(config: dict, records: list[dict], aggregation: dict) -> list[dict]:
    """Assemble the comparison schemes declared by the step config."""

    repo_root = Path(config["repo_root"])
    refs_dir = repo_root / "non_isolated" / "baseline_refs"
    title = config["report_title"]
    own_name = config["scheme_name"]
    own_scheme = {
        "name": own_name,
        "title": title,
        "own": True,
        "fold_records": records,
        "seed_summaries": aggregation["seed_summaries"],
        "final": aggregation["final_summary"],
    }
    schemes = []
    for spec in config.get("reference_schemes", []):
        kind = spec["kind"]
        if kind == "whole_image":
            reference = load_reference_records(refs_dir / spec["file"])
            ref_aggregation = aggregate_records(reference["fold_records"])
            schemes.append({
                "name": spec["name"],
                "title": title,
                "fold_records": reference["fold_records"],
                "seed_summaries": ref_aggregation["seed_summaries"],
                "final": ref_aggregation["final_summary"],
                "reference_source": spec["file"],
            })
        elif kind == "sibling_step":
            sibling_records = load_step_records(repo_root / "non_isolated" / spec["step_dir"] / f"metrics_{spec['step_id']}.json")
            sibling_aggregation = aggregate_records(sibling_records)
            schemes.append({
                "name": spec["name"],
                "title": title,
                "fold_records": sibling_records,
                "seed_summaries": sibling_aggregation["seed_summaries"],
                "final": sibling_aggregation["final_summary"],
                "reference_source": spec["step_id"],
            })
        else:
            raise ValueError(f"unknown reference scheme kind: {kind}")
    schemes.append(own_scheme)
    return schemes
