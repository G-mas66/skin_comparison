"""step 10 N: DeepLIFT and channel perturbation for selected step 09 N models."""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
STEP3_DIR = PROJECT_ROOT / "non_isolated" / "step_09_N"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEP3_DIR) not in sys.path:
    sys.path.insert(0, str(STEP3_DIR))

from dataset import INPUT_SIZE, SkinDataset, filter_available_samples, iter_folds  # noqa: E402
from model import build_classifier  # noqa: E402
from step_09_N import (  # noqa: E402
    BATCH_SIZE,
    FOLD_SEED,
    NUM_CLASSES,
    combination_name,
    combination_slug,
    metric_values,
    run_integrity_checks,
    worker_init_fn,
)
from non_isolated.report_style import (  # noqa: E402
    REPORT_CSS,
    BAR_EDGE_COLOR,
    combination_axis_label,
    modality_color,
    style_axis,
    style_bars,
)

STEP_NAME = "step_10_N"
MODEL_NAME = "resnet50"
TRAINING_SEEDS = (42, 3407, 2026)
NUM_WORKERS = 4
ANALYSIS_BATCH_SIZE = 1
PERTURBATION_BASELINE = 0.0
TARGET_RULE = "predicted_class"
COMBINATIONS = (
    ("M", "MR"),
    ("M", "MB", "MR"),
    ("M", "MB", "MP", "MR", "MUV"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-data-root")
    parser.add_argument("--reference-split", required=True)
    parser.add_argument("--step3-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--analysis-batch-size", type=int, default=ANALYSIS_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_output_root(output_root: Path) -> None:
    if output_root.name != STEP_NAME or output_root.parent.name != "non_isolated":
        raise ValueError(
            "output-root must end with non_isolated/step_10_N; "
            "pass an explicit result directory on the data disk"
        )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal RTX 5090 D run")
    device = torch.device(requested)
    if device.type == "cuda":
        device = torch.device(f"cuda:{device.index or 0}")
        torch.cuda.set_device(device.index)
    return device


def expected_step3_checkpoint(
    step3_root: Path, modalities: tuple[str, ...], fold: int, seed: int
) -> Path:
    return (
        step3_root
        / f"fold_{fold:02d}"
        / combination_slug(modalities)
        / f"seed_{seed}"
        / "checkpoints"
        / "last.pth"
    )


def validate_step3_artifacts(
    step3_root: Path, folded: pd.DataFrame
) -> dict[str, object]:
    errors: list[str] = []
    config_path = step3_root / "config_step_09_N.json"
    split_path = step3_root / "folds_step_09_N.csv"
    if not config_path.is_file():
        errors.append(f"missing step 09 N config: {config_path}")
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_config = {
            "patient_isolation": False,
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "model": MODEL_NAME,
            "pretrained": True,
            "num_classes": NUM_CLASSES,
            "batch_size": BATCH_SIZE,
            "epochs": 50,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": None,
            "early_stopping": False,
            "loss": "SoftCrossEntropy",
            "fold_seed": FOLD_SEED,
            "training_seeds": list(TRAINING_SEEDS),
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                errors.append(
                    f"step 09 N config mismatch for {key}: "
                    f"expected {expected!r}, found {config.get(key)!r}"
                )
    if not split_path.is_file():
        errors.append(f"missing step 09 N fixed split: {split_path}")
    else:
        saved_split = pd.read_csv(split_path)
        current_map = {
            str(sample_id): int(fold)
            for sample_id, fold in zip(folded["sample_id"], folded["fold"])
        }
        saved_map = {
            str(sample_id): int(fold)
            for sample_id, fold in zip(saved_split["sample_id"], saved_split["fold"])
        }
        if current_map != saved_map:
            errors.append("step 09 N fixed split differs from the validated split")

    checkpoint_records: list[dict[str, object]] = []
    for modalities in COMBINATIONS:
        for seed in TRAINING_SEEDS:
            for fold in range(1, 6):
                path = expected_step3_checkpoint(step3_root, modalities, fold, seed)
                if not path.is_file():
                    errors.append(f"missing checkpoint: {path}")
                    continue
                checkpoint_records.append(
                    {
                        "combination": combination_name(modalities),
                        "modalities": list(modalities),
                        "fold": fold,
                        "training_seed": seed,
                        "checkpoint": str(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    report = {
        "passed": not errors,
        "errors": errors,
        "source_step": "step_09_N",
        "selected_combinations": [list(item) for item in COMBINATIONS],
        "expected_checkpoint_count": len(COMBINATIONS) * len(TRAINING_SEEDS) * 5,
        "available_checkpoint_count": len(checkpoint_records),
        "checkpoints": checkpoint_records,
    }
    if errors:
        raise RuntimeError("step 09 N artifact checks failed: " + "; ".join(errors))
    return report


def prepare_deeplift_model(model: nn.Module) -> nn.Module:
    """Trace the model and give every ReLU call a distinct non-inplace module."""

    traced = torch.fx.symbolic_trace(model)
    relu_index = 0
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(str(node.target))
        if isinstance(module, nn.ReLU):
            name = f"_deeplift_relu_{relu_index}"
            traced.add_module(name, nn.ReLU(inplace=False))
            node.target = name
            relu_index += 1
    traced.graph.lint()
    traced.recompile()
    return traced


def remove_attribution_hooks(deeplift: DeepLift) -> None:
    remove_hooks = getattr(deeplift, "remove_hooks", None)
    if remove_hooks is not None:
        remove_hooks()
    else:
        deeplift._remove_hooks([])


def load_attribution_model(
    checkpoint_path: Path,
    modalities: tuple[str, ...],
    fold: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected = {
        "step": "step_09_N",
        "combination": combination_name(modalities),
        "modalities": list(modalities),
        "fold": fold,
        "training_seed": seed,
        "epoch": 50,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(
                f"checkpoint metadata mismatch for {checkpoint_path}: "
                f"{key} expected {value!r}, found {checkpoint.get(key)!r}"
            )
    model = build_classifier(
        name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        pretrained=True,
        dropout=0.3,
        input_channels=3 * len(modalities),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.eval()
    attribution_model = prepare_deeplift_model(model).to(device)
    attribution_model.eval()
    del model
    return attribution_model


def aggregate(values: Iterable[float]) -> dict[str, float]:
    values_list = [float(value) for value in values]
    if not values_list:
        raise ValueError("cannot aggregate an empty value list")
    return {
        "mean": float(mean(values_list)),
        "std": float(stdev(values_list)) if len(values_list) > 1 else 0.0,
    }


def make_analysis_loader(
    frame: pd.DataFrame,
    modalities: tuple[str, ...],
    image_lookup: dict[str, dict[int, Path]],
    data_root: Path,
    seed: int,
    num_workers: int,
    batch_size: int,
) -> DataLoader:
    dataset = SkinDataset(
        frame,
        data_root,
        modalities=modalities,
        training=False,
        input_size=INPUT_SIZE,
        augmentation={},
        image_lookup=image_lookup,
        include_metadata=False,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def analyze_model(
    modalities: tuple[str, ...],
    fold: int,
    seed: int,
    eval_frame: pd.DataFrame,
    image_lookup: dict[str, dict[int, Path]],
    data_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    num_workers: int,
    analysis_batch_size: int,
    spatial_sums: dict[str, dict[str, np.ndarray]],
    step3_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    name = combination_name(modalities)
    loader = make_analysis_loader(
        eval_frame,
        modalities,
        image_lookup,
        data_root,
        seed,
        num_workers,
        analysis_batch_size,
    )
    model = load_attribution_model(checkpoint_path, modalities, fold, seed, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    deeplift = DeepLift(model)
    targets: list[int] = []
    baseline_predictions: list[int] = []
    perturb_predictions = {modality: [] for modality in modalities}
    modality_scores: dict[str, list[float]] = {modality: [] for modality in modalities}
    modality_fractions: dict[str, list[float]] = {modality: [] for modality in modalities}
    for modality in modalities:
        spatial_sums.setdefault(name, {})
        spatial_sums[name].setdefault(
            modality, np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float64)
        )

    try:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch_targets = [int(value) for value in batch["target"].tolist()]
            with torch.no_grad():
                baseline_logits = model(images)
                predictions = baseline_logits.argmax(dim=1)
            attributions = deeplift.attribute(
                images,
                baselines=torch.zeros_like(images),
                target=predictions,
            )
            absolute = attributions.detach().abs().reshape(
                len(images), len(modalities), 3, INPUT_SIZE, INPUT_SIZE
            )
            scores = absolute.mean(dim=(2, 3, 4))
            fractions = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
            for index, modality in enumerate(modalities):
                modality_scores[modality].extend(scores[:, index].cpu().tolist())
                modality_fractions[modality].extend(fractions[:, index].cpu().tolist())
                spatial_sums[name][modality] += (
                    absolute[:, index].mean(dim=1).sum(dim=0).cpu().numpy()
                )

            targets.extend(batch_targets)
            baseline_predictions.extend(predictions.cpu().tolist())
            for index, modality in enumerate(modalities):
                perturbed = images.clone()
                start = index * 3
                perturbed[:, start : start + 3] = PERTURBATION_BASELINE
                with torch.no_grad():
                    perturbed_logits = model(perturbed)
                perturb_predictions[modality].extend(
                    perturbed_logits.argmax(dim=1).cpu().tolist()
                )
            del images, baseline_logits, predictions, attributions, absolute, scores, fractions
    finally:
        remove_attribution_hooks(deeplift)
        del deeplift, model, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_metrics, baseline_confusion = metric_values(targets, baseline_predictions)
    perturbation_rows: list[dict[str, object]] = []
    for modality in modalities:
        perturbed_metrics, perturbed_confusion = metric_values(
            targets, perturb_predictions[modality]
        )
        perturbation_rows.append(
            {
                "step": STEP_NAME,
                "combination": name,
                "modalities": list(modalities),
                "modality": modality,
                "fold": fold,
                "training_seed": seed,
                "evaluation_count": len(eval_frame),
                "baseline_accuracy": baseline_metrics["accuracy"],
                "baseline_macro_f1": baseline_metrics["macro_f1"],
                "perturbed_accuracy": perturbed_metrics["accuracy"],
                "perturbed_macro_f1": perturbed_metrics["macro_f1"],
                "delta_accuracy": perturbed_metrics["accuracy"] - baseline_metrics["accuracy"],
                "delta_macro_f1": perturbed_metrics["macro_f1"] - baseline_metrics["macro_f1"],
                "baseline_confusion_matrix": baseline_confusion,
                "perturbed_confusion_matrix": perturbed_confusion,
                "replacement": "zero_normalized_baseline",
            }
        )

    model_row = {
        "step": STEP_NAME,
        "source_step": "step_09_N",
        "combination": name,
        "modalities": list(modalities),
        "fold": fold,
        "training_seed": seed,
        "evaluation_count": len(eval_frame),
        "checkpoint": str(checkpoint_path.relative_to(step3_root)),
        "parameter_count": parameter_count,
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "baseline_confusion_matrix": baseline_confusion,
        "deeplift": {
            modality: {
                "mean_abs_attribution": float(mean(modality_scores[modality])),
                "std_abs_attribution": float(stdev(modality_scores[modality]))
                if len(modality_scores[modality]) > 1
                else 0.0,
                "mean_attribution_fraction": float(mean(modality_fractions[modality])),
            }
            for modality in modalities
        },
    }
    return model_row, perturbation_rows


def run_smoke(
    folded: pd.DataFrame,
    image_lookup: dict[str, dict[int, Path]],
    data_root: Path,
    step3_root: Path,
    device: torch.device,
    num_workers: int,
    analysis_batch_size: int,
    output_root: Path,
) -> None:
    fold_frame = folded.loc[folded["fold"] == 0].reset_index(drop=True).head(2)
    modalities = COMBINATIONS[0]
    checkpoint_path = expected_step3_checkpoint(step3_root, modalities, 1, TRAINING_SEEDS[0])
    loader = make_analysis_loader(
        fold_frame,
        modalities,
        image_lookup,
        data_root,
        TRAINING_SEEDS[0],
        num_workers,
        analysis_batch_size,
    )
    model = load_attribution_model(
        checkpoint_path, modalities, 1, TRAINING_SEEDS[0], device
    )
    deeplift = DeepLift(model)
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    with torch.no_grad():
        predictions = model(images).argmax(dim=1)
    attributions = deeplift.attribute(
        images, baselines=torch.zeros_like(images), target=predictions
    )
    expected_shape = (len(modalities) * 3, INPUT_SIZE, INPUT_SIZE)
    if tuple(images.shape[1:]) != expected_shape:
        raise RuntimeError(f"Step04_N smoke input shape is invalid: {images.shape}")
    if tuple(attributions.shape) != tuple(images.shape):
        raise RuntimeError(
            f"Step04_N smoke attribution shape is invalid: {attributions.shape}"
        )
    perturb = images.clone()
    perturb[:, :3] = PERTURBATION_BASELINE
    with torch.no_grad():
        perturbed_predictions = model(perturb).argmax(dim=1)
    input_shape = list(images.shape)
    attribution_shape = list(attributions.shape)
    baseline_prediction_values = [int(value) for value in predictions.cpu().tolist()]
    perturbed_prediction_values = [
        int(value) for value in perturbed_predictions.cpu().tolist()
    ]
    remove_attribution_hooks(deeplift)
    del model, deeplift, loader, batch, images, attributions, perturb
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_json(
        output_root / "logs" / "smoke_test.json",
        {
            "passed": True,
            "combination": combination_name(modalities),
            "fold": 1,
            "training_seed": TRAINING_SEEDS[0],
            "input_shape": input_shape,
            "attribution_shape": attribution_shape,
            "baseline": "zero_normalized_tensor",
            "target_rule": TARGET_RULE,
            "perturbation": "M_three_channel_zeroing",
            "baseline_predictions": baseline_prediction_values,
            "perturbed_predictions": perturbed_prediction_values,
        },
    )


def aggregate_results(
    model_rows: list[dict[str, object]], perturbation_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    deeplift_summary: list[dict[str, object]] = []
    for modalities in COMBINATIONS:
        name = combination_name(modalities)
        selected = [row for row in model_rows if row["combination"] == name]
        for modality in modalities:
            values = [row["deeplift"][modality] for row in selected]
            deeplift_summary.append(
                {
                    "combination": name,
                    "modality": modality,
                    "model_count": len(values),
                    "mean_abs_attribution": aggregate(
                        item["mean_abs_attribution"] for item in values
                    ),
                    "mean_attribution_fraction": aggregate(
                        item["mean_attribution_fraction"] for item in values
                    ),
                }
            )

    perturbation_summary: list[dict[str, object]] = []
    for modalities in COMBINATIONS:
        name = combination_name(modalities)
        for modality in modalities:
            selected = [
                row
                for row in perturbation_rows
                if row["combination"] == name and row["modality"] == modality
            ]
            perturbation_summary.append(
                {
                    "combination": name,
                    "modality": modality,
                    "model_count": len(selected),
                    "delta_accuracy": aggregate(
                        row["delta_accuracy"] for row in selected
                    ),
                    "delta_macro_f1": aggregate(
                        row["delta_macro_f1"] for row in selected
                    ),
                    "perturbed_accuracy": aggregate(
                        row["perturbed_accuracy"] for row in selected
                    ),
                    "perturbed_macro_f1": aggregate(
                        row["perturbed_macro_f1"] for row in selected
                    ),
                }
            )
    return deeplift_summary, perturbation_summary


def save_figures(
    deeplift_summary: list[dict[str, object]],
    perturbation_summary: list[dict[str, object]],
    spatial_sums: dict[str, dict[str, np.ndarray]],
    spatial_counts: dict[str, int],
    output_root: Path,
) -> list[Path]:
    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(13, 6))
    labels = [
        f"{combination_axis_label(item['combination'])}\n{item['modality']}" for item in deeplift_summary
    ]
    values = [item["mean_attribution_fraction"]["mean"] for item in deeplift_summary]
    errors = [item["mean_attribution_fraction"]["std"] for item in deeplift_summary]
    colors = [modality_color(item["modality"]) for item in deeplift_summary]
    bars = axis.bar(
        range(len(values)), values, yerr=errors, color=colors, capsize=4,
        edgecolor=BAR_EDGE_COLOR, linewidth=1.35,
        error_kw={"ecolor": BAR_EDGE_COLOR, "elinewidth": 1.1, "capthick": 1.1},
    )
    style_bars(bars)
    axis.set_xticks(range(len(values)), labels, rotation=32, ha="right")
    axis.set_ylabel("Mean attribution fraction")
    axis.set_title("step 10 N DeepLIFT modality importance", loc="left", pad=14, fontweight="bold")
    upper = max(1e-6, max(value + error for value, error in zip(values, errors)) * 1.2)
    axis.set_ylim(0, upper)
    style_axis(axis)
    offset = max(upper * 0.018, 0.008)
    for bar, value, error in zip(bars, values, errors):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    figure.tight_layout()
    path = figures_root / "deeplift_modality_importance.png"
    figure.savefig(path, dpi=160, facecolor="#F7F9FC", bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    for metric, title, filename in (
        ("delta_accuracy", "Accuracy change after modality perturbation", "perturbation_accuracy.png"),
        ("delta_macro_f1", "Macro-F1 change after modality perturbation", "perturbation_macro_f1.png"),
    ):
        figure, axis = plt.subplots(figsize=(13, 6))
        labels = [
            f"{combination_axis_label(item['combination'])}\n{item['modality']}"
            for item in perturbation_summary
        ]
        values = [item[metric]["mean"] for item in perturbation_summary]
        errors = [item[metric]["std"] for item in perturbation_summary]
        colors = [modality_color(item["modality"]) for item in perturbation_summary]
        bars = axis.bar(
            range(len(values)), values, yerr=errors, color=colors, capsize=4,
            edgecolor=BAR_EDGE_COLOR, linewidth=1.35,
            error_kw={"ecolor": BAR_EDGE_COLOR, "elinewidth": 1.1, "capthick": 1.1},
        )
        style_bars(bars)
        axis.axhline(0, color=BAR_EDGE_COLOR, linewidth=0.9)
        axis.set_xticks(range(len(values)), labels, rotation=32, ha="right")
        axis.set_ylabel(metric.replace("delta_", "Delta ").replace("_", " ").title())
        axis.set_title(f"step 10 N {title}", loc="left", pad=14, fontweight="bold")
        limit = max(max(abs(value) + error for value, error in zip(values, errors)) * 1.25, 0.05)
        axis.set_ylim(-limit, limit)
        style_axis(axis)
        for bar, value, error in zip(bars, values, errors):
            offset = max(limit * 0.018, 0.008)
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + offset if value >= 0 else value - error - offset,
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8.5,
                fontweight="bold",
            )
        figure.tight_layout()
        path = figures_root / filename
        figure.savefig(path, dpi=160, facecolor="#F7F9FC", bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    for name, modality_maps in spatial_sums.items():
        modalities = list(modality_maps)
        figure, axes = plt.subplots(1, len(modalities), figsize=(4 * len(modalities), 4))
        axes_array = np.atleast_1d(axes)
        for axis, modality in zip(axes_array, modalities):
            data = modality_maps[modality] / max(spatial_counts[name], 1)
            data = data / max(float(data.max()), 1e-12)
            image = axis.imshow(data, cmap="magma")
            axis.set_title(modality)
            axis.axis("off")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(f"step 10 N mean absolute DeepLIFT map: {name}")
        figure.tight_layout()
        path = figures_root / f"deeplift_spatial_{combination_slug(tuple(modalities))}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def figure_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def html_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body: list[str] = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_report(
    output_root: Path,
    config: dict[str, object],
    integrity: dict[str, object],
    model_rows: list[dict[str, object]],
    deeplift_summary: list[dict[str, object]],
    perturbation_summary: list[dict[str, object]],
    figure_paths: list[Path],
) -> None:
    deep_fraction_rows = [
        {
            "combination": row["combination"],
            "modality": row["modality"],
            "models": row["model_count"],
            "attribution_fraction_mean": row["mean_attribution_fraction"]["mean"],
            "attribution_fraction_std": row["mean_attribution_fraction"]["std"],
        }
        for row in deeplift_summary
    ]
    deep_abs_rows = [
        {
            "combination": row["combination"],
            "modality": row["modality"],
            "models": row["model_count"],
            "mean_abs_attribution": row["mean_abs_attribution"]["mean"],
            "mean_abs_attribution_std": row["mean_abs_attribution"]["std"],
        }
        for row in deeplift_summary
    ]
    perturb_rows = [
        {
            "combination": row["combination"],
            "modality": row["modality"],
            "models": row["model_count"],
            "delta_accuracy_mean": row["delta_accuracy"]["mean"],
            "delta_accuracy_std": row["delta_accuracy"]["std"],
            "delta_macro_f1_mean": row["delta_macro_f1"]["mean"],
            "delta_macro_f1_std": row["delta_macro_f1"]["std"],
        }
        for row in perturbation_summary
    ]
    perturb_summary_rows = [
        {
            "combination": row["combination"],
            "condition": f"ablate_{row['modality']}",
            "accuracy_delta_mean": row["delta_accuracy"]["mean"],
            "accuracy_delta_std": row["delta_accuracy"]["std"],
            "macro_f1_delta_mean": row["delta_macro_f1"]["mean"],
            "macro_f1_delta_std": row["delta_macro_f1"]["std"],
        }
        for row in perturbation_summary
    ]
    figure_map = {path.name: path for path in figure_paths}

    def figure_html(filename: str) -> str:
        path = figure_map[filename]
        return (
            f"<figure><img src=\"{figure_data_uri(path)}\">"
            f"<figcaption>{html.escape(filename)}</figcaption></figure>"
        )

    model_table = html_table(
        [
            {
                "combination": row["combination"],
                "fold": row["fold"],
                "training_seed": row["training_seed"],
                "evaluation_count": row["evaluation_count"],
                "baseline_accuracy": row["baseline_accuracy"],
                "baseline_macro_f1": row["baseline_macro_f1"],
            }
            for row in model_rows
        ],
        [
            "combination",
            "fold",
            "training_seed",
            "evaluation_count",
            "baseline_accuracy",
            "baseline_macro_f1",
        ],
    )
    integrity_record = {
        key: value for key, value in integrity.items() if key != "step3_artifact_check"
    }
    record = {"config": config, "integrity": integrity_record}
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>step_10_N_report</title>
<style>{REPORT_CSS}</style></head><body>
<h1>Step 10_N 五分类 Soft Label DeepLIFT + 通道扰动</h1>
<p>本次分析针对患者不隔离路线（Patient Isolation = False），包含 <code>M + MR</code>、<code>M + MB + MR</code> 和全通道三个经确认的通道组合；每种组合完整分析 3 个 Training Seed × 5 个 Fold，共 15 个 step 09 N Epoch-50 模型，未按评估性能选择代表模型。</p>
<p>DeepLIFT target 为模型预测类别，baseline 为归一化张量空间中的零值。通道扰动将一个模态对应的 RGB 三通道置零，使用该模型自身 held-out Fold 进行评估。报告目录不包含权重、原始医学图片或患者身份信息。</p>
<section><h2>实际配置与完整性记录</h2><pre>{html.escape(json.dumps(record, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>通道总体重要性</h2>
<p>归因分数为绝对 DeepLIFT attribution fraction 的均值 ± 标准差，按每种组合的 15 个模型汇总。表格数据置于图表上方。</p>
{html_table(deep_fraction_rows, ["combination", "modality", "models", "attribution_fraction_mean", "attribution_fraction_std"])}
{figure_html("deeplift_modality_importance.png")}
</section>
<section><h2>通道扰动的性能变化</h2>
<p>Accuracy 和 Macro-F1 的变化均为“扰动后 − baseline”；负值表示性能下降。表格数据置于图表上方。</p>
{html_table(perturb_rows, ["combination", "modality", "models", "delta_accuracy_mean", "delta_accuracy_std", "delta_macro_f1_mean", "delta_macro_f1_std"])}
<div class="gallery">{figure_html("perturbation_accuracy.png")}{figure_html("perturbation_macro_f1.png")}</div>
</section>
<section><h2>DeepLIFT 汇总</h2>
{html_table(deep_abs_rows, ["combination", "modality", "models", "mean_abs_attribution", "mean_abs_attribution_std"])}
</section>
<section><h2>扰动汇总</h2>
{html_table(perturb_summary_rows, ["combination", "condition", "accuracy_delta_mean", "accuracy_delta_std", "macro_f1_delta_mean", "macro_f1_delta_std"])}
</section>
<section><h2>DeepLIFT 空间图</h2>
<div class="gallery">{figure_html("deeplift_spatial_M_MR.png")}{figure_html("deeplift_spatial_M_MB_MR.png")}{figure_html("deeplift_spatial_M_MB_MP_MR_MUV.png")}</div>
</section>
<section><h2>模型级 baseline 记录</h2>
{model_table}
</section>
</body></html>"""
    (output_root / "report_step_10_N.html").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.analysis_batch_size <= 0:
        raise ValueError("analysis-batch-size must be positive")
    output_root = Path(args.output_root).expanduser().resolve()
    validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    step3_root = Path(args.step3_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    reference_data_root = (
        Path(args.reference_data_root).expanduser().resolve()
        if args.reference_data_root
        else None
    )

    if args.mode == "train":
        stale_markers = [
            output_root / "metrics_step_10_N.json",
            output_root / "report_step_10_N.html",
        ]
        if any(path.exists() for path in stale_markers):
            raise RuntimeError("step 10 N contains prior formal results; refusing to mix runs")

    folded, image_lookup, integrity = run_integrity_checks(
        args.manifest, data_root, reference_data_root, args.reference_split
    )
    integrity["step3_artifact_check"] = validate_step3_artifacts(step3_root, folded)
    write_json(output_root / "integrity_report.json", integrity)
    if not integrity["passed"]:
        raise RuntimeError("pre-analysis integrity checks failed; see integrity_report.json")

    device = resolve_device(args.device)
    config = {
        "experiment_name": STEP_NAME,
        "benchmark_type": "DeepLIFT_and_channel_perturbation",
        "task": "five_class",
        "label_strategy": "Soft Label epsilon=0.10 adjacent",
        "patient_isolation": False,
        "source_step": "step_09_N",
        "source_step_root": str(step3_root),
        "model_selection": "all_15_models_per_selected_combination",
        "selected_combinations": [list(modalities) for modalities in COMBINATIONS],
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "fold_seed": FOLD_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation": "each step 09 N model uses its own held-out Fold",
        "deeplift": {
            "method": "Captum DeepLift",
            "baseline": "zero_normalized_tensor",
            "target_rule": TARGET_RULE,
            "relu_preparation": "torch.fx distinct non-inplace ReLU call modules",
        },
        "perturbation": {
            "unit": "three channels per modality",
            "replacement": "zero_normalized_tensor",
            "augmentation": "disabled on evaluation Fold",
        },
        "output_root": str(output_root),
        "save_checkpoints": False,
        "num_workers": args.num_workers,
        "analysis_batch_size": args.analysis_batch_size,
        "device": str(device),
    }
    write_json(output_root / "config_step_10_N.json", config)
    folded[["sample_id", "fold"]].to_csv(output_root / "folds_step_10_N.csv", index=False)

    run_smoke(
        folded,
        image_lookup,
        data_root,
        step3_root,
        device,
        args.num_workers,
        args.analysis_batch_size,
        output_root,
    )
    if args.mode == "smoke":
        print("SMOKE_TEST_OK", flush=True)
        return

    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    formal_log = output_root / "logs" / "formal_run.log"
    model_rows: list[dict[str, object]] = []
    perturbation_rows: list[dict[str, object]] = []
    spatial_sums: dict[str, dict[str, np.ndarray]] = {}
    spatial_counts: dict[str, int] = {}
    with formal_log.open("w", encoding="utf-8") as log:
        for modalities in COMBINATIONS:
            name = combination_name(modalities)
            for seed in TRAINING_SEEDS:
                for fold, _train_raw, eval_raw in iter_folds(
                    folded, patient_isolation=False
                ):
                    eval_frame = filter_available_samples(eval_raw, image_lookup, modalities)
                    checkpoint_path = expected_step3_checkpoint(
                        step3_root, modalities, fold, seed
                    )
                    message = f"START combination={name} seed={seed} fold={fold}"
                    print(message, flush=True)
                    log.write(message + "\n")
                    log.flush()
                    try:
                        model_row, rows = analyze_model(
                            modalities,
                            fold,
                            seed,
                            eval_frame,
                            image_lookup,
                            data_root,
                            checkpoint_path,
                            device,
                            args.num_workers,
                            args.analysis_batch_size,
                            spatial_sums,
                            step3_root,
                        )
                    except Exception as exc:
                        error = f"ERROR combination={name} seed={seed} fold={fold}: {exc}"
                        log.write(error + "\n")
                        log.flush()
                        raise
                    model_rows.append(model_row)
                    perturbation_rows.extend(rows)
                    spatial_counts[name] = spatial_counts.get(name, 0) + len(eval_frame)
                    write_json(
                        output_root / "metrics_step_10_N.json",
                        {
                            "integrity": integrity,
                            "config": config,
                            "model_results": model_rows,
                            "perturbation_results": perturbation_rows,
                        },
                    )
                    save_csv(output_root / "model_results_step_10_N.csv", model_rows)
                    save_csv(
                        output_root / "channel_perturbation_step_10_N.csv",
                        perturbation_rows,
                    )
                    message = (
                        f"DONE combination={name} seed={seed} fold={fold} "
                        f"accuracy={model_row['baseline_accuracy']:.6f} "
                        f"macro_f1={model_row['baseline_macro_f1']:.6f}"
                    )
                    print(message, flush=True)
                    log.write(message + "\n")
                    log.flush()

    deeplift_summary, perturbation_summary = aggregate_results(
        model_rows, perturbation_rows
    )
    write_json(
        output_root / "metrics_step_10_N.json",
        {
            "integrity": integrity,
            "config": config,
            "model_results": model_rows,
            "deeplift_summary": deeplift_summary,
            "perturbation_results": perturbation_rows,
            "perturbation_summary": perturbation_summary,
        },
    )
    save_csv(output_root / "model_results_step_10_N.csv", model_rows)
    save_csv(output_root / "deeplift_summary_step_10_N.csv", deeplift_summary)
    save_csv(output_root / "channel_perturbation_step_10_N.csv", perturbation_rows)
    write_json(output_root / "deeplift_summary_step_10_N.json", deeplift_summary)
    write_json(output_root / "channel_perturbation_step_10_N.json", perturbation_summary)
    figure_paths = save_figures(
        deeplift_summary,
        perturbation_summary,
        spatial_sums,
        spatial_counts,
        output_root,
    )
    write_report(
        output_root,
        config,
        integrity,
        model_rows,
        deeplift_summary,
        perturbation_summary,
        figure_paths,
    )
    print("STEP_10_N_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
