"""Step 05_I - five-class single-channel benchmark under patient isolation.

Hard labels: original 0-4 severity grades used directly (num_classes=5,
dynamic from the manifest; original labels never modified). All five
channels (M, MB, MP, MR, MUV) are trained here - the earlier steps were all
binary, so no run can be reused. White (M) becomes the baseline for the
later five-class multi-channel steps.

Emphasis per TRAINING_PLAN Step 5: adjacent-grade confusion - every run
additionally reports the adjacent-error rate (|true - pred| == 1) and the
far-error rate (|true - pred| >= 2), alongside Accuracy / Macro-F1 / MAE /
per-class precision-recall-F1 / 5x5 confusion matrices.

Run modes:
  python step_05_I.py --manifest ... --data-root ... [--cache-dir ...]
      [--smoke] [--report-only] [--fold N] [--channel M]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    DEFAULT_AUGMENTATION,
    EXCLUDED_IDS,
    INPUT_SIZE,
    SEED as FOLD_SEED,
    SkinDataset,
    discover_images,
    load_manifest,
    make_five_folds,
)
from model import build_classifier, count_parameters  # noqa: E402

STEP_DIR = Path(__file__).resolve().parent
STEP_NAME = STEP_DIR.name
TASK = "5class"
TASK_DISPLAY = "五分类（0-4 严重程度，Hard Label）"
NUM_CLASSES = 5
CHANNELS = ("M", "MB", "MP", "MR", "MUV")
TRAINING_SEEDS = (42, 3407, 2026)
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_NAME = "resnet50"
PATIENT_ISOLATION = True
NUM_WORKERS = 8
CLASS_NAMES = tuple(f"Class {i}" for i in range(5))

METRIC_KEYS = ("accuracy", "macro_f1", "mae")
CHANNEL_COLORS = {
    "M": "#4C72B0", "MB": "#DD8452", "MP": "#55A868",
    "MR": "#C44E52", "MUV": "#8172B3",
}


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def guard_path(path: Path) -> Path:
    """Refuse writes outside this step directory (PROTOCOL section 16)."""

    resolved = path.resolve()
    step_root = STEP_DIR.resolve()
    if step_root != resolved and step_root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside {step_root}: {resolved}")
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CachedSkinDataset(SkinDataset):
    """SkinDataset with optional cache (disabled: data root already holds
    the exact ResizePad(384) PNGs and ResizePad is idempotent)."""

    def __init__(self, *args, cache_dir: str | Path | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None

    def _load_images(self, capture_id: int):
        from PIL import Image, ImageOps

        images = []
        for modality in self.modalities:
            source = self.image_lookup[modality][capture_id]
            if self.cache_dir is None:
                with Image.open(source) as raw:
                    images.append(ImageOps.exif_transpose(raw).convert("RGB"))
                continue
            cached = self.cache_dir / f"{modality}_{source.stem}_{INPUT_SIZE}.png"
            if not cached.exists():
                with Image.open(source) as raw:
                    resized = self.resize_pad(
                        ImageOps.exif_transpose(raw).convert("RGB")
                    )
                tmp = cached.with_suffix(f".tmp{os.getpid()}.png")
                resized.save(tmp, format="PNG", compress_level=1)
                os.replace(tmp, cached)
            with Image.open(cached) as cached_image:
                images.append(cached_image.copy())
        return images


# ---------------------------------------------------------------------------
# Data integrity checks (PROTOCOL section 17)
# ---------------------------------------------------------------------------


def run_integrity_checks(
    manifest_path: Path, data_root: Path, folded: pd.DataFrame
) -> dict:
    checks: dict[str, object] = {}
    frame = load_manifest(manifest_path)
    lookup = discover_images(data_root)

    for excluded in sorted(EXCLUDED_IDS):
        channels_holding = sorted(m for m, ids in lookup.items() if excluded in ids)
        checks[f"excluded_{excluded}_absent"] = {
            "in_manifest": not bool(set(frame["capture_id"]) & {excluded}),
            "channels_still_indexing_it": channels_holding,
            "passed": (not bool(set(frame["capture_id"]) & {excluded}))
            and not channels_holding,
        }

    overlap = {}
    for fold in range(5):
        evaluation = folded.loc[folded["fold"] == fold]
        training = folded.loc[folded["fold"] != fold]
        overlap[f"fold_{fold + 1}"] = {
            "sample_overlap": sorted(set(evaluation["sample_id"]) & set(training["sample_id"])),
            "patient_overlap": sorted(
                set(evaluation["patient_id"]) & set(training["patient_id"])
            ),
        }
    checks["fold_overlap"] = overlap
    checks["fold_overlap_passed"] = {
        key: (not value["sample_overlap"]) and (not value["patient_overlap"])
        for key, value in overlap.items()
    }

    per_channel = {}
    for channel in CHANNELS:
        missing = sorted(set(frame["capture_id"]) - set(lookup[channel]))
        per_channel[channel] = {
            "manifest_samples": int(len(frame)),
            "images_available": int(len(set(frame["capture_id"]) & set(lookup[channel]))),
            "manifest_ids_without_image": missing[:10],
            "passed": not missing,
        }
    checks["channel_available"] = per_channel
    checks["task"] = {
        "task": TASK,
        "num_classes": NUM_CLASSES,
        "mapping": "hard label: original 0-4 used directly",
        "passed": NUM_CLASSES == 5,
    }

    per_fold = {}
    for fold in range(5):
        evaluation = folded.loc[folded["fold"] == fold]
        training = folded.loc[folded["fold"] != fold]
        per_fold[f"fold_{fold + 1}"] = {
            "train_n": int(len(training)),
            "eval_n": int(len(evaluation)),
            "train_label_hist": {
                str(k): int(v) for k, v in sorted(training["label"].value_counts().items())
            },
            "eval_label_hist": {
                str(k): int(v) for k, v in sorted(evaluation["label"].value_counts().items())
            },
        }
    checks["fold_composition"] = per_fold

    passed = all(
        entry["passed"] if isinstance(entry, dict) and "passed" in entry else True
        for entry in checks.values()
    ) and all(checks["fold_overlap_passed"].values())
    return {"passed": passed, "checks": checks}


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def build_loaders(
    channel, train_frame, eval_frame, data_root, cache_dir, seed
):
    train_dataset = CachedSkinDataset(
        train_frame, data_root, modalities=(channel,), training=True, cache_dir=cache_dir
    )
    eval_dataset = CachedSkinDataset(
        eval_frame, data_root, modalities=(channel,), training=False, cache_dir=cache_dir
    )

    def worker_init(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        generator=generator, worker_init_fn=worker_init, pin_memory=True,
        drop_last=False, persistent_workers=False,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, eval_loader


def train_one_run(
    channel, fold_number, training_seed, train_frame, eval_frame,
    data_root, cache_dir, run_dir, epochs=EPOCHS, log_every=10,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(training_seed)
    train_loader, eval_loader = build_loaders(
        channel, train_frame, eval_frame, data_root, cache_dir, training_seed
    )

    model = build_classifier(
        MODEL_NAME, num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
        pretrained=True, dropout=0.3, input_channels=3,
    ).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = guard_path(run_dir / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = guard_path(run_dir / "train.log")
    eval_log_path = guard_path(run_dir / "eval.log")

    started = time.time()
    with open(train_log_path, "w", encoding="utf-8") as train_log:
        train_log.write(
            f"channel={channel} fold={fold_number} training_seed={training_seed} "
            f"epochs={epochs} batch_size={BATCH_SIZE} optimizer=AdamW lr={LR} "
            f"weight_decay={WEIGHT_DECAY} loss=CrossEntropyLoss scheduler=None "
            f"early_stopping=False task=5class\n"
        )
        train_log.flush()
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_started = time.time()
            running_loss, running_correct, running_total = 0.0, 0, 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                running_correct += (logits.argmax(1) == targets).sum().item()
                running_total += images.size(0)
            train_log.write(
                f"epoch={epoch:03d} loss={running_loss / max(1, running_total):.4f} "
                f"acc={running_correct / max(1, running_total):.4f} "
                f"sec={time.time() - epoch_started:.1f}\n"
            )
            train_log.flush()
            if epoch % log_every == 0 or epoch == epochs:
                log(
                    f"{channel} fold {fold_number} seed {training_seed} "
                    f"epoch {epoch}/{epochs} "
                    f"loss={running_loss / max(1, running_total):.4f} "
                    f"acc={running_correct / max(1, running_total):.4f}"
                )

    last_path = checkpoint_dir / "last.pth"
    torch.save(
        {
            "state_dict": model.state_dict(), "channel": channel,
            "fold": fold_number, "training_seed": training_seed,
            "epoch": epochs, "step": STEP_NAME, "task": TASK,
            "input_size": INPUT_SIZE,
        },
        last_path,
    )

    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad(), open(eval_log_path, "w", encoding="utf-8") as eval_log:
        eval_log.write(
            f"channel={channel} fold={fold_number} training_seed={training_seed} "
            f"checkpoint=last.pth (epoch {epochs}) task=5class\n"
        )
        for batch in eval_loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_targets.extend(int(t) for t in batch["target"])
            all_preds.extend(int(p) for p in preds.cpu().tolist())

        targets_array = np.array(all_targets)
        preds_array = np.array(all_preds)
        accuracy = accuracy_score(targets_array, preds_array)
        macro_f1 = f1_score(targets_array, preds_array, average="macro", zero_division=0)
        mae = float(np.mean(np.abs(targets_array - preds_array)))
        precision, recall, f1s, support = precision_recall_fscore_support(
            targets_array, preds_array, labels=list(range(5)), zero_division=0
        )
        cm = confusion_matrix(targets_array, preds_array, labels=list(range(5)))
        n = cm.sum()
        adjacent_rate = float(
            sum(cm[i, j] for i in range(5) for j in range(5) if abs(i - j) == 1) / n
        )
        far_rate = float(
            sum(cm[i, j] for i in range(5) for j in range(5) if abs(i - j) >= 2) / n
        )
        eval_log.write(
            f"accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} mae={mae:.4f}\n"
            f"adjacent_error_rate={adjacent_rate:.4f} far_error_rate={far_rate:.4f}\n"
            f"confusion_matrix (rows=true 0-4, cols=pred)={cm.tolist()}\n"
            f"train_n={len(train_frame)} eval_n={len(eval_frame)}\n"
        )

    return {
        "channel": channel, "fold": fold_number, "training_seed": training_seed,
        "accuracy": float(accuracy), "macro_f1": float(macro_f1), "mae": mae,
        "per_class": {
            "precision": [float(v) for v in precision],
            "recall": [float(v) for v in recall],
            "f1": [float(v) for v in f1s],
            "support": [int(v) for v in support],
        },
        "confusion_matrix": cm.tolist(),
        "adjacent_error_rate": adjacent_rate,
        "far_error_rate": far_rate,
        "train_n": int(len(train_frame)), "eval_n": int(len(eval_frame)),
        "majority_acc": float(max(np.bincount(targets_array, minlength=5)) / len(targets_array)),
        "epochs": epochs, "param_count": int(parameter_count),
        "checkpoint": str(last_path.relative_to(STEP_DIR)),
        "train_minutes": round((time.time() - started) / 60, 2),
    }


# ---------------------------------------------------------------------------
# Aggregation, figures, report
# ---------------------------------------------------------------------------


def aggregate_channel(runs: list[dict]) -> dict:
    summary = {"per_seed": {}, "final": {}}

    def stats(values):
        array = np.array(values, dtype=float)
        return {"mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0}

    for seed in TRAINING_SEEDS:
        seed_runs = [r for r in runs if r["training_seed"] == seed]
        seed_summary = {key: stats([r[key] for r in seed_runs]) for key in
                        (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")}
        seed_summary["per_class"] = {
            metric: [stats([r["per_class"][metric][cls] for r in seed_runs])
                     for cls in range(5)]
            for metric in ("precision", "recall", "f1")
        }
        summary["per_seed"][str(seed)] = {"fold_runs": seed_runs, "summary": seed_summary}

    seed_means = {
        key: [summary["per_seed"][str(s)]["summary"][key]["mean"] for s in TRAINING_SEEDS]
        for key in (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")
    }
    summary["final"] = {key: stats(v) for key, v in seed_means.items()}
    summary["final"]["per_class"] = {
        metric: [
            stats([summary["per_seed"][str(s)]["summary"]["per_class"][metric][cls]["mean"]
                   for s in TRAINING_SEEDS])
            for cls in range(5)
        ]
        for metric in ("precision", "recall", "f1")
    }
    return summary


def make_figures(summaries: dict, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    majority_acc = float(np.mean(
        [r["majority_acc"] for r in summaries["M"]["per_seed"]["42"]["fold_runs"]]
    ))

    series = [("seed 42", "42"), ("seed 3407", "3407"), ("seed 2026", "2026")]
    series_colors = ["#8ea6c4", "#4C72B0", "#2b3f66"]
    x = np.arange(len(CHANNELS))
    for key, label, percent in (
        ("accuracy", "Accuracy", True),
        ("macro_f1", "Macro-F1", True),
        ("mae", "MAE (severity grades)", False),
        ("adjacent_error_rate", "Adjacent-class error rate", True),
        ("far_error_rate", "Far-class error rate (|Δgrade|≥2)", True),
    ):
        fig, ax = plt.subplots(figsize=(10, 4.2))
        width = 0.19
        groups = []
        for offset, (name, seed_key) in enumerate(series):
            means = [summaries[c]["per_seed"][seed_key]["summary"][key]["mean"] for c in CHANNELS]
            stds = [summaries[c]["per_seed"][seed_key]["summary"][key]["std"] for c in CHANNELS]
            bars = ax.bar(x + (offset - 1.5) * width, means, width, yerr=stds,
                          label=name, color=series_colors[offset], capsize=3)
            groups.append((bars, means, stds))
        final_means = [summaries[c]["final"][key]["mean"] for c in CHANNELS]
        final_stds = [summaries[c]["final"][key]["std"] for c in CHANNELS]
        bars = ax.bar(x + 1.5 * width, final_means, width, yerr=final_stds,
                      label="overall (3 seeds)", color="#333333", capsize=3)
        groups.append((bars, final_means, final_stds))
        top = max(m + s for _, means, stds in groups for m, s in zip(means, stds))
        ax.set_ylim(0, max(1.02, top * 1.25))
        if percent:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        for bar, mean, std in zip(bars, final_means, final_stds):
            text = f"{mean * 100:.1f}%" if percent else f"{mean:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + top * 0.02,
                    text, ha="center", va="bottom", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
        if key == "accuracy":
            ax.axhline(majority_acc, color="red", linestyle="--", linewidth=1.2)
            ax.text(len(CHANNELS) - 0.42, majority_acc + 0.012,
                    f"majority baseline {majority_acc * 100:.1f}%", color="darkred",
                    fontsize=8, ha="right",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(CHANNELS)
        ax.set_ylabel(label)
        ax.set_title(
            f"Step 05_I 5-class hard-label single-channel benchmark | {label} "
            "(bar = mean over 5 folds; error bar = std)"
        )
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures_dir / f"{key}_by_channel.{suffix}"
            fig.savefig(path, dpi=160)
            if suffix == "png":
                figure_paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        width = 0.15
        cx = np.arange(5)
        for index, channel in enumerate(CHANNELS):
            means = [summaries[channel]["final"]["per_class"][metric][cls]["mean"] for cls in range(5)]
            stds = [summaries[channel]["final"]["per_class"][metric][cls]["std"] for cls in range(5)]
            ax.bar(cx + (index - 2) * width, means, width, yerr=stds, label=channel,
                   color=CHANNEL_COLORS[channel], capsize=2)
        ax.set_xticks(cx)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.capitalize())
        ax.set_ylabel(metric.capitalize())
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Step 05_I 5-class | per-class metrics by channel (final 3-seed mean ± std)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"per_class_by_channel.{suffix}"
        fig.savefig(path, dpi=160)
        if suffix == "png":
            figure_paths.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(4.0 * len(CHANNELS), 3.8))
    for ax, channel in zip(axes, CHANNELS):
        runs = [r for s in TRAINING_SEEDS for r in summaries[channel]["per_seed"][str(s)]["fold_runs"]]
        summed = np.sum([np.array(r["confusion_matrix"]) for r in runs], axis=0)
        im = ax.imshow(summed, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(int(summed[i, j])), ha="center", va="center",
                        fontsize=7,
                        color="white" if summed[i, j] > summed.max() / 2 else "black")
        ax.set_xticks(range(5), labels=CLASS_NAMES, fontsize=7)
        ax.set_yticks(range(5), labels=CLASS_NAMES, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{channel} (sum of 15 runs)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Step 05_I 5-class | confusion matrices by channel (5 folds x 3 seeds)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"confusion_matrices_by_channel.{suffix}"
        fig.savefig(path, dpi=160)
        if suffix == "png":
            figure_paths.append(path)
    plt.close(fig)

    return figure_paths


def image_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_html_report(config, integrity, summaries, figure_paths, notes, output_path):
    sections = []
    config_rows = "".join(
        f"<tr><td>{k}</td><td>{json.dumps(v, ensure_ascii=False, default=str)}</td></tr>"
        for k, v in config.items()
    )
    sections.append(
        f"<h2>实验配置</h2><table class='cfg'><tr><th>项</th><th>值</th></tr>{config_rows}</table>"
    )

    check_rows = []
    for name, outcome in integrity["checks"].items():
        if isinstance(outcome, dict) and "passed" in outcome:
            flag = "PASS" if outcome["passed"] else "FAIL"
            check_rows.append(f"<tr><td>{name}</td><td>{flag}</td></tr>")
        elif name == "fold_overlap":
            for fold, res in outcome.items():
                flag = "PASS" if not res["sample_overlap"] and not res["patient_overlap"] else "FAIL"
                check_rows.append(f"<tr><td>{name} {fold}</td><td>{flag}</td></tr>")
        elif name == "channel_available":
            for channel, res in outcome.items():
                flag = "PASS" if res["passed"] else "FAIL"
                check_rows.append(f"<tr><td>channel_available {channel}</td><td>{flag}</td></tr>")
    sections.append(
        f"<h2>数据完整性检查（PROTOCOL §17）</h2>"
        f"<p>总体：<b>{'PASS' if integrity['passed'] else 'FAIL'}</b></p>"
        f"<table class='cfg'><tr><th>检查项</th><th>结果</th></tr>{''.join(check_rows)}</table>"
    )

    for channel in CHANNELS:
        rows = []
        for seed in TRAINING_SEEDS:
            for run in summaries[channel]["per_seed"][str(seed)]["fold_runs"]:
                rows.append(
                    "<tr>"
                    f"<td>{run['fold']}</td><td>{seed}</td>"
                    f"<td>{run['train_n']}</td><td>{run['eval_n']}</td>"
                    f"<td>{run['accuracy']:.4f}</td><td>{run['macro_f1']:.4f}</td>"
                    f"<td>{run['mae']:.4f}</td>"
                    f"<td>{run['adjacent_error_rate']:.4f}</td><td>{run['far_error_rate']:.4f}</td>"
                    f"<td>{run['train_minutes']:.1f}</td></tr>"
                )
        sections.append(
            f"<h3>通道 {channel}（15 条 fold 记录）</h3>"
            "<table><tr><th>Fold</th><th>Training Seed</th><th>Train N</th><th>Eval N</th>"
            "<th>Accuracy</th><th>Macro-F1</th><th>MAE</th><th>相邻误判率</th>"
            "<th>远距误判率</th><th>Minutes</th></tr>"
            f"{''.join(rows)}</table>"
        )

    summary_rows = []
    for channel in CHANNELS:
        final = summaries[channel]["final"]
        summary_rows.append(
            f"<tr><td>{channel}</td>"
            + "".join(
                f"<td>{final[key]['mean']:.4f} ± {final[key]['std']:.4f}</td>"
                for key in (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")
            )
            + "</tr>"
        )
    sections.append(
        "<h2>各通道最终汇总（每格 = 三种子最终 Mean ± Std；种子内先对五折 Mean ± Std）</h2>"
        "<table><tr><th>Channel</th><th>Accuracy</th><th>Macro-F1</th><th>MAE</th>"
        "<th>相邻误判率</th><th>远距误判率</th></tr>"
        f"{''.join(summary_rows)}</table>"
    )

    images = "".join(
        f"<h3>{path.stem}</h3><img src='{image_data_uri(path)}' alt='{path.stem}'>"
        for path in figure_paths
    )
    sections.append(f"<h2>图表</h2>{images}")
    notes_html = "".join(f"<li>{note}</li>" for note in notes)
    sections.append(f"<h2>说明与记录</h2><ul>{notes_html}</ul>")

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>{STEP_NAME} 报告</title>
<style>
body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ border-bottom: 2px solid #4C72B0; padding-bottom: 4px; }}
table {{ border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
th, td {{ border: 1px solid #bbb; padding: 4px 10px; text-align: center; }}
th {{ background: #eef2f8; }}
table.cfg td:last-child {{ text-align: left; }}
img {{ max-width: 1100px; margin: 8px 0; border: 1px solid #ddd; }}
</style></head>
<body>
<h1>{STEP_NAME}：{TASK_DISPLAY} 单通道实验（Patient Isolation = True）</h1>
<p>生成时间：{datetime.now(timezone.utc).isoformat()}</p>
{''.join(sections)}
</body></html>"""
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/root/autodl-tmp/data_all_384/manifest.csv")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data_all_384")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--channel", choices=CHANNELS)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    data_root = Path(args.data_root).expanduser()
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None

    logs_dir = guard_path(STEP_DIR / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = guard_path(STEP_DIR / "figures")
    metrics_json_path = guard_path(STEP_DIR / f"metrics_{STEP_NAME}.json")
    metrics_csv_path = guard_path(STEP_DIR / f"metrics_{STEP_NAME}.csv")
    config_path = guard_path(STEP_DIR / f"config_{STEP_NAME}.json")
    report_path = guard_path(STEP_DIR / f"report_{STEP_NAME}.html")
    status_path = logs_dir / "status.json"

    if args.report_only:
        payload = json.loads(metrics_json_path.read_text(encoding="utf-8"))
        figure_paths = make_figures(payload["summaries"], figures_dir)
        build_html_report(payload["config"], payload["integrity"],
                          payload["summaries"], figure_paths,
                          payload.get("notes", []), report_path)
        log(f"report rebuilt at {report_path}")
        return

    log(
        f"{STEP_NAME} start | task=5class num_classes=5 channels={CHANNELS} "
        f"resolution={INPUT_SIZE} isolation={PATIENT_ISOLATION} "
        f"fold_seed={FOLD_SEED} training_seeds={TRAINING_SEEDS}"
    )
    folded = make_five_folds(manifest_path, patient_isolation=PATIENT_ISOLATION, seed=FOLD_SEED)
    integrity = run_integrity_checks(manifest_path, data_root, folded)
    (logs_dir / "integrity_checks.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"integrity checks passed: {integrity['passed']}")
    if not integrity["passed"]:
        raise SystemExit(2)

    config = {
        "experiment_name": STEP_NAME,
        "benchmark": "five-class hard-label single-channel benchmark",
        "task": TASK,
        "task_display": TASK_DISPLAY,
        "num_classes": NUM_CLASSES,
        "label_strategy": "hard label: original 0-4 severity used directly",
        "patient_isolation": PATIENT_ISOLATION,
        "fold_seed": FOLD_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "input_resolution": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "channels": list(CHANNELS),
        "model": MODEL_NAME,
        "pretrained": "ImageNet",
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "scheduler": None,
        "early_stopping": False,
        "loss": "CrossEntropyLoss",
        "augmentation": DEFAULT_AUGMENTATION,
        "augmentation_policy": (
            "identical policy for every channel: flip 0.5, rotation U(-7,7) deg, "
            "brightness U(0.90,1.10), contrast U(0.90,1.10); no random augmentation "
            "on evaluation folds"
        ),
        "num_workers": NUM_WORKERS,
        "precision": "fp32",
        "metrics": list(METRIC_KEYS) + ["adjacent_error_rate", "far_error_rate"],
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "fold_counts": {
            f"fold_{fold + 1}": int((folded["fold"] == fold).sum()) for fold in range(5)
        },
        "total_samples": int(len(folded)),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.smoke:
        smoke_dir = guard_path(logs_dir / "smoke" / "M_fold_01_seed_42")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        train_frame = folded.loc[folded["fold"] != 0].reset_index(drop=True).head(96)
        eval_frame = folded.loc[folded["fold"] == 0].reset_index(drop=True).head(64)
        log("SMOKE TEST: channel M, fold 1, seed 42, 2 epochs, 5-class")
        result = train_one_run(
            "M", 1, 42, train_frame, eval_frame, data_root, cache_dir, smoke_dir,
            epochs=2,
        )
        (smoke_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"SMOKE TEST PASSED: acc={result['accuracy']:.4f} mae={result['mae']:.4f}")
        return

    channels_to_run = [args.channel] if args.channel else list(CHANNELS)
    fold_numbers = [args.fold] if args.fold else list(range(1, 6))

    status = {"runs": {}}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))

    for channel in channels_to_run:
        for fold_number in fold_numbers:
            fold_dir = guard_path(STEP_DIR / f"fold_{fold_number:02d}")
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_frame = folded.loc[folded["fold"] != fold_number - 1].reset_index(drop=True)
            eval_frame = folded.loc[folded["fold"] == fold_number - 1].reset_index(drop=True)
            (fold_dir / f"config_fold_{fold_number:02d}.json").write_text(
                json.dumps({
                    "fold": fold_number, "channels": list(CHANNELS),
                    "train_n": int(len(train_frame)), "eval_n": int(len(eval_frame)),
                    "training_seeds": list(TRAINING_SEEDS),
                    "shared": {k: v for k, v in config.items() if k != "fold_counts"},
                }, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            fold_metrics = {}
            existing = fold_dir / f"metrics_fold_{fold_number:02d}.json"
            if existing.exists():
                fold_metrics = json.loads(existing.read_text(encoding="utf-8"))
            for training_seed in TRAINING_SEEDS:
                key = f"{channel}_seed_{training_seed}"
                run_id = f"{channel}/fold_{fold_number:02d}/{key}"
                done = (
                    status["runs"].get(run_id) == "done"
                    and key in fold_metrics
                    and (fold_dir / key / "checkpoints" / "last.pth").exists()
                )
                if done:
                    log(f"skip completed {run_id}")
                    continue
                run_dir = guard_path(fold_dir / key)
                run_dir.mkdir(parents=True, exist_ok=True)
                log(f"training {run_id} ...")
                result = train_one_run(
                    channel, fold_number, training_seed, train_frame, eval_frame,
                    data_root, cache_dir, run_dir,
                )
                fold_metrics[key] = result
                existing.write_text(
                    json.dumps(fold_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                status["runs"][run_id] = "done"
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                log(
                    f"{run_id} done | acc={result['accuracy']:.4f} "
                    f"macro_f1={result['macro_f1']:.4f} mae={result['mae']:.4f} "
                    f"({result['train_minutes']} min)"
                )

    if args.fold or args.channel:
        log("partial mode finished; skipping aggregation")
        return

    summaries = {}
    for channel in CHANNELS:
        runs = []
        for fold_number in range(1, 6):
            fold_metrics = json.loads(
                (STEP_DIR / f"fold_{fold_number:02d}" / f"metrics_fold_{fold_number:02d}.json")
                .read_text(encoding="utf-8")
            )
            runs.extend(fold_metrics[f"{channel}_seed_{seed}"] for seed in TRAINING_SEEDS)
        if len(runs) != 15:
            raise SystemExit(f"channel {channel}: expected 15 runs, found {len(runs)}")
        summaries[channel] = aggregate_channel(runs)

    notes = [
        "五分类使用原始 0-4 Hard Label（程序内直接读取，未修改 manifest），"
        "num_classes=5；此前所有步骤均为二分类，故五个通道全部重新训练，"
        "无任何可复用的 run（5 通道 × 3 Training Seeds × 5 Folds = 75 次训练）。",
        "所有通道使用与 Step 01/02 完全一致的训练配置（384×384、ResNet50 "
        "ImageNet、AdamW 1e-4/1e-4、Batch Size 32、50 Epoch、CrossEntropyLoss、"
        "无 Scheduler、无 Early Stopping、统一增强策略、同一五折划分与 "
        "Fold Seed 42）；唯一变化是分类任务由二分类切换为五分类（本 Step 的主要变量）。",
        "五个通道对全部 996 个样本完全覆盖，各通道样本集与折划分完全一致。",
        "MAE 为五分类严重程度任务的重点辅助指标（|预测等级 − 真实等级| 的均值）；"
        "另记录相邻误判率（|Δ等级|=1）与远距误判率（|Δ等级|≥2），"
        "对应 TRAINING_PLAN 对相邻等级误判的重点关注。",
        "White (M) 通道结果将作为后续五分类多通道实验（Step 06）的 Baseline。",
        "汇总方式：先对每个 Training Seed 的五折计算 Mean ± Std，"
        "再对三个种子的均值计算最终 Mean ± Std（样本标准差，ddof=1）。",
    ]
    payload = {"config": config, "integrity": integrity, "summaries": summaries, "notes": notes}
    metrics_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    for channel in CHANNELS:
        for seed in TRAINING_SEEDS:
            for run in summaries[channel]["per_seed"][str(seed)]["fold_runs"]:
                row = {
                    "record_type": "fold_raw", "channel": channel,
                    "training_seed": seed,
                    **{k: run[k] for k in (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")},
                    "fold": run["fold"], "train_n": run["train_n"], "eval_n": run["eval_n"],
                    "majority_acc": run["majority_acc"],
                    "train_minutes": run["train_minutes"],
                }
                for cls in range(5):
                    row[f"precision_{cls}"] = run["per_class"]["precision"][cls]
                    row[f"recall_{cls}"] = run["per_class"]["recall"][cls]
                    row[f"f1_{cls}"] = run["per_class"]["f1"][cls]
                rows.append(row)
        for seed in TRAINING_SEEDS:
            seed_summary = summaries[channel]["per_seed"][str(seed)]["summary"]
            row = {"record_type": "seed_summary", "channel": channel, "training_seed": seed}
            row.update({k: seed_summary[k]["mean"] for k in
                        (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")})
            row.update({f"{k}_std": seed_summary[k]["std"] for k in
                        (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")})
            rows.append(row)
        final = summaries[channel]["final"]
        row = {"record_type": "final", "channel": channel, "training_seed": "all"}
        row.update({k: final[k]["mean"] for k in
                    (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")})
        row.update({f"{k}_std": final[k]["std"] for k in
                    (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")})
        rows.append(row)
    pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)

    figure_paths = make_figures(summaries, figures_dir)
    build_html_report(config, integrity, summaries, figure_paths, notes, report_path)

    log("FINAL | " + " | ".join(
        f"{c}: acc={summaries[c]['final']['accuracy']['mean']:.4f}"
        f"±{summaries[c]['final']['accuracy']['std']:.4f} "
        f"f1={summaries[c]['final']['macro_f1']['mean']:.4f}"
        f"±{summaries[c]['final']['macro_f1']['std']:.4f} "
        f"mae={summaries[c]['final']['mae']['mean']:.4f}"
        f"±{summaries[c]['final']['mae']['std']:.4f}"
        for c in CHANNELS
    ))
    log(f"report: {report_path}")


if __name__ == "__main__":
    main()
