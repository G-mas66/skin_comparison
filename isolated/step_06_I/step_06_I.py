"""Step 06_I - five-class multi-channel combinations under patient isolation.

Pre-registered combinations (fixed 2026-09-15 BEFORE any Step 05/06
multi-channel training, mirroring the user-approved Step 03 list for
cross-task comparability):
  M+MR, M+MB+MR, M+MB+MP+MR+MUV (all five), M+MB, MB+MR.
The M (White) baseline is imported verbatim from Step 05_I (identical
five-class configuration; not retrained).

Approved speed optimizations (user, 2026-09-15, effective from this step;
infrastructure only - no protocol hyperparameter touched):
  - persistent_workers=True on the training loader (removes per-epoch
    worker respawn; augmentation remains fully determined by the Training
    Seed via worker_init_fn)
  - datasets built once per (scheme, fold) and reused across the three seeds
  - torch.backends.cudnn.benchmark=True + channels_last memory format

Everything else matches Step 05 exactly: hard labels 0-4, ResNet50 ImageNet,
AdamW 1e-4/1e-4, batch 32, 50 epochs, CrossEntropyLoss, no scheduler, no
early stopping, unified augmentation, same five-fold split (Fold Seed 42).
"""

from __future__ import annotations

import argparse
import base64
import json
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
from torch.utils.data import DataLoader, Dataset

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
TRAINING_SEEDS = (42, 3407, 2026)
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_NAME = "resnet50"
PATIENT_ISOLATION = True
NUM_WORKERS = 8
CLASS_NAMES = tuple(f"Class {i}" for i in range(5))
STEP_05_DIR = STEP_DIR.parent / "step_05_I"

SCHEMES_TRAINED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("M_MR", ("M", "MR")),
    ("M_MB_MR", ("M", "MB", "MR")),
    ("ALL5", ("M", "MB", "MP", "MR", "MUV")),
    ("M_MB", ("M", "MB")),
    ("MB_MR", ("MB", "MR")),
)
SCHEME_DISPLAY = {
    "M_MR": "M+MR",
    "M_MB_MR": "M+MB+MR",
    "ALL5": "M+MB+MP+MR+MUV",
    "M_MB": "M+MB",
    "MB_MR": "MB+MR",
}
REFERENCE_CHANNEL = "M"
SINGLE_CHANNELS = ("M", "MB", "MP", "MR", "MUV")
METRIC_KEYS = ("accuracy", "macro_f1", "mae")
SCHEME_COLORS = {
    "M": "#4C72B0", "MB": "#DD8452", "MP": "#55A868", "MR": "#C44E52",
    "MUV": "#8172B3", "M_MR": "#937860", "M_MB_MR": "#da8bc3",
    "ALL5": "#8c8c8c", "M_MB": "#ccb974", "MB_MR": "#64b5cd",
}
INFRASTRUCTURE = {
    "persistent_workers": True,
    "dataset_reuse_per_fold": True,
    "cudnn_benchmark": True,
    "channels_last": True,
    "rlimit_nofile_raise": 65535,
    "note": (
        "user-approved 2026-09-15, effective from Step 06; infrastructure "
        "only, no protocol hyperparameter changed. Incident 2026-09-16 "
        "~04:5x: after 58/75 runs the process died with 'Too many open "
        "files' (errno 24) - persistent workers accumulate shared-memory "
        "FDs up to the default ulimit. Fix: raise RLIMIT_NOFILE to 65535 "
        "in-process (resource.setrlimit) and relaunch; training resumed "
        "from status.json with no lost runs."
    ),
}


def raise_nofile_limit() -> None:
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65535, hard if hard != resource.RLIM_INFINITY else 65535)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass

REPORT_NOTES_ZH = [
    "多通道组合于 2026-09-15 在任何五分类多通道训练开始前预先注册，"
    "沿用 Step 03 经用户批准的组合清单（M+MR、M+MB+MR、全五通道、M+MB、MB+MR），"
    "便于二分类与五分类路线横向对照。组合清单记录于 config_step_06_I.json。",
    "White（M）基线原样引用 Step 05_I 的 15 条真实五分类记录——其配置与本 Step "
    "在所有协议相关维度完全一致（任务、Hard Label 0-4、分辨率、五折划分、"
    "Fold Seed 42、Training Seeds、超参数、增强与预处理）。",
    "多通道输入按通道拼接 3 个 RGB 平面；ResNet50 conv1 权重在通道组间复制并除以 n。"
    "未做任何其他结构改动。",
    "【经批准的提速优化】（用户，2026-09-15，自本 Step 生效）：persistent_workers、"
    "每折数据集复用、cudnn.benchmark + channels_last。均为基础设施层面改动，"
    "协议超参数（Batch Size 32、Epoch 50、AdamW 1e-4/1e-4 等）零变化。",
    "全部方案使用与 Step 05 完全相同的 996 个样本与折划分。",
    "重点观察 Macro-F1、Accuracy、MAE 与相邻等级误判的变化，以及相邻类别之间的"
    "混淆变化（见混淆矩阵与相邻/远距误判率）。",
    "汇总方式：先对每个 Training Seed 的五折计算 Mean ± Std，再对三个种子的均值"
    "计算最终 Mean ± Std（样本标准差，ddof=1）。",
]


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def guard_path(path: Path) -> Path:
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


def run_integrity_checks(manifest_path: Path, data_root: Path, folded) -> dict:
    frame = load_manifest(manifest_path)
    lookup = discover_images(data_root)
    checks: dict[str, object] = {}
    for excluded in sorted(EXCLUDED_IDS):
        channels_holding = sorted(m for m, ids in lookup.items() if excluded in ids)
        checks[f"excluded_{excluded}_absent"] = {
            "in_manifest": not bool(set(frame["capture_id"]) & {excluded}),
            "channels_still_indexing_it": channels_holding,
            "passed": not channels_holding
            and not bool(set(frame["capture_id"]) & {excluded}),
        }
    overlap_ok = True
    overlap = {}
    for fold in range(5):
        evaluation = folded.loc[folded["fold"] == fold]
        training = folded.loc[folded["fold"] != fold]
        sample_overlap = sorted(set(evaluation["sample_id"]) & set(training["sample_id"]))
        patient_overlap = sorted(
            set(evaluation["patient_id"]) & set(training["patient_id"])
        )
        overlap[f"fold_{fold + 1}"] = {
            "sample_overlap": sample_overlap, "patient_overlap": patient_overlap,
        }
        overlap_ok = overlap_ok and not sample_overlap and not patient_overlap
    checks["fold_overlap"] = overlap
    checks["fold_overlap_passed"] = {"passed": overlap_ok}
    per_channel = {}
    for channel in SINGLE_CHANNELS:
        missing = sorted(set(frame["capture_id"]) - set(lookup[channel]))
        per_channel[channel] = {"manifest_ids_without_image": missing[:5],
                                "passed": not missing}
    checks["channel_available"] = per_channel
    checks["task"] = {"task": TASK, "num_classes": NUM_CLASSES,
                      "mapping": "hard label: original 0-4 used directly",
                      "passed": NUM_CLASSES == 5}
    per_fold = {}
    for fold in range(5):
        evaluation = folded.loc[folded["fold"] == fold]
        training = folded.loc[folded["fold"] != fold]
        per_fold[f"fold_{fold + 1}"] = {
            "train_n": int(len(training)), "eval_n": int(len(evaluation)),
            "train_label_hist": {str(k): int(v) for k, v in
                                 sorted(training["label"].value_counts().items())},
            "eval_label_hist": {str(k): int(v) for k, v in
                                sorted(evaluation["label"].value_counts().items())},
        }
    checks["fold_composition"] = per_fold
    passed = all(
        entry["passed"] if isinstance(entry, dict) and "passed" in entry else True
        for entry in checks.values()
    )
    return {"passed": passed, "checks": checks}


def build_loaders(train_dataset: Dataset, eval_dataset: Dataset, seed: int):
    def worker_init(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        generator=generator, worker_init_fn=worker_init, pin_memory=True,
        drop_last=False, persistent_workers=INFRASTRUCTURE["persistent_workers"],
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=True,
    )
    return train_loader, eval_loader


def train_one_run(
    scheme_key, modalities, fold_number, training_seed,
    train_dataset, eval_dataset, train_frame_len, eval_frame_len,
    run_dir, epochs=EPOCHS, log_every=10,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = INFRASTRUCTURE["cudnn_benchmark"]
    seed_everything(training_seed)
    train_loader, eval_loader = build_loaders(train_dataset, eval_dataset, training_seed)

    model = build_classifier(
        MODEL_NAME, num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
        pretrained=True, dropout=0.3, input_channels=3 * len(modalities),
    ).to(device)
    if INFRASTRUCTURE["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
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
            f"scheme={SCHEME_DISPLAY[scheme_key]} fold={fold_number} "
            f"training_seed={training_seed} epochs={epochs} batch_size={BATCH_SIZE} "
            f"optimizer=AdamW lr={LR} weight_decay={WEIGHT_DECAY} "
            f"loss=CrossEntropyLoss scheduler=None early_stopping=False task=5class\n"
        )
        train_log.flush()
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_started = time.time()
            running_loss, running_correct, running_total = 0.0, 0, 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                if INFRASTRUCTURE["channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
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
                    f"{SCHEME_DISPLAY[scheme_key]} fold {fold_number} seed {training_seed} "
                    f"epoch {epoch}/{epochs} "
                    f"loss={running_loss / max(1, running_total):.4f} "
                    f"acc={running_correct / max(1, running_total):.4f}"
                )

    last_path = checkpoint_dir / "last.pth"
    torch.save(
        {"state_dict": model.state_dict(), "scheme": scheme_key,
         "channels": list(modalities), "fold": fold_number,
         "training_seed": training_seed, "epoch": epochs, "step": STEP_NAME,
         "task": TASK, "input_size": INPUT_SIZE},
        last_path,
    )

    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad(), open(eval_log_path, "w", encoding="utf-8") as eval_log:
        eval_log.write(
            f"scheme={SCHEME_DISPLAY[scheme_key]} fold={fold_number} "
            f"training_seed={training_seed} checkpoint=last.pth (epoch {epochs})\n"
        )
        for batch in eval_loader:
            images = batch["image"].to(device, non_blocking=True)
            if INFRASTRUCTURE["channels_last"]:
                images = images.contiguous(memory_format=torch.channels_last)
            logits = model(images)
            all_targets.extend(int(t) for t in batch["target"])
            all_preds.extend(int(p) for p in logits.argmax(1).cpu().tolist())

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
            f"train_n={train_frame_len} eval_n={eval_frame_len}\n"
        )

    return {
        "scheme": SCHEME_DISPLAY[scheme_key], "scheme_key": scheme_key,
        "channels": list(modalities), "fold": fold_number,
        "training_seed": training_seed,
        "accuracy": float(accuracy), "macro_f1": float(macro_f1), "mae": mae,
        "per_class": {"precision": [float(v) for v in precision],
                      "recall": [float(v) for v in recall],
                      "f1": [float(v) for v in f1s],
                      "support": [int(v) for v in support]},
        "confusion_matrix": cm.tolist(),
        "adjacent_error_rate": adjacent_rate, "far_error_rate": far_rate,
        "train_n": int(train_frame_len), "eval_n": int(eval_frame_len),
        "majority_acc": float(max(np.bincount(targets_array, minlength=5)) / len(targets_array)),
        "epochs": epochs, "param_count": int(parameter_count),
        "checkpoint": str(last_path.relative_to(STEP_DIR)),
        "train_minutes": round((time.time() - started) / 60, 2),
    }


def aggregate_channel(runs: list[dict]) -> dict:
    summary = {"per_seed": {}, "final": {}}

    def stats(values):
        array = np.array(values, dtype=float)
        return {"mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0}

    all_keys = (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")
    for seed in TRAINING_SEEDS:
        seed_runs = [r for r in runs if r["training_seed"] == seed]
        seed_summary = {key: stats([r[key] for r in seed_runs]) for key in all_keys}
        seed_summary["per_class"] = {
            metric: [stats([r["per_class"][metric][cls] for r in seed_runs])
                     for cls in range(5)]
            for metric in ("precision", "recall", "f1")
        }
        summary["per_seed"][str(seed)] = {"fold_runs": seed_runs, "summary": seed_summary}
    seed_means = {
        key: [summary["per_seed"][str(s)]["summary"][key]["mean"] for s in TRAINING_SEEDS]
        for key in all_keys
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


def load_single_channel_context():
    payload = json.loads(
        (STEP_05_DIR / "metrics_step_05_I.json").read_text(encoding="utf-8")
    )
    summaries = payload["summaries"]
    all_runs = []
    for channel in SINGLE_CHANNELS:
        for seed in TRAINING_SEEDS:
            all_runs.extend(summaries[channel]["per_seed"][str(seed)]["fold_runs"])
    if len(all_runs) != 75:
        raise RuntimeError(f"expected 75 context runs from step_05_I, found {len(all_runs)}")
    return summaries, all_runs


def make_figures(context_summaries, combo_summaries, figures_dir) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    schemes = (
        [(SCHEME_DISPLAY.get(k, k), k, "single") for k in SINGLE_CHANNELS]
        + [(SCHEME_DISPLAY[k], k, "combo") for k, _ in SCHEMES_TRAINED]
    )
    all_summaries = {**context_summaries, **combo_summaries}
    majority_acc = float(np.mean(
        [r["majority_acc"] for r in context_summaries["M"]["per_seed"]["42"]["fold_runs"]]
    ))
    series = [("seed 42", "42"), ("seed 3407", "3407"), ("seed 2026", "2026")]
    series_colors = ["#8ea6c4", "#4C72B0", "#2b3f66"]
    x = np.arange(len(schemes))
    width = 0.19
    for key, label, percent in (
        ("accuracy", "Accuracy", True),
        ("macro_f1", "Macro-F1", True),
        ("mae", "MAE (severity grades)", False),
        ("adjacent_error_rate", "Adjacent-class error rate", True),
        ("far_error_rate", "Far-class error rate (|Δgrade|≥2)", True),
    ):
        fig, ax = plt.subplots(figsize=(13, 4.6))
        groups = []
        for offset, (name, seed_key) in enumerate(series):
            means = [all_summaries[k]["per_seed"][seed_key]["summary"][key]["mean"]
                     for _, k, _ in schemes]
            stds = [all_summaries[k]["per_seed"][seed_key]["summary"][key]["std"]
                    for _, k, _ in schemes]
            bars = ax.bar(x + (offset - 1.5) * width, means, width, yerr=stds,
                          label=name, color=series_colors[offset], capsize=3)
            groups.append((bars, means, stds))
        final_means = [all_summaries[k]["final"][key]["mean"] for _, k, _ in schemes]
        final_stds = [all_summaries[k]["final"][key]["std"] for _, k, _ in schemes]
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
                    text, ha="center", va="bottom", fontsize=7.5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
        if key == "accuracy":
            ax.axhline(majority_acc, color="red", linestyle="--", linewidth=1.2)
            ax.text(len(schemes) - 0.45, majority_acc + 0.012,
                    f"majority baseline {majority_acc * 100:.1f}%", color="darkred",
                    fontsize=8, ha="right",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels([d for d, _, _ in schemes], rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(
            f"Step 06_I 5-class | White baseline vs multi-channel schemes | {label} "
            "(bar = mean over 5 folds; singles reused from Step 05)"
        )
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures_dir / f"{key}_by_scheme.{suffix}"
            fig.savefig(path, dpi=160)
            if suffix == "png":
                figure_paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        width = 0.075
        cx = np.arange(5)
        for index, (display, scheme_key, _) in enumerate(schemes):
            means = [all_summaries[scheme_key]["final"]["per_class"][metric][cls]["mean"]
                     for cls in range(5)]
            stds = [all_summaries[scheme_key]["final"]["per_class"][metric][cls]["std"]
                    for cls in range(5)]
            ax.bar(cx + (index - len(schemes) / 2 + 0.5) * width, means, width,
                   yerr=stds, label=display, color=SCHEME_COLORS.get(scheme_key),
                   capsize=2)
        ax.set_xticks(cx)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.capitalize())
        ax.set_ylabel(metric.capitalize())
    axes[-1].legend(fontsize=6.5, loc="upper right", ncol=2)
    fig.suptitle("Step 06_I 5-class | per-class metrics by scheme (final 3-seed mean ± std)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"per_class_by_scheme.{suffix}"
        fig.savefig(path, dpi=160)
        if suffix == "png":
            figure_paths.append(path)
    plt.close(fig)

    matrix_schemes = [REFERENCE_CHANNEL] + [k for k, _ in SCHEMES_TRAINED]
    fig, axes = plt.subplots(1, len(matrix_schemes),
                             figsize=(3.6 * len(matrix_schemes), 3.8))
    for ax, scheme_key in zip(axes, matrix_schemes):
        runs = [r for s in TRAINING_SEEDS
                for r in all_summaries[scheme_key]["per_seed"][str(s)]["fold_runs"]]
        summed = np.sum([np.array(r["confusion_matrix"]) for r in runs], axis=0)
        im = ax.imshow(summed, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(int(summed[i, j])), ha="center", va="center",
                        fontsize=6,
                        color="white" if summed[i, j] > summed.max() / 2 else "black")
        ax.set_xticks(range(5), labels=CLASS_NAMES, fontsize=6)
        ax.set_yticks(range(5), labels=CLASS_NAMES, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{SCHEME_DISPLAY.get(scheme_key, scheme_key)} (15 runs)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Step 06_I 5-class | confusion matrices (5 folds x 3 seeds; M reused from Step 05)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"confusion_matrices_by_scheme.{suffix}"
        fig.savefig(path, dpi=160)
        if suffix == "png":
            figure_paths.append(path)
    plt.close(fig)
    return figure_paths


def image_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_html_report(config, integrity, context_summaries, combo_summaries,
                      figure_paths, notes, output_path):
    all_summaries = {**context_summaries, **combo_summaries}
    schemes = (
        [(SCHEME_DISPLAY.get(k, k), k, "引用 Step 05") for k in SINGLE_CHANNELS]
        + [(SCHEME_DISPLAY[k], k, "本次训练") for k, _ in SCHEMES_TRAINED]
    )
    sections = []
    config_rows = "".join(
        f"<tr><td>{k}</td><td>{json.dumps(v, ensure_ascii=False, default=str)}</td></tr>"
        for k, v in config.items()
    )
    sections.append(f"<h2>实验配置</h2><table class='cfg'><tr><th>项</th><th>值</th></tr>{config_rows}</table>")

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

    sections.append("<h2>多通道方案 Fold 级原始结果（本次训练 75 条）</h2>")
    for scheme_key, _ in SCHEMES_TRAINED:
        rows = []
        for seed in TRAINING_SEEDS:
            for run in combo_summaries[scheme_key]["per_seed"][str(seed)]["fold_runs"]:
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
            f"<h3>方案 {SCHEME_DISPLAY[scheme_key]}</h3>"
            "<table><tr><th>Fold</th><th>Training Seed</th><th>Train N</th><th>Eval N</th>"
            "<th>Accuracy</th><th>Macro-F1</th><th>MAE</th><th>相邻误判率</th>"
            "<th>远距误判率</th><th>Minutes</th></tr>"
            f"{''.join(rows)}</table>"
        )

    all_keys = (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")
    summary_rows = []
    for display, scheme_key, origin in schemes:
        final = all_summaries[scheme_key]["final"]
        summary_rows.append(
            f"<tr><td>{display}</td><td>{origin}</td>"
            + "".join(
                f"<td>{final[key]['mean']:.4f} ± {final[key]['std']:.4f}</td>"
                for key in all_keys
            )
            + "</tr>"
        )
    sections.append(
        "<h2>全部方案最终汇总（每格 = 三种子最终 Mean ± Std）</h2>"
        "<table><tr><th>方案</th><th>来源</th><th>Accuracy</th><th>Macro-F1</th><th>MAE</th>"
        "<th>相邻误判率</th><th>远距误判率</th></tr>"
        f"{''.join(summary_rows)}</table>"
    )

    white = context_summaries[REFERENCE_CHANNEL]["final"]
    compare_rows = []
    for scheme_key, _ in SCHEMES_TRAINED:
        final = combo_summaries[scheme_key]["final"]
        compare_rows.append(
            f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td>"
            f"<td>{(final['accuracy']['mean'] - white['accuracy']['mean']) * 100:+.2f} pp</td>"
            f"<td>{(final['macro_f1']['mean'] - white['macro_f1']['mean']) * 100:+.2f} pp</td>"
            f"<td>{final['mae']['mean'] - white['mae']['mean']:+.4f}</td>"
            f"<td>{(final['adjacent_error_rate']['mean'] - white['adjacent_error_rate']['mean']) * 100:+.2f} pp</td>"
            f"</tr>"
        )
    sections.append(
        "<h2>相对 White 基线的变化</h2>"
        "<table><tr><th>方案</th><th>ΔAccuracy (pp)</th><th>ΔMacro-F1 (pp)</th>"
        "<th>ΔMAE</th><th>Δ相邻误判率 (pp)</th></tr>"
        f"{''.join(compare_rows)}</table>"
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
img {{ max-width: 1200px; margin: 8px 0; border: 1px solid #ddd; }}
</style></head>
<body>
<h1>{STEP_NAME}：{TASK_DISPLAY} 多通道实验（Patient Isolation = True）</h1>
<p>生成时间：{datetime.now(timezone.utc).isoformat()}</p>
{''.join(sections)}
</body></html>"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/root/autodl-tmp/data_all_384/manifest.csv")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data_all_384")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--scheme", choices=[k for k, _ in SCHEMES_TRAINED])
    args = parser.parse_args()

    raise_nofile_limit()

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
        figure_paths = make_figures(
            payload["context_summaries"], payload["combo_summaries"], figures_dir
        )
        build_html_report(
            payload["config"], payload["integrity"], payload["context_summaries"],
            payload["combo_summaries"], figure_paths, REPORT_NOTES_ZH, report_path,
        )
        log(f"report rebuilt at {report_path}")
        return

    log(f"{STEP_NAME} start | schemes={[SCHEME_DISPLAY[k] for k, _ in SCHEMES_TRAINED]} "
        f"resolution={INPUT_SIZE} isolation={PATIENT_ISOLATION} "
        f"fold_seed={FOLD_SEED} training_seeds={TRAINING_SEEDS}")
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
        "benchmark": "five-class multi-channel combinations vs White baseline",
        "task": TASK,
        "task_display": TASK_DISPLAY,
        "num_classes": NUM_CLASSES,
        "label_strategy": "hard label: original 0-4 severity used directly",
        "patient_isolation": PATIENT_ISOLATION,
        "fold_seed": FOLD_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "input_resolution": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "schemes_preregistered": [
            {"key": key, "channels": list(modalities),
             "rationale": "mirrors the user-approved Step 03 combination list "
                          "(pre-registered before any 5-class multi-channel training)"}
            for key, modalities in SCHEMES_TRAINED
        ],
        "white_baseline_source": (
            "Step 05_I metrics imported verbatim (identical five-class "
            "configuration; not retrained)"
        ),
        "single_channel_context_source": "Step 05_I metrics imported verbatim",
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
        "infrastructure": INFRASTRUCTURE,
        "num_workers": NUM_WORKERS,
        "precision": "fp32",
        "metrics": list(METRIC_KEYS) + ["adjacent_error_rate", "far_error_rate"],
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "fold_counts": {
            f"fold_{fold + 1}": int((folded["fold"] == fold).sum()) for fold in range(5)
        },
        "total_samples": int(len(folded)),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.smoke:
        smoke_dir = guard_path(logs_dir / "smoke" / "ALL5_fold_01_seed_42")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        train_frame = folded.loc[folded["fold"] != 0].reset_index(drop=True).head(96)
        eval_frame = folded.loc[folded["fold"] == 0].reset_index(drop=True).head(64)
        train_dataset = SkinDataset(train_frame, data_root,
                                    modalities=("M", "MB", "MP", "MR", "MUV"), training=True)
        eval_dataset = SkinDataset(eval_frame, data_root,
                                   modalities=("M", "MB", "MP", "MR", "MUV"), training=False)
        log("SMOKE TEST: scheme ALL5 (15-channel), fold 1, seed 42, 2 epochs, 5-class")
        result = train_one_run(
            "ALL5", ("M", "MB", "MP", "MR", "MUV"), 1, 42,
            train_dataset, eval_dataset, len(train_frame), len(eval_frame),
            smoke_dir, epochs=2,
        )
        (smoke_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"SMOKE TEST PASSED: acc={result['accuracy']:.4f} mae={result['mae']:.4f}")
        return

    schemes_to_run = [args.scheme] if args.scheme else [k for k, _ in SCHEMES_TRAINED]
    fold_numbers = [args.fold] if args.fold else list(range(1, 6))

    status = {"runs": {}}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))

    for scheme_key in schemes_to_run:
        modalities = dict(SCHEMES_TRAINED)[scheme_key]
        for fold_number in fold_numbers:
            fold_dir = guard_path(STEP_DIR / f"fold_{fold_number:02d}")
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_frame = folded.loc[folded["fold"] != fold_number - 1].reset_index(drop=True)
            eval_frame = folded.loc[folded["fold"] == fold_number - 1].reset_index(drop=True)
            (fold_dir / f"config_fold_{fold_number:02d}.json").write_text(
                json.dumps({
                    "fold": fold_number,
                    "schemes": {k: list(m) for k, m in SCHEMES_TRAINED},
                    "train_n": int(len(train_frame)), "eval_n": int(len(eval_frame)),
                    "training_seeds": list(TRAINING_SEEDS),
                    "shared": {k: v for k, v in config.items() if k != "fold_counts"},
                }, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            # datasets built once per (scheme, fold), reused across the three seeds
            train_dataset = SkinDataset(
                train_frame, data_root, modalities=modalities, training=True
            )
            eval_dataset = SkinDataset(
                eval_frame, data_root, modalities=modalities, training=False
            )
            fold_metrics = {}
            existing = fold_dir / f"metrics_fold_{fold_number:02d}.json"
            if existing.exists():
                fold_metrics = json.loads(existing.read_text(encoding="utf-8"))
            for training_seed in TRAINING_SEEDS:
                key = f"{scheme_key}_seed_{training_seed}"
                run_id = f"{scheme_key}/fold_{fold_number:02d}/{key}"
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
                    scheme_key, modalities, fold_number, training_seed,
                    train_dataset, eval_dataset, len(train_frame), len(eval_frame),
                    run_dir,
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

    if args.fold or args.scheme:
        log("partial mode finished; skipping aggregation")
        return

    combo_summaries = {}
    for scheme_key, _ in SCHEMES_TRAINED:
        runs = []
        for fold_number in range(1, 6):
            fold_metrics = json.loads(
                (STEP_DIR / f"fold_{fold_number:02d}" / f"metrics_fold_{fold_number:02d}.json")
                .read_text(encoding="utf-8")
            )
            runs.extend(fold_metrics[f"{scheme_key}_seed_{seed}"] for seed in TRAINING_SEEDS)
        if len(runs) != 15:
            raise SystemExit(f"scheme {scheme_key}: expected 15 runs, found {len(runs)}")
        combo_summaries[scheme_key] = aggregate_channel(runs)

    context_summaries, context_runs = load_single_channel_context()

    notes = [
        "Multi-channel combinations pre-registered before any five-class "
        "multi-channel training, mirroring the user-approved Step 03 list "
        "(M+MR, M+MB+MR, all five, M+MB, MB+MR) for cross-task comparability.",
        "White (M) baseline and the four single-channel results imported "
        "verbatim from Step 05_I (identical five-class hard-label configuration).",
        "Approved infrastructure optimizations effective from this step "
        "(persistent_workers, per-fold dataset reuse, cudnn.benchmark + "
        "channels_last); no protocol hyperparameter changed.",
        "All schemes use exactly the same 996 samples and fold split as Step 05.",
        "Aggregation: per-training-seed five-fold Mean +/- Std, then final "
        "Mean +/- Std over the three seed means (sample std, ddof=1).",
        "Dataset: 384x384 lossless PNG ResizePad outputs reused (no re-upload).",
    ]
    payload = {
        "config": config, "integrity": integrity,
        "context_summaries": context_summaries, "combo_summaries": combo_summaries,
        "notes": notes,
    }
    metrics_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    all_keys = (*METRIC_KEYS, "adjacent_error_rate", "far_error_rate")
    for run in context_runs:
        rows.append({
            "record_type": "fold_raw_reference",
            "scheme": run.get("scheme", run.get("channel")),
            "training_seed": run["training_seed"],
            **{k: run[k] for k in all_keys},
            "fold": run["fold"], "train_n": run["train_n"], "eval_n": run["eval_n"],
            "train_minutes": run.get("train_minutes"),
        })
    for scheme_key, _ in SCHEMES_TRAINED:
        for seed in TRAINING_SEEDS:
            for run in combo_summaries[scheme_key]["per_seed"][str(seed)]["fold_runs"]:
                row = {
                    "record_type": "fold_raw", "scheme": SCHEME_DISPLAY[scheme_key],
                    "training_seed": seed, **{k: run[k] for k in all_keys},
                    "fold": run["fold"], "train_n": run["train_n"], "eval_n": run["eval_n"],
                    "majority_acc": run["majority_acc"],
                    "train_minutes": run["train_minutes"],
                }
                rows.append(row)
        for seed in TRAINING_SEEDS:
            seed_summary = combo_summaries[scheme_key]["per_seed"][str(seed)]["summary"]
            row = {"record_type": "seed_summary", "scheme": SCHEME_DISPLAY[scheme_key],
                   "training_seed": seed}
            row.update({k: seed_summary[k]["mean"] for k in all_keys})
            row.update({f"{k}_std": seed_summary[k]["std"] for k in all_keys})
            rows.append(row)
        final = combo_summaries[scheme_key]["final"]
        row = {"record_type": "final", "scheme": SCHEME_DISPLAY[scheme_key],
               "training_seed": "all"}
        row.update({k: final[k]["mean"] for k in all_keys})
        row.update({f"{k}_std": final[k]["std"] for k in all_keys})
        rows.append(row)
    pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)

    figure_paths = make_figures(context_summaries, combo_summaries, figures_dir)
    build_html_report(config, integrity, context_summaries, combo_summaries,
                      figure_paths, REPORT_NOTES_ZH, report_path)

    white = context_summaries[REFERENCE_CHANNEL]["final"]
    log("FINAL | " + " | ".join(
        f"{SCHEME_DISPLAY[k]}: acc={combo_summaries[k]['final']['accuracy']['mean']:.4f}"
        f" f1={combo_summaries[k]['final']['macro_f1']['mean']:.4f} "
        f"mae={combo_summaries[k]['final']['mae']['mean']:.4f}"
        for k, _ in SCHEMES_TRAINED
    ) + f" | White ref: acc={white['accuracy']['mean']:.4f} f1={white['macro_f1']['mean']:.4f} mae={white['mae']['mean']:.4f}")
    log(f"report: {report_path}")


if __name__ == "__main__":
    main()
