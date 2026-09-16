"""Step 01_I - fixed-resolution White baseline under patient isolation.

Binary task (0-2 -> mild=0, 3-4 -> severe=1), single M (White) channel,
384x384 input, ResNet50 ImageNet-pretrained, AdamW 1e-4/1e-4, batch 32,
50 epochs, no scheduler, no early stopping.

Fold Seed = 42 builds the fixed five-fold split (Patient Isolation = True).
Training Seeds = [42, 3407, 2026] each run all five folds independently,
for a total of 15 training+evaluation runs.

Run modes:
  python step_01_I.py --manifest ... --data-root ... [--cache-dir ...]
      [--smoke] [--report-only] [--fold N]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
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
    roc_auc_score,
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
    validate_folds,
)
from model import build_classifier, count_parameters  # noqa: E402

STEP_DIR = Path(__file__).resolve().parent
STEP_NAME = STEP_DIR.name
TASK = "binary"
NUM_CLASSES = 2
MODALITIES = ("M",)
TRAINING_SEEDS = (42, 3407, 2026)
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_NAME = "resnet50"
PATIENT_ISOLATION = True
NUM_WORKERS = 8
CLASS_NAMES = ("Mild (0-2)", "Severe (3-4)")

METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "mae",
    "auc",
    "sensitivity",
    "specificity",
)

# 中文版报告说明（渲染进 HTML 报告；metrics JSON 保留英文原始记录作为数据档案）。
REPORT_NOTES_ZH = [
    "输入分辨率按 PROTOCOL §15.1 固定为 384×384；本 Step 执行该固定配置，"
    "不重新开展分辨率搜索。",
    "二分类标签（0-2 → 轻度 0，3-4 → 重度 1）在程序运行时由原始 0-4 标签动态生成，"
    "manifest 中始终保留原始标签。",
    "Fold Seed 42 生成患者隔离五折划分；Training Seeds 42/3407/2026 各自完整训练"
    "全部五个 Fold（共 15 次独立训练与评估）。",
    "汇总方式：先对每个 Training Seed 的五折计算 Mean ± Std，再对三个种子的均值"
    "计算最终 Mean ± Std（样本标准差，ddof=1）。",
    "本数据集的 patient_id 与 sample_id 一一对应，因此患者隔离条件天然满足。",
    "manifest 由 CEA.csv 规范化生成（M_id → sample_id，CEA_class → label，"
    "patient_id 原样保留）；划分前已剔除排除编号 69/296/769/770。",
    "【经批准的训练前修复】（用户，2026-09-15）：dataset.py 中的患者隔离折分配算法"
    "在任何训练开始前被替换。旧的距离求和贪心产生了退化的 396/396/202/1/1 划分"
    "（第 4、5 折各只含 1 个评估样本），且从未被任何已完成实验使用。新的确定性"
    "分层最少装载分配保持 Fold Seed = 42、患者隔离规则与完全相同的 996 个样本，"
    "产生均衡的每折约 199 个样本。",
    "【经批准的策略澄清】（用户，2026-09-15）：五个通道 M/MB/MP/MR/MUV 统一使用"
    "同一增强策略——水平翻转 p=0.5、旋转 U(-7°, +7°)、亮度 U(0.90, 1.10)、"
    "对比度 U(0.90, 1.10)；同一样本各通道共用同一组随机参数；评估集不使用随机增强。"
    "dataset.py 已修改为亮度/对比度作用于全部通道（此前仅作用于第一个 M 通道）。"
    "对本单 M 通道 Step 无行为影响。流水线顺序：等比缩放 → 黑色中心 Padding 到 "
    "384×384 → 增强 → ImageNet Normalize。",
    "【经批准的数据物流变更】（用户，2026-09-15）：图像在本地使用 dataset.py 中"
    "原版 ResizePad(384) 实现预处理并以无损 PNG 保存（无有损压缩）后上传，"
    "远程数据根目录指向这些 384×384 PNG。ResizePad 对已是 384 正方形的图像是幂等的，"
    "因此服务器端流水线不变，仅改变了确定性缩放的执行位置。预处理数据集中不含"
    "排除编号 69/296/769/770，包括 ID 69 的两个重复 MR 文件。",
    "【运行后报告审核】（用户批准的连续执行流程，2026-09-15）：对已生成图表的视觉"
    "审核发现四张横向柱状图的 Mean ± Std 数值标注与误差线端帽重叠（违反 "
    "PROTOCOL §13.4），且 majority 基线文字与柱子碰撞。仅修正了图表代码，"
    "并通过 --report-only 从未改动的已存指标重建图表与报告；所有训练结果、"
    "指标与配置均未改变。",
]


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
    """SkinDataset with a lossless cache of the deterministic ResizePad step.

    The cache stores the exact ResizePad(384) output as PNG for every source
    image. ResizePad is idempotent on an already-384 square canvas, so every
    downstream step (augmentation, normalization) is unchanged.
    """

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


def binary_target(label: int) -> int:
    return 1 if label >= 3 else 0


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
        in_manifest = bool(set(frame["capture_id"]) & {excluded})
        channels_holding = sorted(
            modality for modality, ids in lookup.items() if excluded in ids
        )
        checks[f"excluded_{excluded}_absent"] = {
            "in_manifest": not in_manifest,
            "channels_still_indexing_it": channels_holding,
            "passed": (not in_manifest) and not channels_holding,
        }

    mr69_files = sorted(
        p.name for p in data_root.rglob("*") if p.name.upper() in {"MR0069.JPG", "MR069.JPG"}
    )
    checks["mr_double_69_both_excluded"] = {
        "files_present_on_disk": mr69_files,
        "indexed_mr_69": 69 in lookup["MR"],
        "passed": (not mr69_files) or (69 not in lookup["MR"]),
    }

    overlap = {}
    for fold in range(5):
        evaluation = folded.loc[folded["fold"] == fold]
        training = folded.loc[folded["fold"] != fold]
        overlap[f"fold_{fold + 1}"] = {
            "sample_overlap": sorted(set(evaluation["sample_id"]) & set(training["sample_id"])),
            "patient_overlap": (
                sorted(set(evaluation["patient_id"]) & set(training["patient_id"]))
                if PATIENT_ISOLATION
                else "not_required"
            ),
        }
    checks["fold_overlap"] = overlap
    checks["fold_overlap_passed"] = {
        key: (not value["sample_overlap"])
        and (value["patient_overlap"] in ("not_required",) or not value["patient_overlap"])
        for key, value in overlap.items()
    }

    missing_m = sorted(set(frame["capture_id"]) - set(lookup["M"]))
    checks["image_label_match"] = {
        "manifest_samples": int(len(frame)),
        "m_images_available": int(len(set(frame["capture_id"]) & set(lookup["M"]))),
        "manifest_ids_without_m_image": missing_m[:10],
        "passed": not missing_m,
    }

    checks["task"] = {
        "task": TASK,
        "num_classes": NUM_CLASSES,
        "mapping": "0,1,2->0 (mild); 3,4->1 (severe); generated dynamically",
        "passed": NUM_CLASSES == 2,
    }

    checks["channel_available"] = {
        "required": list(MODALITIES),
        "m_image_count": len(lookup["M"]),
        "passed": bool(all(lookup[modality] for modality in MODALITIES)),
    }

    per_fold = {}
    for fold in range(5):
        evaluation = folded.loc[folded["fold"] == fold]
        training = folded.loc[folded["fold"] != fold]
        per_fold[f"fold_{fold + 1}"] = {
            "train_n": int(len(training)),
            "eval_n": int(len(evaluation)),
            "train_label_hist_5c": {str(k): int(v) for k, v in sorted(training["label"].value_counts().items())},
            "eval_label_hist_5c": {str(k): int(v) for k, v in sorted(evaluation["label"].value_counts().items())},
            "train_label_hist_binary": {
                str(k): int(v) for k, v in sorted(training["label"].map(binary_target).value_counts().items())
            },
            "eval_label_hist_binary": {
                str(k): int(v) for k, v in sorted(evaluation["label"].map(binary_target).value_counts().items())
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
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    data_root: Path,
    cache_dir: Path | None,
    seed: int,
) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
    train_dataset = CachedSkinDataset(
        train_frame,
        data_root,
        modalities=MODALITIES,
        training=True,
        cache_dir=cache_dir,
    )
    eval_dataset = CachedSkinDataset(
        eval_frame,
        data_root,
        modalities=MODALITIES,
        training=False,
        cache_dir=cache_dir,
    )

    def worker_init(worker_id: int) -> None:
        # torch.initial_seed() inside a worker = per-epoch base seed + worker id,
        # so augmentation randomness varies across epochs while staying seeded.
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
        worker_init_fn=worker_init,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataset, eval_dataset, train_loader, eval_loader


def train_one_run(
    fold_number: int,
    training_seed: int,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    data_root: Path,
    cache_dir: Path | None,
    run_dir: Path,
    epochs: int = EPOCHS,
    log_every: int = 10,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(training_seed)

    _, _, train_loader, eval_loader = build_loaders(
        train_frame, eval_frame, data_root, cache_dir, training_seed
    )

    model = build_classifier(
        MODEL_NAME,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        pretrained=True,
        dropout=0.3,
        input_channels=3,
    ).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = guard_path(run_dir / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = guard_path(run_dir / "train.log")
    eval_log_path = guard_path(run_dir / "eval.log")

    started = time.time()
    with open(train_log_path, "w", encoding="utf-8") as train_log:
        train_log.write(
            f"fold={fold_number} training_seed={training_seed} epochs={epochs} "
            f"batch_size={BATCH_SIZE} optimizer=AdamW lr={LR} weight_decay={WEIGHT_DECAY} "
            f"loss=CrossEntropyLoss scheduler=None early_stopping=False\n"
        )
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_started = time.time()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets01 = torch.tensor(
                    [binary_target(int(t)) for t in batch["target"]], device=device
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets01)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                running_correct += (
                    (logits.argmax(dim=1) == targets01).sum().item()
                )
                running_total += images.size(0)
            train_loss = running_loss / max(1, running_total)
            train_acc = running_correct / max(1, running_total)
            train_log.write(
                f"epoch={epoch:03d} loss={train_loss:.4f} acc={train_acc:.4f} "
                f"lr={LR} sec={time.time() - epoch_started:.1f}\n"
            )
            if epoch % log_every == 0 or epoch == epochs:
                log(
                    f"fold {fold_number} seed {training_seed} epoch {epoch}/{epochs} "
                    f"loss={train_loss:.4f} acc={train_acc:.4f}"
                )

    last_path = checkpoint_dir / "last.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "fold": fold_number,
            "training_seed": training_seed,
            "epoch": epochs,
            "step": STEP_NAME,
            "task": TASK,
            "modalities": MODALITIES,
            "input_size": INPUT_SIZE,
        },
        last_path,
    )

    # Evaluation - always with the epoch-50 (last) weights.
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    with torch.no_grad(), open(eval_log_path, "w", encoding="utf-8") as eval_log:
        eval_log.write(
            f"fold={fold_number} training_seed={training_seed} checkpoint=last.pth (epoch {epochs})\n"
        )
        for batch in eval_loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_targets.extend(binary_target(int(t)) for t in batch["target"])
            all_preds.extend(int(p) for p in preds.cpu().tolist())
            all_probs.extend(float(p) for p in probs.cpu().tolist())

        targets_array = np.array(all_targets)
        preds_array = np.array(all_preds)
        probs_array = np.array(all_probs)
        accuracy = accuracy_score(targets_array, preds_array)
        macro_f1 = f1_score(targets_array, preds_array, average="macro")
        mae = float(np.mean(np.abs(targets_array - preds_array)))
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            targets_array, preds_array, labels=[0, 1], zero_division=0
        )
        cm = confusion_matrix(targets_array, preds_array, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        try:
            auc = float(roc_auc_score(targets_array, probs_array))
        except ValueError:
            auc = float("nan")

        eval_log.write(
            f"accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} mae={mae:.4f} auc={auc:.4f} "
            f"sensitivity={sensitivity:.4f} specificity={specificity:.4f}\n"
            f"confusion_matrix (rows=true mild/severe, cols=pred)={cm.tolist()}\n"
            f"train_n={len(train_frame)} eval_n={len(eval_frame)}\n"
        )

    minutes = (time.time() - started) / 60
    return {
        "fold": fold_number,
        "training_seed": training_seed,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "mae": mae,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "per_class": {
            "precision": [float(v) for v in precision],
            "recall": [float(v) for v in recall],
            "f1": [float(v) for v in f1_per_class],
            "support": [int(v) for v in support],
        },
        "confusion_matrix": cm.tolist(),
        "train_n": int(len(train_frame)),
        "eval_n": int(len(eval_frame)),
        "majority_acc": float(max(np.bincount(targets_array)) / len(targets_array)),
        "epochs": epochs,
        "param_count": int(parameter_count),
        "checkpoint": str(last_path.relative_to(STEP_DIR)),
        "train_minutes": round(minutes, 2),
    }


# ---------------------------------------------------------------------------
# Aggregation, figures, report
# ---------------------------------------------------------------------------


def aggregate(all_runs: list[dict]) -> dict:
    summary: dict[str, object] = {"per_seed": {}, "final": {}}

    def stats(values: list[float]) -> dict:
        array = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        }

    for seed in TRAINING_SEEDS:
        seed_runs = [r for r in all_runs if r["training_seed"] == seed]
        seed_summary = {key: stats([r[key] for r in seed_runs]) for key in METRIC_KEYS}
        seed_summary["per_class"] = {
            metric: [
                stats([r["per_class"][metric][cls] for r in seed_runs]) for cls in (0, 1)
            ]
            for metric in ("precision", "recall", "f1")
        }
        summary["per_seed"][str(seed)] = {
            "fold_runs": seed_runs,
            "summary": seed_summary,
        }

    seed_means = {
        key: [summary["per_seed"][str(seed)]["summary"][key]["mean"] for seed in TRAINING_SEEDS]
        for key in METRIC_KEYS
    }
    summary["final"] = {key: stats(values) for key, values in seed_means.items()}
    summary["final"]["per_class"] = {
        metric: [
            stats(
                [
                    summary["per_seed"][str(seed)]["summary"]["per_class"][metric][cls]["mean"]
                    for seed in TRAINING_SEEDS
                ]
            )
            for cls in (0, 1)
        ]
        for metric in ("precision", "recall", "f1")
    }
    return summary


def save_metrics_csv(summary: dict, path: Path) -> None:
    rows = []
    for seed in TRAINING_SEEDS:
        for run in summary["per_seed"][str(seed)]["fold_runs"]:
            row = {"record_type": "fold_raw", "training_seed": seed, **{
                key: run[key] for key in METRIC_KEYS
            }}
            row.update({
                "fold": run["fold"],
                "train_n": run["train_n"],
                "eval_n": run["eval_n"],
                "majority_acc": run["majority_acc"],
                "precision_0": run["per_class"]["precision"][0],
                "recall_0": run["per_class"]["recall"][0],
                "f1_0": run["per_class"]["f1"][0],
                "precision_1": run["per_class"]["precision"][1],
                "recall_1": run["per_class"]["recall"][1],
                "f1_1": run["per_class"]["f1"][1],
                "train_minutes": run["train_minutes"],
            })
            rows.append(row)
        seed_summary = summary["per_seed"][str(seed)]["summary"]
        row = {"record_type": "seed_summary", "training_seed": seed}
        row.update({
            key: seed_summary[key]["mean"] for key in METRIC_KEYS
        })
        row.update({
            f"{key}_std": seed_summary[key]["std"] for key in METRIC_KEYS
        })
        rows.append(row)
    final = summary["final"]
    row = {"record_type": "final", "training_seed": "all"}
    row.update({key: final[key]["mean"] for key in METRIC_KEYS})
    row.update({f"{key}_std": final[key]["std"] for key in METRIC_KEYS})
    rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_figures(summary: dict, figures_dir: Path, integrity: dict) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    scheme = "White (M) 384x384"
    seeds = [str(seed) for seed in TRAINING_SEEDS]
    majority_acc = float(
        np.mean(
            [
                run["majority_acc"]
                for run in summary["per_seed"][seeds[0]]["fold_runs"]
            ]
        )
    )

    for key, label, percent in (
        ("accuracy", "Accuracy", True),
        ("macro_f1", "Macro-F1", True),
        ("mae", "MAE (binary)", False),
        ("auc", "AUC", True),
    ):
        fig, ax = plt.subplots(figsize=(9, 3.2))
        names = [f"seed {seed}" for seed in seeds] + ["overall"]
        means = [summary["per_seed"][seed]["summary"][key]["mean"] for seed in seeds]
        stds = [summary["per_seed"][seed]["summary"][key]["std"] for seed in seeds]
        means.append(summary["final"][key]["mean"])
        stds.append(summary["final"][key]["std"])
        colors = ["#4C72B0", "#DD8452", "#55A868", "#333333"]
        bars = ax.barh(names, means, xerr=stds, color=colors, capsize=4)
        ax.invert_yaxis()
        if percent:
            ax.set_xlim(0, max(1.05, max(m + s for m, s in zip(means, stds)) + 0.28))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        else:
            ax.set_xlim(0, max(means + stds) * 1.45 + 1e-9)
        # Labels sit beyond the error-bar cap with a white box so they never
        # overlap the bar, error bar, legend, or baseline text (PROTOCOL 13.4).
        for bar, mean, std in zip(bars, means, stds):
            text = f"{mean * 100:.2f}%" if percent else f"{mean:.4f}"
            offset = 0.015 if percent else max(means) * 0.02
            ax.text(
                mean + std + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{text} ± {std * 100:.2f}" if percent else f"{text} ± {std:.4f}",
                va="center",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
            )
        if key == "accuracy":
            ax.axvline(majority_acc, color="red", linestyle="--", linewidth=1.2)
            ax.set_ylim(len(names) - 0.4, -1.0)  # headroom row above the first bar
            ax.text(
                majority_acc,
                -0.72,
                f"majority baseline {majority_acc * 100:.2f}%",
                color="darkred",
                fontsize=9,
                ha="left",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
            )
        ax.set_xlabel(label)
        ax.set_title(
            f"Step 01_I {TASK} | {scheme} | {label} "
            "(bar = mean over 5 folds; error bar = std)"
        )
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures_dir / f"{key}_bar.{suffix}"
            fig.savefig(path, dpi=160)
            if suffix == "png":
                figure_paths.append(path)
        plt.close(fig)

    # Per-class grouped bars (precision/recall/F1 per class, per seed + overall).
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        width = 0.2
        x = np.arange(len(CLASS_NAMES))
        for index, (name, color) in enumerate(
            zip([f"seed {s}" for s in seeds] + ["overall"], ["#4C72B0", "#DD8452", "#55A868", "#333333"])
        ):
            if name == "overall":
                means = [summary["final"]["per_class"][metric][cls]["mean"] for cls in (0, 1)]
                stds = [summary["final"]["per_class"][metric][cls]["std"] for cls in (0, 1)]
            else:
                means = [
                    summary["per_seed"][name.split()[1]]["summary"]["per_class"][metric][cls]["mean"]
                    for cls in (0, 1)
                ]
                stds = [
                    summary["per_seed"][name.split()[1]]["summary"]["per_class"][metric][cls]["std"]
                    for cls in (0, 1)
                ]
            ax.bar(
                x + (index - 1.5) * width,
                means,
                width,
                yerr=stds,
                label=name,
                color=color,
                capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.1)
        ax.set_title(metric.capitalize())
        ax.set_ylabel(metric.capitalize())
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Step 01_I {TASK} | {scheme} | per-class metrics (mean ± std over folds)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"per_class_metrics.{suffix}"
        fig.savefig(path, dpi=160)
        if suffix == "png":
            figure_paths.append(path)
    plt.close(fig)

    # Confusion matrices summed over folds, one panel per seed.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, seed in zip(axes, seeds):
        summed = np.sum(
            [np.array(run["confusion_matrix"]) for run in summary["per_seed"][seed]["fold_runs"]],
            axis=0,
        )
        im = ax.imshow(summed, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, str(int(summed[row, col])),
                    ha="center", va="center",
                    color="white" if summed[row, col] > summed.max() / 2 else "black",
                )
        ax.set_xticks([0, 1], labels=CLASS_NAMES)
        ax.set_yticks([0, 1], labels=CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"seed {seed} (sum of 5 folds)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Step 01_I {TASK} | {scheme} | confusion matrices")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"confusion_matrices.{suffix}"
        fig.savefig(path, dpi=160)
        if suffix == "png":
            figure_paths.append(path)
    plt.close(fig)

    return figure_paths


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_html_report(
    config: dict,
    integrity: dict,
    summary: dict,
    figure_paths: list[Path],
    notes: list[str],
    output_path: Path,
) -> None:
    final = summary["final"]
    sections: list[str] = []

    config_rows = "".join(
        f"<tr><td>{key}</td><td>{json.dumps(value, ensure_ascii=False, default=str)}</td></tr>"
        for key, value in config.items()
    )
    sections.append(f"<h2>实验配置</h2><table class='cfg'><tr><th>项</th><th>值</th></tr>{config_rows}</table>")

    check_rows = []
    for name, outcome in integrity["checks"].items():
        if isinstance(outcome, dict) and "passed" in outcome:
            flag = "PASS" if outcome["passed"] else "FAIL"
            check_rows.append(f"<tr><td>{name}</td><td>{flag}</td></tr>")
        elif name == "fold_overlap":
            for fold, res in outcome.items():
                flag = "PASS" if not res["sample_overlap"] and (res["patient_overlap"] in ("not_required",) or not res["patient_overlap"]) else "FAIL"
                check_rows.append(f"<tr><td>{name} {fold}</td><td>{flag}</td></tr>")
    sections.append(
        f"<h2>数据完整性检查（PROTOCOL §17）</h2>"
        f"<p>总体：<b>{'PASS' if integrity['passed'] else 'FAIL'}</b></p>"
        f"<table class='cfg'><tr><th>检查项</th><th>结果</th></tr>{''.join(check_rows)}</table>"
    )

    fold_rows = []
    for seed in TRAINING_SEEDS:
        for run in summary["per_seed"][str(seed)]["fold_runs"]:
            fold_rows.append(
                "<tr>"
                f"<td>{run['fold']}</td><td>{seed}</td>"
                f"<td>{run['train_n']}</td><td>{run['eval_n']}</td>"
                f"<td>{run['accuracy']:.4f}</td><td>{run['macro_f1']:.4f}</td>"
                f"<td>{run['mae']:.4f}</td><td>{run['auc']:.4f}</td>"
                f"<td>{run['sensitivity']:.4f}</td><td>{run['specificity']:.4f}</td>"
                f"<td>{run['train_minutes']:.1f}</td>"
                "</tr>"
            )
    sections.append(
        "<h2>15 条 Fold 级原始结果</h2>"
        "<table><tr><th>Fold</th><th>Training Seed</th><th>Train N</th><th>Eval N</th>"
        "<th>Accuracy</th><th>Macro-F1</th><th>MAE</th><th>AUC</th>"
        "<th>Sensitivity</th><th>Specificity</th><th>Minutes</th></tr>"
        f"{''.join(fold_rows)}</table>"
    )

    summary_rows = []
    for seed in TRAINING_SEEDS:
        seed_summary = summary["per_seed"][str(seed)]["summary"]
        summary_rows.append(
            "<tr>"
            f"<td>seed {seed}（五折 Mean ± Std）</td>"
            + "".join(
                f"<td>{seed_summary[key]['mean']:.4f} ± {seed_summary[key]['std']:.4f}</td>"
                for key in METRIC_KEYS
            )
            + "</tr>"
        )
    summary_rows.append(
        "<tr><td><b>最终（三种子 Mean ± Std）</b></td>"
        + "".join(
            f"<td><b>{final[key]['mean']:.4f} ± {final[key]['std']:.4f}</b></td>"
            for key in METRIC_KEYS
        )
        + "</tr>"
    )
    sections.append(
        "<h2>两层 Mean ± Std 汇总</h2>"
        "<table><tr><th>汇总层</th><th>Accuracy</th><th>Macro-F1</th><th>MAE</th>"
        "<th>AUC</th><th>Sensitivity</th><th>Specificity</th></tr>"
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
<head>
<meta charset="utf-8">
<title>{STEP_NAME} 报告</title>
<style>
body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ border-bottom: 2px solid #4C72B0; padding-bottom: 4px; }}
table {{ border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
th, td {{ border: 1px solid #bbb; padding: 4px 10px; text-align: center; }}
th {{ background: #eef2f8; }}
table.cfg td:last-child {{ text-align: left; }}
img {{ max-width: 1000px; margin: 8px 0; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>{STEP_NAME}：二分类 White (M) 384×384 基线（Patient Isolation = True）</h1>
<p>生成时间：{datetime.now(timezone.utc).isoformat()}</p>
{''.join(sections)}
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/root/autodl-tmp/data_all/manifest.csv")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data_all")
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/cache_384")
    parser.add_argument("--smoke", action="store_true", help="quick pipeline test only")
    parser.add_argument("--report-only", action="store_true", help="rebuild report from saved metrics")
    parser.add_argument("--fold", type=int, choices=range(1, 6), help="run a single fold only")
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
        figure_paths = make_figures(
            payload["summary"], figures_dir, payload["integrity"]
        )
        build_html_report(
            payload["config"], payload["integrity"], payload["summary"],
            figure_paths, REPORT_NOTES_ZH, report_path,
        )
        log(f"report rebuilt at {report_path}")
        return

    log(f"{STEP_NAME} start | task={TASK} num_classes={NUM_CLASSES} "
        f"channels={MODALITIES} resolution={INPUT_SIZE} isolation={PATIENT_ISOLATION} "
        f"fold_seed={FOLD_SEED} training_seeds={TRAINING_SEEDS}")

    folded = make_five_folds(manifest_path, patient_isolation=PATIENT_ISOLATION, seed=FOLD_SEED)
    integrity = run_integrity_checks(manifest_path, data_root, folded)
    (logs_dir / "integrity_checks.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"integrity checks passed: {integrity['passed']}")
    if not integrity["passed"]:
        log("INTEGRITY CHECKS FAILED - refusing to train")
        raise SystemExit(2)

    config = {
        "experiment_name": STEP_NAME,
        "benchmark": "fixed-resolution white-light baseline (resolution already fixed by PROTOCOL 15.1)",
        "task": TASK,
        "num_classes": NUM_CLASSES,
        "label_strategy": "hard label; binary mapping 0-2->0, 3-4->1 generated dynamically",
        "patient_isolation": PATIENT_ISOLATION,
        "fold_seed": FOLD_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "input_resolution": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "input_channels": list(MODALITIES),
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
        "num_workers": NUM_WORKERS,
        "precision": "fp32",
        "metrics": list(METRIC_KEYS),
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "cache_note": "lossless PNG cache of the deterministic ResizePad(384) output; identical preprocessing semantics",
        "fold_counts": {
            f"fold_{fold + 1}": int((folded["fold"] == fold).sum()) for fold in range(5)
        },
        "total_samples": int(len(folded)),
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.smoke:
        smoke_dir = guard_path(logs_dir / "smoke" / "fold_01_seed_42")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        train_frame = folded.loc[folded["fold"] != 0].reset_index(drop=True).head(96)
        eval_frame = folded.loc[folded["fold"] == 0].reset_index(drop=True).head(64)
        log("SMOKE TEST: fold 1, seed 42, 2 epochs, 96 train / 64 eval samples")
        result = train_one_run(
            1, 42, train_frame, eval_frame, data_root, cache_dir,
            smoke_dir, epochs=2,
        )
        (smoke_dir / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        log(f"SMOKE TEST PASSED: acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")
        return

    status = {"runs": {}}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))

    fold_numbers = [args.fold] if args.fold else list(range(1, 6))
    for fold_number in fold_numbers:
        fold_dir = guard_path(STEP_DIR / f"fold_{fold_number:02d}")
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_frame = folded.loc[folded["fold"] != fold_number - 1].reset_index(drop=True)
        eval_frame = folded.loc[folded["fold"] == fold_number - 1].reset_index(drop=True)

        fold_config = {
            "fold": fold_number,
            "train_n": int(len(train_frame)),
            "eval_n": int(len(eval_frame)),
            "training_seeds": list(TRAINING_SEEDS),
            "shared": {k: v for k, v in config.items() if k not in {"fold_counts"}},
        }
        fold_config_path = guard_path(fold_dir / f"config_fold_{fold_number:02d}.json")
        fold_config_path.write_text(
            json.dumps(fold_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        fold_metrics: dict[str, dict] = {}
        existing = fold_dir / f"metrics_fold_{fold_number:02d}.json"
        if existing.exists():
            fold_metrics = json.loads(existing.read_text(encoding="utf-8"))

        for training_seed in TRAINING_SEEDS:
            key = f"seed_{training_seed}"
            run_id = f"fold_{fold_number:02d}/{key}"
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
                fold_number, training_seed, train_frame, eval_frame,
                data_root, cache_dir, run_dir,
            )
            fold_metrics[key] = result
            existing.write_text(
                json.dumps(fold_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            status["runs"][run_id] = "done"
            status_path.write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            log(
                f"{run_id} done | acc={result['accuracy']:.4f} "
                f"macro_f1={result['macro_f1']:.4f} auc={result['auc']:.4f} "
                f"({result['train_minutes']} min)"
            )

    if args.fold:
        log(f"single-fold mode: fold {args.fold} finished; skipping aggregation")
        return

    all_runs = []
    for fold_number in range(1, 6):
        fold_metrics = json.loads(
            (STEP_DIR / f"fold_{fold_number:02d}" / f"metrics_fold_{fold_number:02d}.json")
            .read_text(encoding="utf-8")
        )
        all_runs.extend(fold_metrics[key] for key in (f"seed_{s}" for s in TRAINING_SEEDS))
    if len(all_runs) != len(TRAINING_SEEDS) * 5:
        log(f"expected 15 runs, found {len(all_runs)}; aborting aggregation")
        raise SystemExit(3)

    summary = aggregate(all_runs)
    notes = [
        "Input resolution fixed at 384x384 per PROTOCOL 15.1; this step runs the "
        "fixed configuration rather than a new resolution search.",
        "Binary labels (0-2 mild -> 0, 3-4 severe -> 1) are generated dynamically "
        "from the original 0-4 labels; the manifest keeps original labels.",
        "Fold Seed 42 builds the patient-isolated five-fold split; Training Seeds "
        "42/3407/2026 each train all five folds independently (15 runs total).",
        "Aggregation: per-training-seed five-fold Mean +/- Std, then final Mean +/- Std "
        "over the three seed means (sample std, ddof=1).",
        "Provided metadata maps patient_id 1:1 to sample_id, so patient isolation is "
        "trivially satisfied for this manifest.",
        "Manifest normalized from CEA.csv (M_id -> sample_id, CEA_class -> label, "
        "patient_id kept as-is); excluded ids 69/296/769/770 removed before splitting.",
        "APPROVED PRE-TRAINING FIX (user, 2026-09-15): the patient-isolated fold "
        "assignment in dataset.py was replaced before any training run. The old "
        "distance-summing greedy produced a degenerate 396/396/202/1/1 split "
        "(folds 4-5 held a single evaluation sample each) and was never used by "
        "any completed experiment. The new deterministic stratified least-loaded "
        "assignment keeps Fold Seed = 42, the patient-isolation rules, and the "
        "exact same 996-sample set; it yields balanced folds (~199 each).",
        "APPROVED POLICY CLARIFICATION (user, 2026-09-15): one unified augmentation "
        "policy for every channel M/MB/MP/MR/MUV - horizontal flip p=0.5, rotation "
        "U(-7, +7) deg, brightness U(0.90, 1.10), contrast U(0.90, 1.10); identical "
        "parameter values shared across all channels of a sample; no random "
        "augmentation on evaluation folds. dataset.py was updated so brightness/"
        "contrast apply to every modality (previously only the first M channel). "
        "For this single-M-channel step the change has no behavioural effect. "
        "Pipeline order: aspect-preserving resize -> black center padding to "
        "384x384 -> augmentation -> ImageNet normalize.",
        "APPROVED DATA LOGISTICS CHANGE (user, 2026-09-15): images were "
        "preprocessed locally with the exact ResizePad(384) implementation from "
        "dataset.py and stored as lossless PNG (no lossy recompression); the "
        "remote data root points at these 384x384 PNGs. ResizePad is idempotent "
        "on an already-384 square canvas, so the on-server pipeline is "
        "unchanged; this only moves where the deterministic resize executes. "
        "Excluded ids 69/296/769/770 are absent from the preprocessed set, "
        "including both duplicate MR files for id 69.",
        "POST-RUN REPORT REVIEW (user-approved continuation flow, 2026-09-15): "
        "a visual audit of the generated figures found the Mean +/- Std value "
        "labels on the four horizontal bar charts overlapping the error-bar "
        "caps (PROTOCOL 13.4 violation) and the majority-baseline caption "
        "colliding with a bar. Only the figure code was corrected and the "
        "figures/report were rebuilt from the unchanged saved metrics via "
        "--report-only; no training result, metric, or configuration changed.",
    ]
    payload = {
        "config": config,
        "integrity": integrity,
        "summary": summary,
        "notes": notes,
    }
    metrics_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_metrics_csv(summary, metrics_csv_path)
    figure_paths = make_figures(summary, figures_dir, integrity)
    build_html_report(config, integrity, summary, figure_paths, REPORT_NOTES_ZH, report_path)

    final = summary["final"]
    log(
        "FINAL | "
        + " | ".join(
            f"{key}={final[key]['mean']:.4f}±{final[key]['std']:.4f}" for key in METRIC_KEYS
        )
    )
    log(f"report: {report_path}")


if __name__ == "__main__":
    main()
