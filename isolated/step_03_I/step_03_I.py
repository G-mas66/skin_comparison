"""Step 03_I - binary multi-channel combinations under patient isolation.

Pre-registered combinations (fixed before any multi-channel aggregation was
produced, based on the completed Step 02_I single-channel results):
  1. M+MR           - White plus the only channel that beat White in Step 02
                      (MR: acc +1.6pp, macro-F1 +1.6pp, lowest variance).
  2. M+MB+MR        - best pair plus MB, which performed on par with White,
                      to test for additive gains.
  3. M+MB+MP+MR+MUV - all five channels, the full-input upper bound.
  4. M+MB           - White plus the on-par channel; does MB add anything
                      to White? (added by user instruction 2026-09-15 while
                      schemes 1-3 were training, before any combination
                      result had been aggregated or compared)
  5. MB+MR          - best non-White pair; do the two strongest individual
                      channels suffice without White? (same amendment)

White (M) baseline and the four other single-channel results are imported
verbatim from Step 01_I / Step 02_I (identical configuration in every
protocol-relevant dimension); only the combination runs above are newly
trained here. Multi-modal inputs concatenate 3 RGB planes per channel into
3*n channels; the ImageNet ResNet50 conv1 weights are replicated across the
extra channel groups. Everything else (task, split, seeds, hyperparameters,
augmentation, preprocessing) is unchanged.
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
    roc_auc_score,
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
    validate_folds,
)
from model import build_classifier, count_parameters  # noqa: E402

STEP_DIR = Path(__file__).resolve().parent
STEP_NAME = STEP_DIR.name
TASK = "binary"
NUM_CLASSES = 2
TRAINING_SEEDS = (42, 3407, 2026)
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_NAME = "resnet50"
PATIENT_ISOLATION = True
NUM_WORKERS = 8
CLASS_NAMES = ("Mild (0-2)", "Severe (3-4)")
STEP_01_DIR = STEP_DIR.parent / "step_01_I"
STEP_02_DIR = STEP_DIR.parent / "step_02_I"

# Pre-registered multi-channel combinations (fixed before training started;
# M+MB and MB+MR appended by user instruction before any combination result
# was aggregated or compared).
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

METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "mae",
    "auc",
    "sensitivity",
    "specificity",
)

SCHEME_COLORS = {
    "M": "#4C72B0",
    "MB": "#DD8452",
    "MP": "#55A868",
    "MR": "#C44E52",
    "MUV": "#8172B3",
    "M_MR": "#937860",
    "M_MB_MR": "#da8bc3",
    "ALL5": "#8c8c8c",
    "M_MB": "#ccb974",
    "MB_MR": "#64b5cd",
}

# 中文版报告说明（渲染进 HTML 报告；metrics JSON 保留英文原始记录作为数据档案）。
REPORT_NOTES_ZH = [
    "多通道组合在训练开始前基于已完成的 Step 02_I 结果预先注册：M+MR"
    "（White 加上唯一优于 White 的通道）、M+MB+MR（最佳配对加上与 White 持平的通道）、"
    "全五通道。M+MB 与 MB+MR 由用户明确指示在前三个组合训练期间、且在任何组合结果"
    "被聚合或对比之前追加。修订后的五组合清单记录于 config_step_03_I.json，"
    "且在任何组合结果被观察到之后未再改动。",
    "White（M）基线与四个单通道结果原样引用自 Step 01_I / Step 02_I——"
    "其配置在所有协议相关维度完全一致（任务、标签映射、分辨率、五折划分、"
    "Fold Seed 42、Training Seeds 42/3407/2026、ResNet50 ImageNet、"
    "AdamW 1e-4/1e-4、Batch Size 32、50 Epoch、CrossEntropyLoss、无 Scheduler、"
    "无 Early Stopping、相同增强与预处理）。",
    "多通道输入按通道拼接 3 个 RGB 平面；ResNet50 conv1 权重在通道组间复制并除以 n，"
    "保持 ImageNet 预训练特征的意义。未做任何其他结构改动。",
    "全部方案使用与 Step 01/02 完全相同的 996 个样本与折划分（无通道相关过滤）。",
    "汇总方式：先对每个 Training Seed 的五折计算 Mean ± Std，再对三个种子的均值"
    "计算最终 Mean ± Std（样本标准差，ddof=1）。",
    "数据：384×384 无损 PNG ResizePad 输出复用 Step 01_I 的上传（未重新上传）。",
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
    """SkinDataset with an optional lossless cache (disabled in practice).

    The remote data root already holds the exact ResizePad(384) output as
    PNG and ResizePad is idempotent on it.
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

    per_channel_coverage = {}
    for channel in SINGLE_CHANNELS:
        missing = sorted(set(frame["capture_id"]) - set(lookup[channel]))
        per_channel_coverage[channel] = {
            "manifest_samples": int(len(frame)),
            "images_available": int(len(set(frame["capture_id"]) & set(lookup[channel]))),
            "manifest_ids_without_image": missing[:10],
            "passed": not missing,
        }
    checks["channel_available"] = per_channel_coverage

    checks["task"] = {
        "task": TASK,
        "num_classes": NUM_CLASSES,
        "mapping": "0,1,2->0 (mild); 3,4->1 (severe); generated dynamically",
        "passed": NUM_CLASSES == 2,
    }

    per_fold = {}
    for fold in range(5):
        evaluation = folded.loc[folded["fold"] == fold]
        training = folded.loc[folded["fold"] != fold]
        per_fold[f"fold_{fold + 1}"] = {
            "train_n": int(len(training)),
            "eval_n": int(len(evaluation)),
            "train_label_hist_binary": {
                str(k): int(v)
                for k, v in sorted(training["label"].map(binary_target).value_counts().items())
            },
            "eval_label_hist_binary": {
                str(k): int(v)
                for k, v in sorted(evaluation["label"].map(binary_target).value_counts().items())
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
    modalities: tuple[str, ...],
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    data_root: Path,
    cache_dir: Path | None,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = CachedSkinDataset(
        train_frame, data_root, modalities=modalities, training=True, cache_dir=cache_dir
    )
    eval_dataset = CachedSkinDataset(
        eval_frame, data_root, modalities=modalities, training=False, cache_dir=cache_dir
    )

    def worker_init(worker_id: int) -> None:
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
    return train_loader, eval_loader


def train_one_run(
    scheme_key: str,
    modalities: tuple[str, ...],
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

    train_loader, eval_loader = build_loaders(
        modalities, train_frame, eval_frame, data_root, cache_dir, training_seed
    )

    input_channels = 3 * len(modalities)
    model = build_classifier(
        MODEL_NAME,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        pretrained=True,
        dropout=0.3,
        input_channels=input_channels,
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
            f"scheme={scheme_key} channels={'+'.join(modalities)} "
            f"fold={fold_number} training_seed={training_seed} epochs={epochs} "
            f"batch_size={BATCH_SIZE} optimizer=AdamW lr={LR} "
            f"weight_decay={WEIGHT_DECAY} loss=CrossEntropyLoss "
            f"scheduler=None early_stopping=False input_channels={input_channels}\n"
        )
        train_log.flush()
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
                running_correct += (logits.argmax(dim=1) == targets01).sum().item()
                running_total += images.size(0)
            train_loss = running_loss / max(1, running_total)
            train_acc = running_correct / max(1, running_total)
            train_log.write(
                f"epoch={epoch:03d} loss={train_loss:.4f} acc={train_acc:.4f} "
                f"sec={time.time() - epoch_started:.1f}\n"
            )
            train_log.flush()
            if epoch % log_every == 0 or epoch == epochs:
                log(
                    f"{SCHEME_DISPLAY[scheme_key]} fold {fold_number} seed {training_seed} "
                    f"epoch {epoch}/{epochs} loss={train_loss:.4f} acc={train_acc:.4f}"
                )

    last_path = checkpoint_dir / "last.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scheme": scheme_key,
            "channels": list(modalities),
            "fold": fold_number,
            "training_seed": training_seed,
            "epoch": epochs,
            "step": STEP_NAME,
            "task": TASK,
            "input_size": INPUT_SIZE,
        },
        last_path,
    )

    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    with torch.no_grad(), open(eval_log_path, "w", encoding="utf-8") as eval_log:
        eval_log.write(
            f"scheme={SCHEME_DISPLAY[scheme_key]} fold={fold_number} "
            f"training_seed={training_seed} checkpoint=last.pth (epoch {epochs})\n"
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
            f"accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} mae={mae:.4f} "
            f"auc={auc:.4f} sensitivity={sensitivity:.4f} "
            f"specificity={specificity:.4f}\n"
            f"confusion_matrix (rows=true mild/severe, cols=pred)={cm.tolist()}\n"
            f"train_n={len(train_frame)} eval_n={len(eval_frame)}\n"
        )

    minutes = (time.time() - started) / 60
    return {
        "scheme": SCHEME_DISPLAY[scheme_key],
        "scheme_key": scheme_key,
        "channels": list(modalities),
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
# Aggregation, reference import, figures, report
# ---------------------------------------------------------------------------


def aggregate_channel(runs: list[dict]) -> dict:
    summary: dict[str, object] = {"per_seed": {}, "final": {}}

    def stats(values: list[float]) -> dict:
        array = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        }

    for seed in TRAINING_SEEDS:
        seed_runs = [r for r in runs if r["training_seed"] == seed]
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
        key: [
            summary["per_seed"][str(seed)]["summary"][key]["mean"]
            for seed in TRAINING_SEEDS
        ]
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


def load_single_channel_context() -> tuple[dict[str, dict], list[dict]]:
    """Import Step 02_I's aggregated single-channel summaries verbatim.

    Step 02_I already aggregated M (from Step 01_I) and MB/MP/MR/MUV with the
    identical aggregation code, so its per-channel summaries are reused
    directly as the White baseline and single-channel context.
    """

    payload = json.loads(
        (STEP_02_DIR / "metrics_step_02_I.json").read_text(encoding="utf-8")
    )
    summaries = payload["summaries"]
    all_runs = []
    for channel in SINGLE_CHANNELS:
        for seed in TRAINING_SEEDS:
            all_runs.extend(summaries[channel]["per_seed"][str(seed)]["fold_runs"])
    if len(all_runs) != 75:
        raise RuntimeError(f"expected 75 context runs from step_02_I, found {len(all_runs)}")
    return summaries, all_runs


def make_figures(
    context_summaries: dict[str, dict],
    combo_summaries: dict[str, dict],
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    schemes = [
        (SCHEME_DISPLAY.get(key, key), key, "reference (Step 01/02)")
        for key in SINGLE_CHANNELS
    ] + [
        (SCHEME_DISPLAY[key], key, "trained here") for key, _ in SCHEMES_TRAINED
    ]
    all_summaries = {**context_summaries, **combo_summaries}
    display_of = {key: display for display, key, _ in schemes}
    majority_acc = float(
        np.mean(
            [
                run["majority_acc"]
                for run in context_summaries[REFERENCE_CHANNEL]["per_seed"]["42"]["fold_runs"]
            ]
        )
    )

    # Grouped vertical bars per metric: x = scheme, series = seeds + overall.
    series = [("seed 42", "42"), ("seed 3407", "3407"), ("seed 2026", "2026")]
    series_colors = ["#8ea6c4", "#4C72B0", "#2b3f66"]
    x = np.arange(len(schemes))
    for key, label, percent in (
        ("accuracy", "Accuracy", True),
        ("macro_f1", "Macro-F1", True),
        ("mae", "MAE (binary)", False),
        ("auc", "AUC", True),
    ):
        fig, ax = plt.subplots(figsize=(13, 4.6))
        width = 0.19
        groups = []
        for offset, (name, seed_key) in enumerate(series):
            means = [
                all_summaries[scheme_key]["per_seed"][seed_key]["summary"][key]["mean"]
                for _, scheme_key, _ in schemes
            ]
            stds = [
                all_summaries[scheme_key]["per_seed"][seed_key]["summary"][key]["std"]
                for _, scheme_key, _ in schemes
            ]
            bars = ax.bar(
                x + (offset - 1.5) * width,
                means,
                width,
                yerr=stds,
                label=name,
                color=series_colors[offset],
                capsize=3,
            )
            groups.append((bars, means, stds))
        final_means = [all_summaries[k]["final"][key]["mean"] for _, k, _ in schemes]
        final_stds = [all_summaries[k]["final"][key]["std"] for _, k, _ in schemes]
        bars = ax.bar(
            x + 1.5 * width,
            final_means,
            width,
            yerr=final_stds,
            label="overall (3 seeds)",
            color="#333333",
            capsize=3,
        )
        groups.append((bars, final_means, final_stds))

        top = max(m + s for _, means, stds in groups for m, s in zip(means, stds))
        if percent:
            ax.set_ylim(0, max(1.02, top * 1.22))
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
            )
        else:
            ax.set_ylim(0, top * 1.45 + 1e-9)
        for bar, mean, std in zip(bars, final_means, final_stds):
            text = f"{mean * 100:.1f}%" if percent else f"{mean:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + (0.012 if percent else top * 0.02),
                text,
                ha="center",
                va="bottom",
                fontsize=7.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
            )
        if key == "accuracy":
            ax.axhline(majority_acc, color="red", linestyle="--", linewidth=1.2)
            ax.text(
                len(schemes) - 0.45,
                majority_acc + 0.012,
                f"majority baseline {majority_acc * 100:.1f}%",
                color="darkred",
                fontsize=8,
                ha="right",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([display for display, _, _ in schemes], rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(
            f"Step 03_I {TASK} | White baseline vs multi-channel schemes | {label} "
            "(bar = mean over 5 folds; error bar = std; singles reused from Step 01/02)"
        )
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures_dir / f"{key}_by_scheme.{suffix}"
            fig.savefig(path, dpi=160)
            if suffix == "png":
                figure_paths.append(path)
        plt.close(fig)

    # Per-class metrics: x = class, series = schemes (final over 3 seeds).
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        width = 0.09
        cx = np.arange(len(CLASS_NAMES))
        for index, (display, scheme_key, _) in enumerate(schemes):
            means = [
                all_summaries[scheme_key]["final"]["per_class"][metric][cls]["mean"]
                for cls in (0, 1)
            ]
            stds = [
                all_summaries[scheme_key]["final"]["per_class"][metric][cls]["std"]
                for cls in (0, 1)
            ]
            ax.bar(
                cx + (index - len(schemes) / 2 + 0.5) * width,
                means,
                width,
                yerr=stds,
                label=display,
                color=SCHEME_COLORS.get(scheme_key),
                capsize=2,
            )
        ax.set_xticks(cx)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.capitalize())
        ax.set_ylabel(metric.capitalize())
    axes[-1].legend(fontsize=7, loc="lower right", ncol=2)
    fig.suptitle(
        "Step 03_I binary | per-class metrics by scheme "
        "(final 3-seed mean ± std; singles reused from Step 01/02)"
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"per_class_by_scheme.{suffix}"
        fig.savefig(path, dpi=160)
        if suffix == "png":
            figure_paths.append(path)
    plt.close(fig)

    # Confusion matrices: White baseline + the three trained combos.
    matrix_schemes = [REFERENCE_CHANNEL] + [key for key, _ in SCHEMES_TRAINED]
    fig, axes = plt.subplots(1, len(matrix_schemes), figsize=(4.0 * len(matrix_schemes), 3.6))
    for ax, scheme_key in zip(axes, matrix_schemes):
        runs = [
            run
            for seed in TRAINING_SEEDS
            for run in all_summaries[scheme_key]["per_seed"][str(seed)]["fold_runs"]
        ]
        summed = np.sum([np.array(run["confusion_matrix"]) for run in runs], axis=0)
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
        ax.set_title(f"{display_of[scheme_key]} (sum of 15 runs)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Step 03_I binary | confusion matrices (5 folds x 3 seeds)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"confusion_matrices_by_scheme.{suffix}"
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
    context_summaries: dict[str, dict],
    combo_summaries: dict[str, dict],
    figure_paths: list[Path],
    notes: list[str],
    output_path: Path,
) -> None:
    all_summaries = {**context_summaries, **combo_summaries}
    schemes = [(d, k, o) for d, k, o in (
        [(SCHEME_DISPLAY.get(k, k), k, "引用 Step 01/02") for k in SINGLE_CHANNELS]
        + [(SCHEME_DISPLAY[k], k, "本次训练") for k, _ in SCHEMES_TRAINED]
    )]

    sections: list[str] = []

    config_rows = "".join(
        f"<tr><td>{key}</td><td>{json.dumps(value, ensure_ascii=False, default=str)}</td></tr>"
        for key, value in config.items()
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
                flag = (
                    "PASS"
                    if not res["sample_overlap"]
                    and (res["patient_overlap"] in ("not_required",) or not res["patient_overlap"])
                    else "FAIL"
                )
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
                    f"<td>{run['mae']:.4f}</td><td>{run['auc']:.4f}</td>"
                    f"<td>{run['sensitivity']:.4f}</td><td>{run['specificity']:.4f}</td>"
                    f"<td>{run['train_minutes']:.1f}</td>"
                    "</tr>"
                )
        sections.append(
            f"<h3>方案 {SCHEME_DISPLAY[scheme_key]}</h3>"
            "<table><tr><th>Fold</th><th>Training Seed</th><th>Train N</th><th>Eval N</th>"
            "<th>Accuracy</th><th>Macro-F1</th><th>MAE</th><th>AUC</th>"
            "<th>Sensitivity</th><th>Specificity</th><th>Minutes</th></tr>"
            f"{''.join(rows)}</table>"
        )

    summary_rows = []
    for display, scheme_key, origin in schemes:
        final = all_summaries[scheme_key]["final"]
        summary_rows.append(
            f"<tr><td>{display}</td><td>{origin}</td>"
            + "".join(
                f"<td>{final[key]['mean']:.4f} ± {final[key]['std']:.4f}</td>"
                for key in METRIC_KEYS
            )
            + "</tr>"
        )
    sections.append(
        "<h2>全部方案最终汇总（每格 = 三种子最终 Mean ± Std）</h2>"
        "<table><tr><th>方案</th><th>来源</th><th>Accuracy</th><th>Macro-F1</th>"
        "<th>MAE</th><th>AUC</th><th>Sensitivity</th><th>Specificity</th></tr>"
        f"{''.join(summary_rows)}</table>"
    )

    white = context_summaries[REFERENCE_CHANNEL]["final"]
    compare_rows = []
    for scheme_key, _ in SCHEMES_TRAINED:
        final = combo_summaries[scheme_key]["final"]
        delta_f1 = (final["macro_f1"]["mean"] - white["macro_f1"]["mean"]) * 100
        delta_acc = (final["accuracy"]["mean"] - white["accuracy"]["mean"]) * 100
        compare_rows.append(
            f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td>"
            f"<td>{delta_acc:+.2f} pp</td><td>{delta_f1:+.2f} pp</td></tr>"
        )
    sections.append(
        "<h2>相对 White 基线的变化</h2>"
        "<table><tr><th>方案</th><th>ΔAccuracy (pp)</th><th>ΔMacro-F1 (pp)</th></tr>"
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
img {{ max-width: 1200px; margin: 8px 0; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>{STEP_NAME}：二分类多通道实验（Patient Isolation = True）</h1>
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
    parser.add_argument("--manifest", default="/root/autodl-tmp/data_all_384/manifest.csv")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data_all_384")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--scheme", choices=[key for key, _ in SCHEMES_TRAINED])
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
            payload["context_summaries"], payload["combo_summaries"], figures_dir
        )
        build_html_report(
            payload["config"], payload["integrity"], payload["context_summaries"],
            payload["combo_summaries"], figure_paths, REPORT_NOTES_ZH,
            report_path,
        )
        log(f"report rebuilt at {report_path}")
        return

    log(
        f"{STEP_NAME} start | task={TASK} schemes={[SCHEME_DISPLAY[k] for k, _ in SCHEMES_TRAINED]} "
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
        log("INTEGRITY CHECKS FAILED - refusing to train")
        raise SystemExit(2)

    config = {
        "experiment_name": STEP_NAME,
        "benchmark": "binary multi-channel combinations vs White baseline",
        "task": TASK,
        "num_classes": NUM_CLASSES,
        "label_strategy": "hard label; binary mapping 0-2->0, 3-4->1 generated dynamically",
        "patient_isolation": PATIENT_ISOLATION,
        "fold_seed": FOLD_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "input_resolution": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "schemes_preregistered": [
            {"key": key, "channels": list(modalities),
             "rationale": rationale}
            for (key, modalities), rationale in zip(
                SCHEMES_TRAINED,
                (
                    "White plus MR, the only channel that beat White in Step 02_I "
                    "(acc +1.6pp, macro-F1 +1.6pp, lowest variance)",
                    "best pair plus MB, which performed on par with White; tests "
                    "additive gains",
                    "all five channels; full-input upper bound",
                    "White plus MB, the channel on par with White in Step 02_I; "
                    "user-added amendment before any combination result was "
                    "aggregated or compared",
                    "the two strongest individual channels without White; "
                    "user-added amendment before any combination result was "
                    "aggregated or compared",
                ),
            )
        ],
        "white_baseline_source": (
            "Step 01_I metrics imported verbatim via Step 02_I summaries "
            "(identical configuration; not retrained)"
        ),
        "single_channel_context_source": "Step 02_I metrics imported verbatim",
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
        "metrics": list(METRIC_KEYS),
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "fold_counts": {
            f"fold_{fold + 1}": int((folded["fold"] == fold).sum()) for fold in range(5)
        },
        "total_samples": int(len(folded)),
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.smoke:
        smoke_dir = guard_path(logs_dir / "smoke" / "ALL5_fold_01_seed_42")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        train_frame = folded.loc[folded["fold"] != 0].reset_index(drop=True).head(96)
        eval_frame = folded.loc[folded["fold"] == 0].reset_index(drop=True).head(64)
        log("SMOKE TEST: scheme ALL5 (15-channel input), fold 1, seed 42, 2 epochs")
        result = train_one_run(
            "ALL5", ("M", "MB", "MP", "MR", "MUV"), 1, 42,
            train_frame, eval_frame, data_root, cache_dir, smoke_dir, epochs=2,
        )
        (smoke_dir / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        log(
            f"SMOKE TEST PASSED: acc={result['accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f}"
        )
        return

    schemes_to_run = [args.scheme] if args.scheme else [key for key, _ in SCHEMES_TRAINED]
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

            fold_config = {
                "fold": fold_number,
                "schemes": {k: list(m) for k, m in SCHEMES_TRAINED},
                "train_n": int(len(train_frame)),
                "eval_n": int(len(eval_frame)),
                "training_seeds": list(TRAINING_SEEDS),
                "shared": {k: v for k, v in config.items() if k != "fold_counts"},
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
                    train_frame, eval_frame, data_root, cache_dir, run_dir,
                )
                fold_metrics[key] = result
                existing.write_text(
                    json.dumps(fold_metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                status["runs"][run_id] = "done"
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                log(
                    f"{run_id} done | acc={result['accuracy']:.4f} "
                    f"macro_f1={result['macro_f1']:.4f} auc={result['auc']:.4f} "
                    f"({result['train_minutes']} min)"
                )

    if args.fold or args.scheme:
        log("partial mode finished; skipping aggregation")
        return

    combo_summaries: dict[str, dict] = {}
    for scheme_key, _ in SCHEMES_TRAINED:
        runs = []
        for fold_number in range(1, 6):
            fold_metrics = json.loads(
                (
                    STEP_DIR / f"fold_{fold_number:02d}"
                    / f"metrics_fold_{fold_number:02d}.json"
                ).read_text(encoding="utf-8")
            )
            runs.extend(
                fold_metrics[f"{scheme_key}_seed_{seed}"] for seed in TRAINING_SEEDS
            )
        if len(runs) != 15:
            log(f"scheme {scheme_key}: expected 15 runs, found {len(runs)}")
            raise SystemExit(3)
        combo_summaries[scheme_key] = aggregate_channel(runs)

    context_summaries, context_runs = load_single_channel_context()

    notes = [
        "Multi-channel combinations were pre-registered based on the completed "
        "Step 02_I results: M+MR (White plus the only channel that beat White), "
        "M+MB+MR (best pair plus the on-par channel), and all five channels. "
        "M+MB and MB+MR were appended by explicit user instruction while the "
        "first three schemes were training, before any multi-channel result "
        "had been aggregated or compared. The amended five-scheme list is "
        "recorded in config_step_03_I.json and was not modified after any "
        "combination result was observed.",
        "White (M) baseline and the four single-channel results are imported "
        "verbatim from Step 01_I / Step 02_I, whose configurations are identical "
        "in every protocol-relevant dimension (task, label mapping, resolution, "
        "five-fold split, Fold Seed 42, Training Seeds 42/3407/2026, ResNet50 "
        "ImageNet, AdamW 1e-4/1e-4, batch 32, 50 epochs, CrossEntropyLoss, no "
        "scheduler, no early stopping, identical augmentation and preprocessing).",
        "Multi-channel inputs concatenate 3 RGB planes per channel; ResNet50 "
        "conv1 weights are replicated across channel groups and scaled by 1/n so "
        "the pretrained features remain meaningful. No other architectural change.",
        "All schemes use exactly the same 996 samples and fold split as Steps "
        "01/02 (no channel-related filtering was needed).",
        "Aggregation: per-training-seed five-fold Mean +/- Std, then final "
        "Mean +/- Std over the three seed means (sample std, ddof=1).",
        "Dataset: 384x384 lossless PNG ResizePad outputs reused from the Step "
        "01_I upload (no re-upload).",
    ]
    payload = {
        "config": config,
        "integrity": integrity,
        "context_summaries": context_summaries,
        "combo_summaries": combo_summaries,
        "notes": notes,
    }
    metrics_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    for run in context_runs:
        rows.append(
            {
                "record_type": "fold_raw_reference",
                "scheme": run.get("scheme", run.get("channel")),
                "training_seed": run["training_seed"],
                **{key: run[key] for key in METRIC_KEYS},
                "fold": run["fold"],
                "train_n": run["train_n"],
                "eval_n": run["eval_n"],
                "train_minutes": run.get("train_minutes"),
            }
        )
    for scheme_key, _ in SCHEMES_TRAINED:
        for seed in TRAINING_SEEDS:
            for run in combo_summaries[scheme_key]["per_seed"][str(seed)]["fold_runs"]:
                rows.append(
                    {
                        "record_type": "fold_raw",
                        "scheme": SCHEME_DISPLAY[scheme_key],
                        "training_seed": seed,
                        **{key: run[key] for key in METRIC_KEYS},
                        "fold": run["fold"],
                        "train_n": run["train_n"],
                        "eval_n": run["eval_n"],
                        "majority_acc": run["majority_acc"],
                        "train_minutes": run["train_minutes"],
                    }
                )
        for seed in TRAINING_SEEDS:
            seed_summary = combo_summaries[scheme_key]["per_seed"][str(seed)]["summary"]
            row = {
                "record_type": "seed_summary",
                "scheme": SCHEME_DISPLAY[scheme_key],
                "training_seed": seed,
            }
            row.update({key: seed_summary[key]["mean"] for key in METRIC_KEYS})
            row.update({f"{key}_std": seed_summary[key]["std"] for key in METRIC_KEYS})
            rows.append(row)
        final = combo_summaries[scheme_key]["final"]
        row = {
            "record_type": "final",
            "scheme": SCHEME_DISPLAY[scheme_key],
            "training_seed": "all",
        }
        row.update({key: final[key]["mean"] for key in METRIC_KEYS})
        row.update({f"{key}_std": final[key]["std"] for key in METRIC_KEYS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)

    figure_paths = make_figures(context_summaries, combo_summaries, figures_dir)
    build_html_report(
        config, integrity, context_summaries, combo_summaries,
        figure_paths, REPORT_NOTES_ZH, report_path,
    )

    white = context_summaries[REFERENCE_CHANNEL]["final"]
    log("FINAL | " + " | ".join(
        f"{SCHEME_DISPLAY[k]}: acc={combo_summaries[k]['final']['accuracy']['mean']:.4f}"
        f"±{combo_summaries[k]['final']['accuracy']['std']:.4f} "
        f"f1={combo_summaries[k]['final']['macro_f1']['mean']:.4f}"
        f"±{combo_summaries[k]['final']['macro_f1']['std']:.4f}"
        for k, _ in SCHEMES_TRAINED
    ) + f" | White ref: acc={white['accuracy']['mean']:.4f} f1={white['macro_f1']['mean']:.4f}")
    log(f"report: {report_path}")


if __name__ == "__main__":
    main()
