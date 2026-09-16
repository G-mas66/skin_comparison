"""Step 10_I - five-class SOFT-LABEL DeepLIFT + channel perturbation.

Representative schemes (same pre-specified rule as Steps 04/07, fixed
2026-09-15 before any corresponding training ran): M+MB+MR, M+MR, and all
five channels. Weights are loaded read-only from the Step 09_I soft-label
combination checkpoints; ALL 15 models per scheme are analysed on their own
evaluation folds (no checkpoint selection). Perturbation and DeepLIFT
settings are identical to Step 07, and the Step 07 hard-label results are
imported verbatim for the Hard vs Soft interpretability comparison - the
goal is to see whether soft-label training changes which channels the
model relies on.
"""

from __future__ import annotations

import argparse
import base64
import json
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
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    EXCLUDED_IDS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    SEED as FOLD_SEED,
    SkinDataset,
    discover_images,
    load_manifest,
    make_five_folds,
)
from model import build_classifier  # noqa: E402

from captum.attr import DeepLift  # noqa: E402

# Spatial region definition for the center-vs-periphery attribution split:
# the central square covers the middle 50% of each side (192x192 of 384).
CENTER_BOX = (INPUT_SIZE // 4, INPUT_SIZE // 4, 3 * INPUT_SIZE // 4, 3 * INPUT_SIZE // 4)

STEP_DIR = Path(__file__).resolve().parent
STEP_NAME = STEP_DIR.name
TASK = "5class"
NUM_CLASSES = 5
TRAINING_SEEDS = (42, 3407, 2026)
MODEL_NAME = "resnet50"
PATIENT_ISOLATION = True
STEP_09_DIR = STEP_DIR.parent / "step_09_I"
STEP_07_DIR = STEP_DIR.parent / "step_07_I"

SCHEMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("M_MB_MR", ("M", "MB", "MR")),
    ("M_MR", ("M", "MR")),
    ("ALL5", ("M", "MB", "MP", "MR", "MUV")),
)
SCHEME_DISPLAY = {
    "M_MB_MR": "M+MB+MR",
    "M_MR": "M+MR",
    "ALL5": "M+MB+MP+MR+MUV",
}
ALL_CHANNELS = ("M", "MB", "MP", "MR", "MUV")
EXAMPLE_SEED = 42
EXAMPLE_FOLD = 1
EXAMPLES_PER_CLASS = 1

METRIC_KEYS = ("accuracy", "macro_f1", "mae")

# 中文版报告说明（渲染进 HTML 报告；metrics JSON 保留英文原始记录作为数据档案）。
REPORT_NOTES_ZH = [
    "代表性方案沿用 Step 04/07 预注册三方案（M+MB+MR、M+MR、全五通道），"
    "权重只读自 Step 09_I 的 Soft Label 组合 checkpoints。预指定规则与依据"
    "记录于 config_step_10_I.json。",
    "无任何 checkpoint 挑选：每个方案的全部 15 个模型（3 Training Seeds × 5 Folds）"
    "都参与分析，且各自严格只在其对应的评估折上评估，按 15 次 run 的 Mean ± Std 汇总。",
    "扰动方法（所有方案完全一致）：一次将一个通道整幅替换为黑色像素——与黑色中心 "
    "Padding 使用相同的填充色，保持扰动在分布内。不重训、不改划分。指标为五分类 "
    "Accuracy / Macro-F1 / MAE 的变化量。",
    "DeepLIFT：captum 实现；参考输入为经完全相同预处理的纯黑图像；"
    "归因目标为每个样本的预测类别（0-4）；通道重要性 = 每样本内该通道 |attribution| "
    "总和的占比，按 fold 模型平均后在 15 次 run 上汇总。",
    "可视化示例遵循预定规则（seed 42 / fold 1 模型、数据集顺序中每类第 1 个预测正确"
    "的评估样本，共 5 行），仅作定性展示；定量结论以 15 次 run 的完整聚合为准。",
    "权重只读自 Step 09_I 的 fold checkpoints（五分类 Soft Label 多通道模型）；Hard Label 对照引用 Step 07_I，用于 Hard vs Soft 通道依赖对比。",
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


def run_integrity_checks(manifest_path: Path, data_root: Path, folded) -> dict:
    """Compact PROTOCOL section 17 checks for this analysis-only step."""

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
    channel_coverage = {}
    for channel in ALL_CHANNELS:
        missing = sorted(set(frame["capture_id"]) - set(lookup[channel]))
        channel_coverage[channel] = {
            "manifest_ids_without_image": missing[:5], "passed": not missing
        }
    checks["channel_available"] = channel_coverage
    checks["task"] = {"task": TASK, "num_classes": NUM_CLASSES, "passed": True}
    passed = all(
        entry["passed"] if isinstance(entry, dict) and "passed" in entry else True
        for entry in checks.values()
    )
    return {"passed": passed, "checks": checks}


def build_eval_loader(
    modalities: tuple[str, ...], eval_frame: pd.DataFrame, data_root: Path
) -> tuple[DataLoader, SkinDataset]:
    dataset = SkinDataset(
        eval_frame, data_root, modalities=modalities, training=False
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    return loader, dataset


def build_deeplift_loader(dataset: SkinDataset) -> DataLoader:
    """Small batches: DeepLIFT keeps the full backward graph of every
    nonlinearity, which multiplies activation memory far beyond a plain
    forward pass (batch 32 with 15 channels exceeds 31 GB)."""

    return DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )


def load_model(modalities: tuple[str, ...], checkpoint_path: Path, device) -> nn.Module:
    model = build_classifier(
        MODEL_NAME,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        pretrained=False,
        dropout=0.3,
        input_channels=3 * len(modalities),
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.eval()
    # DeepLIFT requires non-inplace ReLU; behaviour is unchanged otherwise.
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    _patch_bottleneck_reuse(model)
    return model


def _patch_bottleneck_reuse(model: nn.Module) -> None:
    """Give each ReLU use inside torchvision Bottleneck blocks its own module.

    torchvision's Bottleneck calls one shared self.relu three times per
    forward; captum's DeepLift requires every module to be used exactly once.
    ReLU is stateless and has no parameters, so replacing the shared instance
    with dedicated ones (and a matching forward) leaves the function computed
    by the network completely unchanged - it only enables attribution hooks.
    """

    import types

    from torchvision.models.resnet import Bottleneck

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    for module in model.modules():
        if isinstance(module, Bottleneck):
            module.relu1 = nn.ReLU(inplace=False)
            module.relu2 = nn.ReLU(inplace=False)
            module.forward = types.MethodType(forward, module)


def evaluate(model: nn.Module, loader: DataLoader, device,
             perturb_channel: str | None, modalities: tuple[str, ...]):
    black = torch.tensor(
        [(0.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        device=device,
    ).view(1, 3, 1, 1)
    targets: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            if perturb_channel is not None:
                start = 3 * modalities.index(perturb_channel)
                images[:, start:start + 3] = black
            logits = model(images)
            preds.extend(int(p) for p in logits.argmax(1).cpu())
            targets.extend(int(t) for t in batch["target"])
    targets_array = np.array(targets)
    preds_array = np.array(preds)
    return {
        "accuracy": float(accuracy_score(targets_array, preds_array)),
        "macro_f1": float(
            f1_score(targets_array, preds_array, average="macro", zero_division=0)
        ),
        "mae": float(np.mean(np.abs(targets_array - preds_array))),
        "predictions": preds,
        "targets": targets,
    }


def deeplift_channel_shares(model: nn.Module, loader: DataLoader, device,
                            modalities: tuple[str, ...]):
    """Per-channel attribution share plus center-region share of the total
    absolute attribution (center box = middle 50% of each side)."""

    deeplift = DeepLift(model)
    black = torch.tensor(
        [(0.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        device=device,
    ).view(1, 3, 1, 1)
    totals = {channel: 0.0 for channel in modalities}
    center_totals = {channel: 0.0 for channel in modalities}
    samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        baseline = images.clone()
        for index in range(len(modalities)):
            start = 3 * index
            baseline[:, start:start + 3] = black
        with torch.no_grad():
            logits = model(images)
            targets = logits.argmax(dim=1)
        attribution = deeplift.attribute(
            images, baselines=baseline, target=targets
        ).detach().abs()
        y0, x0, y1, x1 = CENTER_BOX
        for index, channel in enumerate(modalities):
            planes = slice(3 * index, 3 * index + 3)
            totals[channel] += float(attribution[:, planes].sum())
            center_totals[channel] += float(
                attribution[:, planes, y0:y1, x0:x1].sum()
            )
        samples += images.size(0)
    return {
        channel: {
            "mean_abs": totals[channel] / max(1, samples),
            "center_share": center_totals[channel] / (totals[channel] + 1e-12),
        }
        for channel in modalities
    }


def attribution_heatmap_overlay(attribution: torch.Tensor, image: torch.Tensor,
                                channel_index: int, ax: plt.Axes, title: str) -> None:
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    planes = image[3 * channel_index:3 * channel_index + 3].cpu()
    display = (planes * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    heat = attribution[3 * channel_index:3 * channel_index + 3].detach().abs().sum(0).cpu()
    heat = heat / (heat.max() + 1e-8)
    ax.imshow(display)
    ax.imshow(heat.numpy(), cmap="hot", alpha=0.45)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def render_examples(model: nn.Module, dataset: SkinDataset, device,
                    modalities: tuple[str, ...], out_path: Path) -> None:
    deeplift = DeepLift(model)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    picked = {cls: 0 for cls in range(NUM_CLASSES)}
    items = []
    for index in range(len(dataset)):
        item = dataset[index]
        target = int(item["target"])
        with torch.no_grad():
            logits = model(item["image"].unsqueeze(0).to(device))
        pred = int(logits.argmax(1))
        if pred == target and picked[target] < EXAMPLES_PER_CLASS:
            items.append((index, item, target))
            picked[target] += 1
        if all(v >= EXAMPLES_PER_CLASS for v in picked.values()):
            break

    rows = len(items)
    cols = len(modalities) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    if rows == 1:
        axes = axes[None, :]
    for row, (index, item, target) in enumerate(items):
        images = item["image"].unsqueeze(0).to(device)
        baseline = images.clone()
        for i in range(len(modalities)):
            baseline[:, 3 * i:3 * i + 3] = (0.0 - mean) / std
        with torch.no_grad():
            logits = model(images)
            target_class = int(logits.argmax(1))
        attribution = deeplift.attribute(images, baselines=baseline, target=target_class)[0]
        display = (images[0, 0:3] * std[0] + mean[0]).clamp(0, 1).cpu()
        axes[row, 0].imshow(display.permute(1, 2, 0).numpy())
        axes[row, 0].set_title(
            f"sample {item['sample_id']}\ntrue/pred {'severe' if target else 'mild'}",
            fontsize=8,
        )
        axes[row, 0].axis("off")
        for col, channel in enumerate(modalities, start=1):
            attribution_heatmap_overlay(
                attribution, images[0], col - 1, axes[row, col], channel
            )
    fig.suptitle(
        f"DeepLIFT per-channel attribution | {SCHEME_DISPLAY.get('x', '')}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def finalize_outputs(payload: dict, figures_dir: Path) -> None:
    """Write CSV, figures, and the HTML report from a metrics payload.

    Used both at the end of a fresh analysis run and by --report-only
    (rebuild from saved metrics without re-running any attribution)."""

    records = payload["records"]
    summary = payload["summary"]
    config = payload["config"]
    integrity = payload["integrity"]

    rows = []
    for record in records:
        row = {
            "record_type": "run",
            "scheme": record["scheme"],
            "fold": record["fold"],
            "training_seed": record["training_seed"],
            "baseline_accuracy": record["baseline"]["accuracy"],
            "baseline_macro_f1": record["baseline"]["macro_f1"],
            "baseline_mae": record["baseline"]["mae"],
        }
        for channel in record["channels"]:
            for metric in METRIC_KEYS:
                row[f"delta_{metric}_{channel}"] = record["perturbation_deltas"][channel][metric]
                row[f"after_{metric}_{channel}"] = record["perturbed"][channel][metric]
            row[f"deeplift_share_{channel}"] = record["deeplift_share"][channel]
            row[f"deeplift_center_share_{channel}"] = record["deeplift_center_share"][channel]
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        guard_path(STEP_DIR / f"metrics_{STEP_NAME}.csv"), index=False
    )

    scheme_colors = {"M_MB_MR": "#4C72B0", "M_MR": "#DD8452", "ALL5": "#55A868"}
    x = np.arange(len(ALL_CHANNELS))
    width = 0.25
    for metric, label in (
        ("macro_f1", "ΔMacro-F1 (drop)"),
        ("accuracy", "ΔAccuracy (drop)"),
        ("mae", "ΔMAE (change)"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4))
        for offset, (scheme_key, modalities) in enumerate(SCHEMES):
            xs, means, stds = [], [], []
            for position, channel in enumerate(ALL_CHANNELS):
                if channel in modalities:
                    values = summary[scheme_key]["perturbation_delta"][channel][metric]
                    means.append(values["mean"])
                    stds.append(values["std"])
                    xs.append(x[position] + (offset - 1) * width)
            ax.bar(xs, means, width, yerr=stds, label=SCHEME_DISPLAY[scheme_key],
                   color=scheme_colors[scheme_key], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_CHANNELS)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(label)
        ax.set_title(
            f"Step 10_I 5-class soft | channel perturbation impact | {label} "
            "(mean ± std over 15 runs per scheme)"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(figures_dir / f"perturbation_{metric}.png", dpi=160)
        fig.savefig(figures_dir / f"perturbation_{metric}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, (scheme_key, modalities) in enumerate(SCHEMES):
        xs, means, stds = [], [], []
        for position, channel in enumerate(ALL_CHANNELS):
            if channel in modalities:
                share = summary[scheme_key]["deeplift_share"][channel]
                means.append(share["mean"] * 100)
                stds.append(share["std"] * 100)
                xs.append(x[position] + (offset - 1) * width)
        ax.bar(xs, means, width, yerr=stds, label=SCHEME_DISPLAY[scheme_key],
               color=scheme_colors[scheme_key], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_CHANNELS)
    ax.set_ylabel("DeepLIFT |attribution| share (%)")
    ax.set_title(
        "Step 10_I 5-class soft | per-channel DeepLIFT importance share "
        "(mean ± std over 15 runs per scheme)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "deeplift_share.png", dpi=160)
    fig.savefig(figures_dir / "deeplift_share.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, (scheme_key, modalities) in enumerate(SCHEMES):
        xs, means, stds = [], [], []
        for position, channel in enumerate(ALL_CHANNELS):
            if channel in modalities:
                center = summary[scheme_key]["deeplift_center_share"][channel]
                means.append(center["mean"] * 100)
                stds.append(center["std"] * 100)
                xs.append(x[position] + (offset - 1) * width)
        ax.bar(xs, means, width, yerr=stds, label=SCHEME_DISPLAY[scheme_key],
               color=scheme_colors[scheme_key], capsize=3)
    ax.axhline(25.0, color="gray", linestyle=":", linewidth=1.0)
    ax.text(len(ALL_CHANNELS) - 0.45, 25.8,
            "25% = center-box area proportion", fontsize=8, ha="right",
            color="gray",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_CHANNELS)
    ax.set_ylabel("Share of channel |attribution| in center 192x192 (%)")
    ax.set_title(
        "Step 10_I 5-class soft | DeepLIFT center-vs-periphery split "
        "(center box = middle 50% of each side; mean ± std over 15 runs)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "deeplift_center_share.png", dpi=160)
    fig.savefig(figures_dir / "deeplift_center_share.pdf")
    plt.close(fig)

    figure_files = sorted(figures_dir.glob("*.png"))

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

    baseline_rows = []
    for scheme_key, modalities in SCHEMES:
        base = summary[scheme_key]["baseline"]
        baseline_rows.append(
            f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td>"
            + "".join(
                f"<td>{base[m]['mean']:.4f} ± {base[m]['std']:.4f}</td>"
                for m in ("accuracy", "macro_f1", "mae")
            )
            + "</tr>"
        )
    sections.append(
        "<h2>未扰动基线性能（15 个模型各自在其评估折，Mean ± Std）</h2>"
        "<table><tr><th>方案</th><th>Accuracy</th><th>Macro-F1</th><th>MAE</th></tr>"
        f"{''.join(baseline_rows)}</table>"
    )

    after_rows = []
    for scheme_key, modalities in SCHEMES:
        for channel in modalities:
            after = summary[scheme_key]["perturbed"][channel]
            after_rows.append(
                f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td><td>去掉 {channel}</td>"
                + "".join(
                    f"<td>{after[m]['mean']:.4f} ± {after[m]['std']:.4f}</td>"
                    for m in ("accuracy", "macro_f1", "mae")
                )
                + "</tr>"
            )
    sections.append(
        "<h2>通道扰动后的绝对性能（Mean ± Std over 15 runs）</h2>"
        "<table><tr><th>方案</th><th>扰动</th><th>Accuracy</th><th>Macro-F1</th><th>MAE</th></tr>"
        f"{''.join(after_rows)}</table>"
    )

    summary_rows = []
    for scheme_key, modalities in SCHEMES:
        entry = summary[scheme_key]
        cells = [f"<td>{SCHEME_DISPLAY[scheme_key]}</td>"]
        for channel in modalities:
            delta = entry["perturbation_delta"][channel]["macro_f1"]
            share = entry["deeplift_share"][channel]
            cells.append(
                f"<td>{channel}: ΔF1 {delta['mean']:+.4f}±{delta['std']:.4f}"
                f"<br>DeepLIFT {share['mean'] * 100:.1f}%±{share['std'] * 100:.1f}%</td>"
            )
        summary_rows.append("".join(cells))
    sections.append(
        "<h2>通道重要性汇总（每格 = 15 次 run 的 Mean ± Std）</h2>"
        "<table><tr><th>方案</th>" + "".join(
            f"<th>通道 {i + 1}</th>" for i in range(max(len(m) for _, m in SCHEMES))
        ) + f"{''.join(f'<tr>{row}</tr>' for row in summary_rows)}</table>"
    )

    spatial_rows = []
    for scheme_key, modalities in SCHEMES:
        for channel in modalities:
            center = summary[scheme_key]["deeplift_center_share"][channel]
            spatial_rows.append(
                f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td><td>{channel}</td>"
                f"<td>{center['mean'] * 100:.1f}% ± {center['std'] * 100:.1f}%</td>"
                f"<td>{(1 - center['mean']) * 100:.1f}%</td></tr>"
            )
    sections.append(
        "<h2>DeepLIFT 空间区域分析（中心 vs 外周）</h2>"
        "<p>中心区域定义：图像中央 50% × 50% 的正方形（384 图上的 192×192，"
        "占面积 25%）。若某通道的中心占比显著高于 25%，说明该通道的注意力向病灶中心集中。</p>"
        "<table><tr><th>方案</th><th>通道</th><th>中心占比（Mean ± Std）</th><th>外周占比</th></tr>"
        f"{''.join(spatial_rows)}</table>"
    )

    try:
        hard = json.loads(
            (STEP_07_DIR / "metrics_step_07_I.json").read_text(encoding="utf-8")
        )["summary"]
        compare_rows = []
        for scheme_key, modalities in SCHEMES:
            for channel in modalities:
                h = hard[scheme_key]
                s = summary[scheme_key]
                compare_rows.append(
                    f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td><td>{channel}</td>"
                    f"<td>{h['perturbation_delta'][channel]['macro_f1']['mean']:+.4f}</td>"
                    f"<td>{s['perturbation_delta'][channel]['macro_f1']['mean']:+.4f}</td>"
                    f"<td>{h['deeplift_share'][channel]['mean'] * 100:.1f}%</td>"
                    f"<td>{s['deeplift_share'][channel]['mean'] * 100:.1f}%</td>"
                    f"<td>{h['deeplift_center_share'][channel]['mean'] * 100:.1f}%</td>"
                    f"<td>{s['deeplift_center_share'][channel]['mean'] * 100:.1f}%</td>"
                    f"</tr>"
                )
        sections.append(
            "<h2>Hard（Step 07）vs Soft（本 Step）通道依赖对比</h2>"
            "<table><tr><th>方案</th><th>通道</th><th>ΔF1 扰动 (Hard)</th>"
            "<th>ΔF1 扰动 (Soft)</th><th>DeepLIFT 占比 (Hard)</th>"
            "<th>DeepLIFT 占比 (Soft)</th><th>中心占比 (Hard)</th>"
            "<th>中心占比 (Soft)</th></tr>"
            f"{''.join(compare_rows)}</table>"
        )
    except Exception as exc:  # comparison is supplementary, never fatal
        sections.append(f"<h2>Hard vs Soft 对比</h2><p>对比表不可用：{exc}</p>")

    images = "".join(
        f"<h3>{p.name}</h3><img src='data:image/png;base64,"
        f"{base64.b64encode(p.read_bytes()).decode('ascii')}' alt='{p.name}'>"
        for p in figure_files
    )
    sections.append(f"<h2>图表</h2>{images}")
    notes_html = "".join(f"<li>{note}</li>" for note in REPORT_NOTES_ZH)
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
<h1>{STEP_NAME}：五分类 Soft Label DeepLIFT + 通道扰动（Patient Isolation = True）</h1>
<p>生成时间：{datetime.now(timezone.utc).isoformat()}</p>
{''.join(sections)}
</body></html>"""
    report_path = guard_path(STEP_DIR / f"report_{STEP_NAME}.html")
    report_path.write_text(html, encoding="utf-8")
    log(f"report written: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/root/autodl-tmp/data_all_384/manifest.csv")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data_all_384")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    data_root = Path(args.data_root).expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logs_dir = guard_path(STEP_DIR / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = guard_path(STEP_DIR / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        payload = json.loads(
            (STEP_DIR / f"metrics_{STEP_NAME}.json").read_text(encoding="utf-8")
        )
        finalize_outputs(payload, figures_dir)
        return

    log(f"{STEP_NAME} start | schemes={[SCHEME_DISPLAY[k] for k, _ in SCHEMES]}")

    folded = make_five_folds(manifest_path, patient_isolation=PATIENT_ISOLATION, seed=FOLD_SEED)
    integrity = run_integrity_checks(manifest_path, data_root, folded)
    log(f"integrity checks passed: {integrity['passed']}")
    if not integrity["passed"]:
        log("INTEGRITY CHECKS FAILED - refusing to run analysis")
        raise SystemExit(2)

    config = {
        "experiment_name": STEP_NAME,
        "benchmark": "five-class soft-label DeepLIFT + channel perturbation (interpretability)",
        "task": TASK,
        "num_classes": NUM_CLASSES,
        "patient_isolation": PATIENT_ISOLATION,
        "fold_seed": FOLD_SEED,
        "representative_schemes": [
            {
                "key": key,
                "channels": list(modalities),
                "selection_basis": basis,
            }
            for (key, modalities), basis in zip(
                SCHEMES,
                (
                    "pre-specified mirror of the Step 04 interpretability schemes",
                    "pre-specified mirror of the Step 04 interpretability schemes",
                    "full five-channel input reference",
                ),
            )
        ],
        "scheme_selection_rule": (
            "explicit user confirmation 2026-09-15 before any Step 04 analysis; "
            "all 15 models per scheme analysed (no checkpoint selection)"
        ),
        "perturbation_method": (
            "one channel at a time replaced by black pixels (identical to the "
            "black center-padding fill, in-distribution); applied identically "
            "to every scheme"
        ),
        "deeplift_settings": {
            "implementation": "captum DeepLift",
            "reference": "all-black image through identical preprocessing",
            "target": "each sample's predicted class",
            "channel_importance": "mean share of summed |attribution| per channel per sample",
        },
        "visual_example_rule": (
            f"seed {EXAMPLE_SEED} / fold {EXAMPLE_FOLD} model; first "
            f"{EXAMPLES_PER_CLASS} correctly classified eval samples per class in "
            "dataset order (pre-specified, no cherry-picking)"
        ),
        "weights_source": "step_03_I fold checkpoints (read-only)",
        "input_resolution": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "model": MODEL_NAME,
        "no_training": True,
        "metrics": list(METRIC_KEYS),
    }
    config_path = guard_path(STEP_DIR / f"config_{STEP_NAME}.json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    records: list[dict] = []
    started = time.time()
    for scheme_key, modalities in SCHEMES:
        for fold_number in range(1, 6):
            eval_frame = folded.loc[folded["fold"] == fold_number - 1].reset_index(drop=True)
            fold_dir = guard_path(STEP_DIR / f"fold_{fold_number:02d}")
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / f"config_fold_{fold_number:02d}.json").write_text(
                json.dumps(
                    {"fold": fold_number, "scheme": SCHEME_DISPLAY[scheme_key],
                     "eval_n": int(len(eval_frame)), "shared": config},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            fold_metrics: dict[str, dict] = {}
            existing = fold_dir / f"metrics_fold_{fold_number:02d}.json"
            if existing.exists():
                fold_metrics = json.loads(existing.read_text(encoding="utf-8"))

            for seed in TRAINING_SEEDS:
                key = f"{scheme_key}_seed_{seed}"
                if key in fold_metrics:
                    log(f"skip completed {key}")
                    continue
                checkpoint = (
                    STEP_09_DIR / f"fold_{fold_number:02d}" / key / "checkpoints" / "last.pth"
                )
                model = load_model(modalities, checkpoint, device)
                loader, dataset = build_eval_loader(modalities, eval_frame, data_root)

                baseline = evaluate(model, loader, device, None, modalities)
                perturbed = {
                    channel: evaluate(model, loader, device, channel, modalities)
                    for channel in modalities
                }
                torch.cuda.empty_cache()
                shares = deeplift_channel_shares(
                    model, build_deeplift_loader(dataset), device, modalities
                )
                total_share = sum(v["mean_abs"] for v in shares.values()) + 1e-12

                run_record = {
                    "scheme": SCHEME_DISPLAY[scheme_key],
                    "scheme_key": scheme_key,
                    "channels": list(modalities),
                    "fold": fold_number,
                    "training_seed": seed,
                    "eval_n": int(len(eval_frame)),
                    "baseline": {
                        metric: baseline[metric]
                        for metric in METRIC_KEYS
                    },
                    "perturbed": {
                        channel: {
                            metric: perturbed[channel][metric]
                            for metric in METRIC_KEYS
                        }
                        for channel in modalities
                    },
                    "perturbation_deltas": {
                        channel: {
                            metric: perturbed[channel][metric] - baseline[metric]
                            for metric in METRIC_KEYS
                        }
                        for channel in modalities
                    },
                    "deeplift_share": {
                        channel: shares[channel]["mean_abs"] / total_share
                        for channel in modalities
                    },
                    "deeplift_center_share": {
                        channel: shares[channel]["center_share"]
                        for channel in modalities
                    },
                }
                if seed == EXAMPLE_SEED and fold_number == EXAMPLE_FOLD:
                    out = figures_dir / f"deeplift_examples_{scheme_key}.png"
                    render_examples(model, dataset, device, modalities, out)
                    log(f"rendered DeepLIFT examples -> {out.name}")

                fold_metrics[key] = run_record
                existing.write_text(
                    json.dumps(fold_metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                records.append(run_record)
                log(
                    f"{SCHEME_DISPLAY[scheme_key]} fold {fold_number} seed {seed} | "
                    f"base f1={baseline['macro_f1']:.4f} | "
                    + " ".join(
                        f"d({c})F1={run_record['perturbation_deltas'][c]['macro_f1']:+.3f}"
                        for c in modalities
                    )
                )
                del model
                torch.cuda.empty_cache()

    # ----- aggregation -----
    def stats(values):
        array = np.array(values, dtype=float)
        return {"mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0}

    summary: dict[str, dict] = {}
    all_metrics = METRIC_KEYS
    for scheme_key, modalities in SCHEMES:
        scheme_records = [r for r in records if r["scheme_key"] == scheme_key]
        entry = {
            "n_runs": len(scheme_records),
            "baseline": {
                metric: stats([r["baseline"][metric] for r in scheme_records])
                for metric in all_metrics
            },
            "perturbed": {
                channel: {
                    metric: stats(
                        [r["perturbed"][channel][metric] for r in scheme_records]
                    )
                    for metric in all_metrics
                }
                for channel in modalities
            },
            "perturbation_delta": {
                channel: {
                    metric: stats([r["perturbation_deltas"][channel][metric] for r in scheme_records])
                    for metric in all_metrics
                }
                for channel in modalities
            },
            "deeplift_share": {
                channel: stats([r["deeplift_share"][channel] for r in scheme_records])
                for channel in modalities
            },
            "deeplift_center_share": {
                channel: stats([r["deeplift_center_share"][channel] for r in scheme_records])
                for channel in modalities
            },
        }
        summary[scheme_key] = entry

    notes = [
        "Representative schemes were fixed by explicit user confirmation "
        "(2026-09-15) before any Step 04 analysis ran: M+MB+MR, M+MR, and all "
        "five channels. The selection basis (Step 03_I ranking) and the "
        "confirmation are recorded in config_step_10_I.json.",
        "No checkpoint selection: all 15 models per scheme (3 training seeds x "
        "5 folds) were analysed, each strictly on its own evaluation fold, and "
        "aggregated as Mean +/- Std across the 15 runs.",
        "Perturbation method (identical for every scheme): one channel at a "
        "time replaced by black pixels - the same fill used for black center "
        "padding, keeping the perturbation in-distribution. No retraining, no "
        "split changes.",
        "DeepLIFT: captum implementation, reference = all-black image through "
        "the identical preprocessing, attribution target = each sample's "
        "predicted class; channel importance = per-sample share of summed "
        "absolute attribution, averaged per fold-model and aggregated over 15 "
        "runs.",
        "Visual examples follow a pre-specified rule (seed 42 / fold 1 model, "
        "first two correctly classified eval samples per class in dataset "
        "order); they are illustrative, while quantitative conclusions use the "
        "full 15-run aggregation.",
        "SUPPLEMENT (user-requested, 2026-09-15, after comparing with the "
        "previous rosacea_step4 report format): added (a) quantitative "
        "center-vs-periphery spatial attribution split (center box = middle "
        "50% of each side, 192x192 of 384), (b) post-ablation absolute "
        "performance tables, and (c) the "
        "data-integrity section in this report. Analysis-only changes; no "
        "retraining, no split changes, same 15 models per scheme.",
    ]
    payload = {
        "config": config,
        "integrity": integrity,
        "records": records,
        "summary": summary,
        "notes": notes,
    }
    metrics_json = guard_path(STEP_DIR / f"metrics_{STEP_NAME}.json")
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for record in records:
        row = {
            "record_type": "run",
            "scheme": record["scheme"],
            "fold": record["fold"],
            "training_seed": record["training_seed"],
            "baseline_accuracy": record["baseline"]["accuracy"],
            "baseline_macro_f1": record["baseline"]["macro_f1"],
            "baseline_mae": record["baseline"]["mae"],
        }
        for channel in record["channels"]:
            for metric in METRIC_KEYS:
                row[f"delta_{metric}_{channel}"] = record["perturbation_deltas"][channel][metric]
                row[f"after_{metric}_{channel}"] = record["perturbed"][channel][metric]
            row[f"deeplift_share_{channel}"] = record["deeplift_share"][channel]
            row[f"deeplift_center_share_{channel}"] = record["deeplift_center_share"][channel]
        rows.append(row)
    pd.DataFrame(rows).to_csv(guard_path(STEP_DIR / f"metrics_{STEP_NAME}.csv"), index=False)

    # ----- figures -----
    scheme_colors = {"M_MB_MR": "#4C72B0", "M_MR": "#DD8452", "ALL5": "#55A868"}
    for metric, label in (("macro_f1", "ΔMacro-F1 (drop)"), ("accuracy", "ΔAccuracy (drop)"), ("mae", "ΔMAE (change)")):
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(ALL_CHANNELS))
        width = 0.25
        for offset, (scheme_key, modalities) in enumerate(SCHEMES):
            means, stds, xs = [], [], []
            for position, channel in enumerate(ALL_CHANNELS):
                if channel in modalities:
                    values = summary[scheme_key]["perturbation_delta"][channel][metric]
                    means.append(values["mean"])
                    stds.append(values["std"])
                    xs.append(x[position] + (offset - 1) * width)
            ax.bar(xs, means, width, yerr=stds, label=SCHEME_DISPLAY[scheme_key],
                   color=scheme_colors[scheme_key], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_CHANNELS)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(label)
        ax.set_title(
            f"Step 10_I 5-class soft | channel perturbation impact | {label} "
            "(mean ± std over 15 runs per scheme)"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(figures_dir / f"perturbation_{metric}.png", dpi=160)
        fig.savefig(figures_dir / f"perturbation_{metric}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, (scheme_key, modalities) in enumerate(SCHEMES):
        xs, means, stds = [], [], []
        for position, channel in enumerate(ALL_CHANNELS):
            if channel in modalities:
                share = summary[scheme_key]["deeplift_share"][channel]
                means.append(share["mean"] * 100)
                stds.append(share["std"] * 100)
                xs.append(x[position] + (offset - 1) * width)
        ax.bar(xs, means, width, yerr=stds, label=SCHEME_DISPLAY[scheme_key],
               color=scheme_colors[scheme_key], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_CHANNELS)
    ax.set_ylabel("DeepLIFT |attribution| share (%)")
    ax.set_title(
        "Step 10_I 5-class soft | per-channel DeepLIFT importance share "
        "(mean ± std over 15 runs per scheme)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "deeplift_share.png", dpi=160)
    fig.savefig(figures_dir / "deeplift_share.pdf")
    plt.close(fig)

    # Center-vs-periphery spatial attribution split per channel and scheme.
    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, (scheme_key, modalities) in enumerate(SCHEMES):
        xs, means, stds = [], [], []
        for position, channel in enumerate(ALL_CHANNELS):
            if channel in modalities:
                center = summary[scheme_key]["deeplift_center_share"][channel]
                means.append(center["mean"] * 100)
                stds.append(center["std"] * 100)
                xs.append(x[position] + (offset - 1) * width)
        ax.bar(xs, means, width, yerr=stds, label=SCHEME_DISPLAY[scheme_key],
               color=scheme_colors[scheme_key], capsize=3)
    ax.axhline(25.0, color="gray", linestyle=":", linewidth=1.0)
    ax.text(len(ALL_CHANNELS) - 0.45, 25.8,
            "25% = center-box area proportion", fontsize=8, ha="right",
            color="gray",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_CHANNELS)
    ax.set_ylabel("Share of channel |attribution| in center 192x192 (%)")
    ax.set_title(
        "Step 10_I 5-class soft | DeepLIFT center-vs-periphery split "
        "(center box = middle 50% of each side; mean ± std over 15 runs)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "deeplift_center_share.png", dpi=160)
    fig.savefig(figures_dir / "deeplift_center_share.pdf")
    plt.close(fig)

    figure_files = sorted(figures_dir.glob("*.png"))

    # ----- HTML report -----
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

    baseline_rows = []
    for scheme_key, modalities in SCHEMES:
        base = summary[scheme_key]["baseline"]
        baseline_rows.append(
            f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td>"
            + "".join(
                f"<td>{base[m]['mean']:.4f} ± {base[m]['std']:.4f}</td>"
                for m in ("accuracy", "macro_f1", "mae")
            )
            + "</tr>"
        )
    sections.append(
        "<h2>未扰动基线性能（15 个模型各自在其评估折，Mean ± Std）</h2>"
        "<table><tr><th>方案</th><th>Accuracy</th><th>Macro-F1</th><th>MAE</th></tr>"
        f"{''.join(baseline_rows)}</table>"
    )

    after_rows = []
    for scheme_key, modalities in SCHEMES:
        for channel in modalities:
            after = summary[scheme_key]["perturbed"][channel]
            after_rows.append(
                f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td><td>去掉 {channel}</td>"
                + "".join(
                    f"<td>{after[m]['mean']:.4f} ± {after[m]['std']:.4f}</td>"
                    for m in ("accuracy", "macro_f1", "mae")
                )
                + "</tr>"
            )
    sections.append(
        "<h2>通道扰动后的绝对性能（Mean ± Std over 15 runs）</h2>"
        "<table><tr><th>方案</th><th>扰动</th><th>Accuracy</th><th>Macro-F1</th><th>MAE</th></tr>"
        f"{''.join(after_rows)}</table>"
    )

    summary_rows = []
    for scheme_key, modalities in SCHEMES:
        entry = summary[scheme_key]
        cells = [f"<td>{SCHEME_DISPLAY[scheme_key]}</td>"]
        for channel in modalities:
            delta = entry["perturbation_delta"][channel]["macro_f1"]
            share = entry["deeplift_share"][channel]
            cells.append(
                f"<td>{channel}: ΔF1 {delta['mean']:+.4f}±{delta['std']:.4f}"
                f"<br>DeepLIFT {share['mean'] * 100:.1f}%±{share['std'] * 100:.1f}%</td>"
            )
        summary_rows.append("".join(cells))
    sections.append(
        "<h2>通道重要性汇总（每格 = 15 次 run 的 Mean ± Std）</h2>"
        "<table><tr><th>方案</th>" + "".join(
            f"<th>通道 {i + 1}</th>" for i in range(max(len(m) for _, m in SCHEMES))
        ) + f"{''.join(f'<tr>{row}</tr>' for row in summary_rows)}</table>"
    )

    spatial_rows = []
    for scheme_key, modalities in SCHEMES:
        for channel in modalities:
            center = summary[scheme_key]["deeplift_center_share"][channel]
            spatial_rows.append(
                f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td><td>{channel}</td>"
                f"<td>{center['mean'] * 100:.1f}% ± {center['std'] * 100:.1f}%</td>"
                f"<td>{(1 - center['mean']) * 100:.1f}%</td></tr>"
            )
    sections.append(
        "<h2>DeepLIFT 空间区域分析（中心 vs 外周）</h2>"
        "<p>中心区域定义：图像中央 50% × 50% 的正方形（384 图上的 192×192，"
        "占面积 25%）。若某通道的中心占比显著高于 25%，说明该通道的注意力向病灶中心集中。</p>"
        "<table><tr><th>方案</th><th>通道</th><th>中心占比（Mean ± Std）</th><th>外周占比</th></tr>"
        f"{''.join(spatial_rows)}</table>"
    )

    try:
        hard = json.loads(
            (STEP_07_DIR / "metrics_step_07_I.json").read_text(encoding="utf-8")
        )["summary"]
        compare_rows = []
        for scheme_key, modalities in SCHEMES:
            for channel in modalities:
                h = hard[scheme_key]
                s = summary[scheme_key]
                compare_rows.append(
                    f"<tr><td>{SCHEME_DISPLAY[scheme_key]}</td><td>{channel}</td>"
                    f"<td>{h['perturbation_delta'][channel]['macro_f1']['mean']:+.4f}</td>"
                    f"<td>{s['perturbation_delta'][channel]['macro_f1']['mean']:+.4f}</td>"
                    f"<td>{h['deeplift_share'][channel]['mean'] * 100:.1f}%</td>"
                    f"<td>{s['deeplift_share'][channel]['mean'] * 100:.1f}%</td>"
                    f"<td>{h['deeplift_center_share'][channel]['mean'] * 100:.1f}%</td>"
                    f"<td>{s['deeplift_center_share'][channel]['mean'] * 100:.1f}%</td>"
                    f"</tr>"
                )
        sections.append(
            "<h2>Hard（Step 07）vs Soft（本 Step）通道依赖对比</h2>"
            "<table><tr><th>方案</th><th>通道</th><th>ΔF1 扰动 (Hard)</th>"
            "<th>ΔF1 扰动 (Soft)</th><th>DeepLIFT 占比 (Hard)</th>"
            "<th>DeepLIFT 占比 (Soft)</th><th>中心占比 (Hard)</th>"
            "<th>中心占比 (Soft)</th></tr>"
            f"{''.join(compare_rows)}</table>"
        )
    except Exception as exc:  # comparison is supplementary, never fatal
        sections.append(f"<h2>Hard vs Soft 对比</h2><p>对比表不可用：{exc}</p>")

    images = "".join(
        f"<h3>{p.name}</h3><img src='data:image/png;base64,"
        f"{base64.b64encode(p.read_bytes()).decode('ascii')}' alt='{p.name}'>"
        for p in figure_files
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
<h1>{STEP_NAME}：五分类 Soft Label DeepLIFT + 通道扰动（Patient Isolation = True）</h1>
<p>生成时间：{datetime.now(timezone.utc).isoformat()}</p>
{''.join(sections)}
</body></html>"""
    report_path = guard_path(STEP_DIR / f"report_{STEP_NAME}.html")
    report_path.write_text(html, encoding="utf-8")

    log(f"analysis complete: {len(records)} runs in {(time.time() - started) / 60:.1f} min")
    for scheme_key, modalities in SCHEMES:
        entry = summary[scheme_key]
        log(
            f"{SCHEME_DISPLAY[scheme_key]} | "
            + " | ".join(
                f"{c}: dF1={entry['perturbation_delta'][c]['macro_f1']['mean']:+.4f}"
                f" share={entry['deeplift_share'][c]['mean'] * 100:.1f}%"
                for c in modalities
            )
        )
    log(f"report: {report_path}")

    # Overwrite the report built above with the Chinese-notes version
    # (figures and CSV are regenerated identically).
    finalize_outputs(payload, figures_dir)


if __name__ == "__main__":
    main()
