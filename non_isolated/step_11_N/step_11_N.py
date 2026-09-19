"""Step 11_N: MR ROI data preparation (Patient Isolation = False route).

No model training. Following TRAINING_PLAN.md ("Step 11_N: MR ROI 数据准备"):

- JSON-MR mapping check via the JSON-internal imagePath (duplicate annotation
  files such as MR0182_1.json are mapped to the same image and recorded);
- polygon coordinate conversion 3448x4600 -> 384x384 (x*288/3448+48, y*384/4600);
- binary ROI masks (pixel-center rasterization, union of all `red` polygons)
  cached to <roi-cache>/masks/MRxxxx.png;
- joint bounding boxes (+ the 20 % context boxes derived later on the fly);
- per-fold Training Mean Fill images from the training folds only;
- ROI statistics (polygon counts, area ratios, box sizes, upsampling factors,
  stratified by label and by fold) with anonymized figures;
- local-only visualization QA saved next to the data (never in results / Git);
- integrity checks incl. the fixed N-route five-fold structure against the
  step_05_N reference records.

Outputs live in non_isolated/step_11_N/ (audit + figures + report); caches
live under /root/autodl-tmp/data_all_384/roi_cache/.
"""

from __future__ import annotations

import json
import random as py_random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import EXCLUDED_IDS, INPUT_SIZE, SEED, iter_folds, make_five_folds
from roi_common import (
    CONTEXT_RATIO,
    build_roi_record,
    compute_mean_fill,
    context_bbox,
    rasterize_mask,
)
from train_common import load_reference_records

MANIFEST = "/root/autodl-tmp/data_all_384/manifest.csv"
DATA_ROOT = Path("/root/autodl-tmp/data_all_384")
CACHE_DIR = DATA_ROOT / "roi_cache"
MASK_DIR = CACHE_DIR / "masks"
QA_DIR = DATA_ROOT / "roi_qa_step11_N"


def main() -> None:
    (STEP_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (STEP_DIR / "logs").mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    folded = make_five_folds(MANIFEST, patient_isolation=False, seed=SEED)

    # ---- JSON <-> MR mapping ------------------------------------------------ #
    json_paths = sorted(DATA_ROOT.glob("MR*.json"))
    mapping_rows = []
    duplicate_files = []
    for path in json_paths:
        with open(path, encoding="utf-8") as handle:
            header = json.load(handle)
        image_path_name = str(header.get("imagePath", ""))
        sample_id = path.stem
        for suffix in ("_1", "_2"):
            if sample_id.endswith(suffix):
                sample_id = sample_id[: -len(suffix)]
        png_path = DATA_ROOT / f"{sample_id}.png"
        mapping_rows.append({
            "json_file": path.name,
            "sample_id": sample_id,
            "imagePath": image_path_name,
            "png_exists": png_path.exists(),
            "is_duplicate_name": path.name != f"{sample_id}.json",
            "n_shapes_total": len(header.get("shapes", [])),
            "n_red_shapes": sum(
                1 for s in header.get("shapes", []) if str(s.get("label", "")).strip().lower() == "red"
            ),
        })
        if path.name != f"{sample_id}.json":
            duplicate_files.append(path.name)
    mapping = pd.DataFrame(mapping_rows)
    # JSONs of globally excluded IDs have no PNG by design (69, 296, 769, 770);
    # only non-excluded JSONs without a PNG are real mapping errors.
    def _capture_id(sample: str) -> int:
        return int(sample[2:])

    unexpected_missing = [
        row["json_file"]
        for _, row in mapping.iterrows()
        if not row["png_exists"] and _capture_id(row["sample_id"]) not in EXCLUDED_IDS
    ]
    if unexpected_missing:
        errors.append(f"JSON without matching MR PNG (non-excluded): {unexpected_missing}")

    manifest_ids = set(folded["sample_id"].astype(str))
    mapped_ids = set(mapping.loc[mapping["png_exists"], "sample_id"])
    if manifest_ids != mapped_ids:
        errors.append(
            f"mapping/manifest mismatch: no_json={sorted(manifest_ids - mapped_ids)[:5]} "
            f"no_manifest={sorted(mapped_ids - manifest_ids)[:5]}"
        )
    excluded_json = sorted(
        int(row["sample_id"][2:]) for _, row in mapping.iterrows()
        if int(row["sample_id"][2:]) in EXCLUDED_IDS
    )

    # ---- ROI records: conversion, masks, bounding boxes --------------------- #
    # One row per image: duplicate annotation files (e.g. MR0182_1.json) map to
    # the same sample; keep the canonical "<sample_id>.json" annotation.
    mapping = mapping.sort_values("is_duplicate_name").drop_duplicates("sample_id", keep="first")
    roi_rows = []
    for _, row in mapping.iterrows():
        sample_id = row["sample_id"]
        if not row["png_exists"] or sample_id not in manifest_ids:
            continue
        record = build_roi_record(DATA_ROOT / row["json_file"], DATA_ROOT / f"{sample_id}.png")
        mask = rasterize_mask(record.polygons)
        mask_area = int(mask.sum())
        from PIL import Image

        Image.fromarray((mask.astype(np.uint8)) * 255).save(MASK_DIR / f"{sample_id}.png")
        x0, y0, x1, y1 = record.bbox
        box_w, box_h = x1 - x0, y1 - y0
        cx0, cy0, cx1, cy1 = context_bbox(record.bbox)
        label = int(folded.loc[folded["sample_id"] == sample_id, "label"].iloc[0])
        fold = int(folded.loc[folded["sample_id"] == sample_id, "fold"].iloc[0])
        roi_rows.append({
            "sample_id": sample_id,
            "label": label,
            "fold": fold + 1,
            "n_red_polygons": record.n_red_polygons,
            "out_of_range_vertices": record.out_of_range_vertices,
            "mask_area": mask_area,
            "area_ratio": mask_area / (INPUT_SIZE * INPUT_SIZE),
            "bbox_x0": x0, "bbox_y0": y0, "bbox_x1": x1, "bbox_y1": y1,
            "bbox_width": box_w, "bbox_height": box_h,
            "bbox_max_side": max(box_w, box_h),
            "upsample_factor": INPUT_SIZE / max(box_w, box_h),
            "context_box_width": cx1 - cx0, "context_box_height": cy1 - cy0,
        })
    roi = pd.DataFrame(roi_rows)
    if (roi["mask_area"] <= 0).any():
        errors.append(f"all-zero masks: {roi.loc[roi['mask_area'] <= 0, 'sample_id'].tolist()[:5]}")
    if (roi["mask_area"] >= INPUT_SIZE * INPUT_SIZE).any():
        errors.append("at least one mask covers the full image")
    if (roi["n_red_polygons"] < 1).any():
        errors.append("samples without any red polygon")
    roi.to_csv(CACHE_DIR / "roi_manifest.csv", index=False)

    # ---- fold structure check against the step_05_N reference --------------- #
    reference = load_reference_records(REPO_ROOT / "non_isolated" / "baseline_refs" / "metrics_step_05_N.json")
    reference_structure = reference["integrity"]["modality_stats"]["MR"]
    fold_stats = []
    for (fold_number, train_frame, eval_frame), want in zip(
        iter_folds(folded, patient_isolation=False), reference_structure
    ):
        got_eval = {str(k): int(v) for k, v in eval_frame["label"].value_counts().sort_index().items()}
        if got_eval != want["evaluation_class_counts"] or len(eval_frame) != want["evaluation_count"]:
            errors.append(f"fold {fold_number} structure differs from step_05_N reference")
        fold_stats.append({
            "fold": fold_number,
            "train_count": int(len(train_frame)),
            "evaluation_count": int(len(eval_frame)),
            "train_class_counts": {str(k): int(v) for k, v in train_frame["label"].value_counts().sort_index().items()},
            "evaluation_class_counts": got_eval,
        })

    # ---- per-fold training Mean Fill (training folds only) ------------------ #
    mean_fill_stats = []
    for fold_number, train_frame, _ in iter_folds(folded, patient_isolation=False):
        paths = [DATA_ROOT / f"{sid}.png" for sid in train_frame["sample_id"]]
        mean_fill = compute_mean_fill(paths)
        np.save(CACHE_DIR / f"mean_fill_fold_{fold_number:02d}.npy", mean_fill)
        mean_fill_stats.append({
            "fold": fold_number,
            "n_training_images": len(paths),
            "channel_mean_0-255": [float(mean_fill[c].mean()) for c in range(3)],
            "channel_std_0-255": [float(mean_fill[c].std()) for c in range(3)],
        })

    # ---- statistics figures (anonymized) ------------------------------------ #
    _save_hist(roi, "n_red_polygons", "red polygons per image", STEP_DIR / "figures" / "polygon_counts")
    _save_hist(roi, "area_ratio", "ROI area / image area", STEP_DIR / "figures" / "area_ratio")
    _save_hist(roi, "bbox_max_side", "joint bbox max side (px)", STEP_DIR / "figures" / "bbox_size")
    _save_hist(roi, "upsample_factor", "upsampling factor (384 / bbox max side)", STEP_DIR / "figures" / "upsample_factor")
    _save_label_stratified(roi, STEP_DIR / "figures" / "area_ratio_by_label")
    _save_label_stratified(roi, STEP_DIR / "figures" / "bbox_by_label", value="bbox_max_side")
    _save_fold_stats(roi, STEP_DIR / "figures" / "roi_stats_by_fold")

    # ---- local visualization QA (data dir only, never results / Git) -------- #
    rng = py_random.Random(42)
    qa_ids = rng.sample(sorted(roi["sample_id"]), 8)
    _render_qa(qa_ids, roi)
    qa_note = f"8 random samples rendered to {QA_DIR} (overlay mask + bbox + context + crop) for local visual QA"

    # ---- audit summary ------------------------------------------------------ #
    audit = {
        "step": "step_11_N",
        "patient_isolation": False,
        "fold_seed": SEED,
        "json_files_total": len(json_paths),
        "duplicate_annotation_files": duplicate_files,
        "excluded_json_capture_ids": excluded_json,
        "roi_valid_samples": int(len(roi)),
        "manifest_samples": int(len(folded)),
        "mask_cache_dir": str(MASK_DIR),
        "mean_fill_files": [f"mean_fill_fold_{n:02d}.npy" for n in range(1, 6)],
        "mean_fill_stats": mean_fill_stats,
        "fold_structure_matches_step_05_N": not any("structure differs" in e for e in errors),
        "fold_stats": fold_stats,
        "roi_summary": {
            "polygon_count": {"min": int(roi.n_red_polygons.min()), "median": float(roi.n_red_polygons.median()),
                               "mean": float(roi.n_red_polygons.mean()), "max": int(roi.n_red_polygons.max()),
                               "total": int(roi.n_red_polygons.sum())},
            "area_ratio": {"min": float(roi.area_ratio.min()), "median": float(roi.area_ratio.median()),
                            "mean": float(roi.area_ratio.mean()), "max": float(roi.area_ratio.max())},
            "bbox_max_side": {"min": int(roi.bbox_max_side.min()), "median": float(roi.bbox_max_side.median()),
                               "max": int(roi.bbox_max_side.max())},
            "upsample_factor": {"median": float(roi.upsample_factor.median()), "max": float(roi.upsample_factor.max())},
            "out_of_range_vertices_total": int(roi.out_of_range_vertices.sum()),
        },
        "per_fold_roi_stats": [
            {
                "fold": int(fold),
                "n": int(len(group)),
                "polygon_mean": float(group.n_red_polygons.mean()),
                "area_ratio_median": float(group.area_ratio.median()),
                "area_ratio_min": float(group.area_ratio.min()),
                "area_ratio_max": float(group.area_ratio.max()),
            }
            for fold, group in roi.groupby("fold")
        ],
        "qa_note": qa_note,
        "errors": errors,
        "passed": not errors,
    }
    with open(STEP_DIR / "roi_audit_step_11_N.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=1, ensure_ascii=False)
    roi.to_csv(STEP_DIR / "roi_statistics_step_11_N.csv", index=False)
    mapping.to_csv(STEP_DIR / "json_mapping_step_11_N.csv", index=False)

    if errors:
        print("STEP 11_N AUDIT FAILED:", *errors, sep="\n  - ")
        raise SystemExit(1)
    print(
        f"[step_11_N] passed: {len(roi)} ROI samples, masks cached, 5 mean fills written, "
        f"fold structure matches step_05_N, QA rendered to {QA_DIR}"
    )


def _save_hist(frame: pd.DataFrame, column: str, label: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.4))
    axis.hist(frame[column], bins=40, color="#2A9D8F", edgecolor="#111827")
    axis.set_xlabel(label)
    axis.set_ylabel("images")
    axis.set_title(f"Distribution of {label} (n={len(frame)})")
    axis.grid(axis="y", color="#D8DEE8")
    axis.set_axisbelow(True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(f"{path}.{extension}", dpi=170)
    plt.close(fig)


def _save_label_stratified(frame: pd.DataFrame, path: Path, value: str = "area_ratio") -> None:
    fig, axis = plt.subplots(figsize=(7.6, 4.4))
    data = [frame.loc[frame["label"] == label, value] for label in range(5)]
    axis.boxplot(data, tick_labels=[f"Class {label}" for label in range(5)], showfliers=False)
    axis.set_xlabel("CEA severity class")
    axis.set_ylabel(value)
    axis.set_title(f"{value} stratified by severity class")
    axis.grid(axis="y", color="#D8DEE8")
    axis.set_axisbelow(True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(f"{path}.{extension}", dpi=170)
    plt.close(fig)


def _save_fold_stats(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    folds = sorted(frame["fold"].unique())
    axes[0].bar(folds, [frame.loc[frame["fold"] == f, "n_red_polygons"].mean() for f in folds],
                color="#264653", edgecolor="#111827")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("mean red polygons per image")
    axes[1].bar(folds, [frame.loc[frame["fold"] == f, "area_ratio"].median() for f in folds],
                color="#E76F51", edgecolor="#111827")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("median ROI area ratio")
    for axis in axes:
        axis.grid(axis="y", color="#D8DEE8")
        axis.set_axisbelow(True)
    fig.suptitle("ROI annotation statistics per fold")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(f"{path}.{extension}", dpi=170)
    plt.close(fig)


def _render_qa(sample_ids: list[str], roi: pd.DataFrame) -> None:
    from PIL import Image, ImageDraw

    for sample_id in sample_ids:
        record = build_roi_record(DATA_ROOT / f"{sample_id}.json", DATA_ROOT / f"{sample_id}.png")
        image = Image.open(DATA_ROOT / f"{sample_id}.png").convert("RGB")
        mask = Image.open(MASK_DIR / f"{sample_id}.png").convert("L")
        overlay = image.copy()
        red = Image.new("RGB", image.size, (231, 111, 81))
        overlay.paste(red, (0, 0), mask.point(lambda v: 90 if v > 0 else 0))
        draw = ImageDraw.Draw(overlay)
        x0, y0, x1, y1 = record.bbox
        cx0, cy0, cx1, cy1 = context_bbox(record.bbox)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(42, 70, 83), width=3)
        draw.rectangle([cx0, cy0, cx1 - 1, cy1 - 1], outline=(42, 157, 143), width=2)
        crop = image.crop((x0, y0, x1, y1)).resize((384, 384))
        canvas = Image.new("RGB", (384 * 3 + 40, 420), (247, 249, 252))
        canvas.paste(image, (10, 30))
        canvas.paste(overlay, (384 + 20, 30))
        canvas.paste(crop, (384 * 2 + 30, 30))
        canvas.save(QA_DIR / f"{sample_id}_qa.png")


if __name__ == "__main__":
    main()
