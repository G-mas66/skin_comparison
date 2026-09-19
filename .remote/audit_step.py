"""Post-step auditor for the MR ROI N-route steps (run after each step).

Validates that a completed step satisfies PROTOCOL.md requirements before the
next step is allowed to start: 15 seed x fold records with correct counts,
finite in-range metrics, fold structure equal to the step_05_N reference,
all mandated artifacts on disk (config, integrity, CSV/JSON, seed summaries,
final summary, figures, HTML report, 15 last.pth checkpoints).

Usage: python audit_step.py step_12_N
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path("/root/autodl-tmp/skin_comparison")
DATA = Path("/root/autodl-tmp/data_all_384")


def main() -> None:
    step_id = sys.argv[1]
    step_dir = REPO / "non_isolated" / step_id
    problems: list[str] = []

    required = [
        f"{step_id}.py",
        f"config_{step_id}.json",
        f"integrity_report.json",
        f"metrics_{step_id}.csv",
        f"metrics_{step_id}.json",
        f"seed_summaries_{step_id}.json",
        f"final_summary_{step_id}.json",
        f"report_{step_id}.html",
        "figures",
        "logs",
    ]
    for name in required:
        if not (step_dir / name).exists():
            problems.append(f"missing artifact: {name}")

    try:
        with open(step_dir / f"metrics_{step_id}.json", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        print(f"AUDIT FAIL {step_id}: cannot read metrics json ({exc})")
        raise SystemExit(1)

    records = payload["fold_results"]
    if len(records) != 15:
        problems.append(f"expected 15 fold records, found {len(records)}")
    expected_pairs = {(seed, fold) for seed in (42, 3407, 2026) for fold in range(1, 6)}
    got_pairs = {(r["training_seed"], r["fold"]) for r in records}
    if got_pairs != expected_pairs:
        problems.append(f"seed/fold coverage mismatch: missing {sorted(expected_pairs - got_pairs)[:4]}")

    with open(REPO / "non_isolated" / "baseline_refs" / "metrics_step_05_N.json", encoding="utf-8") as handle:
        reference = json.load(handle)["integrity"]["modality_stats"]["MR"]
    for record in records:
        want = reference[record["fold"] - 1]
        if record["evaluation_count"] != want["evaluation_count"]:
            problems.append(f"seed {record['training_seed']} fold {record['fold']}: eval count mismatch")
        if {str(k): v for k, v in record["evaluation_class_counts"].items()} != want["evaluation_class_counts"]:
            problems.append(f"seed {record['training_seed']} fold {record['fold']}: eval class counts mismatch")
        for key in ("accuracy", "macro_f1", "mae"):
            value = record[key]
            if not math.isfinite(value):
                problems.append(f"non-finite {key} in seed {record['training_seed']} fold {record['fold']}")
        if not (0.0 <= record["accuracy"] <= 1.0 and 0.0 <= record["macro_f1"] <= 1.0):
            problems.append(f"metric out of range in seed {record['training_seed']} fold {record['fold']}")
        checkpoint = step_dir / f"fold_{record['fold']:02d}" / f"seed_{record['training_seed']}" / "checkpoints" / "last.pth"
        if not checkpoint.exists():
            problems.append(f"missing checkpoint: {checkpoint.relative_to(step_dir)}")

    figure_names = [
        f"benchmark_accuracy.png", f"benchmark_macro_f1.png", f"benchmark_mae.png",
        f"benchmark_seed_detail.png", f"benchmark_per_class.png", f"own_confusion.png",
    ]
    for name in figure_names:
        if not (step_dir / "figures" / name).exists():
            problems.append(f"missing figure: figures/{name}")

    html = step_dir / f"report_{step_id}.html"
    if html.exists() and len(html.read_bytes()) < 200_000:
        problems.append("HTML report suspiciously small (figures likely missing)")

    final = payload["final_summaries"][0]["metrics"]
    summary_line = (
        f"final: acc={final['accuracy']['mean']:.4f}±{final['accuracy']['std']:.4f} "
        f"macro_f1={final['macro_f1']['mean']:.4f}±{final['macro_f1']['std']:.4f} "
        f"mae={final['mae']['mean']:.4f}±{final['mae']['std']:.4f}"
    )

    if problems:
        print(f"AUDIT FAIL {step_id}:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print(f"AUDIT PASS {step_id}: {summary_line}")


if __name__ == "__main__":
    main()
