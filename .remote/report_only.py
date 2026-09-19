"""Rebuild figures + HTML report for a completed step without retraining.

Usage: python report_only.py step_12_N
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/root/autodl-tmp/skin_comparison")
sys.path.insert(0, str(REPO))

from train_common import aggregate_records, finalize_step  # noqa: E402


def main() -> None:
    step_id = sys.argv[1]
    step_dir = REPO / "non_isolated" / step_id
    config = json.loads((step_dir / f"config_{step_id}.json").read_text())
    config.setdefault("repo_root", str(REPO))
    payload = json.loads((step_dir / f"metrics_{step_id}.json").read_text())
    records = payload["fold_results"]
    aggregation = {
        "seed_summaries": payload["seed_summaries"],
        "final_summary": payload["final_summaries"][0],
    }
    finalize_step(
        {
            "full_config": config,
            "step_dir": step_dir,
            "records": records,
            "aggregation": aggregation,
            "integrity": payload["integrity"],
        }
    )


if __name__ == "__main__":
    main()
