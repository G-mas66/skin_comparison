"""Step step_18_N: MR ROI benchmark (Patient Isolation = False).

Thin declaration file: all training / evaluation / aggregation / reporting
logic lives in the shared ``train_common.py``; ROI input construction lives
in ``roi_common.py``. Both are imported from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train_common import run_step

CONFIG = {
    "step_dir": str(STEP_DIR),
    "experiment_name": "step_18_N",
    "benchmark_type": "roi_input_region",
    "description": "step_18_N: context input, Soft Label, five-class, patient isolation = false",
    "input_variant": "context",
    "label_strategy": "Soft Label",
    "scheme_name": "ROI + 20% Context (Soft Label)",
    "report_title": "ROI + 20% Context (Soft Label) vs reference schemes (CEA five-class, N route)",
    "reference_schemes": [
        {"kind": "whole_image", "file": "metrics_step_08_N.json", "name": "Whole Image (Soft, step_08_N)"},
        {"kind": "sibling_step", "step_id": "step_14_N", "step_dir": "step_14_N", "name": "ROI + 20% Context (Hard, step_14_N)"},
    ],
    "report_notes": [
        "Soft Label matrix fixed to [0.9 diag / 0.1 adjacent]; loss = soft-target cross entropy.",
        "Hard Label ROI + 20% Context referenced from step_14_N; Whole Image Soft referenced from step_08_N MR records.",
        "Input construction identical to step_14_N (four sides +20% context).",
    ],
}

if __name__ == "__main__":
    run_step(CONFIG)
