"""Step step_16_N: MR ROI benchmark (Patient Isolation = False).

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
    "experiment_name": "step_16_N",
    "benchmark_type": "roi_input_region",
    "description": "step_16_N: mask input, Soft Label, five-class, patient isolation = false",
    "input_variant": "mask",
    "label_strategy": "Soft Label",
    "scheme_name": "ROI Mask (Soft Label)",
    "report_title": "ROI Mask (Soft Label) vs reference schemes (CEA five-class, N route)",
    "reference_schemes": [
        {"kind": "whole_image", "file": "metrics_step_08_N.json", "name": "Whole Image (Soft, step_08_N)"},
        {"kind": "sibling_step", "step_id": "step_12_N", "step_dir": "step_12_N", "name": "ROI Mask (Hard, step_12_N)"},
    ],
    "report_notes": [
        "Soft Label matrix fixed to [0.9 diag / 0.1 adjacent]; loss = soft-target cross entropy.",
        "Hard Label ROI Mask referenced from step_12_N; Whole Image Soft referenced from step_08_N MR records.",
        "Input construction identical to step_12_N.",
    ],
}

if __name__ == "__main__":
    run_step(CONFIG)
