"""Step step_14_N: MR ROI benchmark (Patient Isolation = False).

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
    "experiment_name": "step_14_N",
    "benchmark_type": "roi_input_region",
    "description": "step_14_N: context input, Hard Label, five-class, patient isolation = false",
    "input_variant": "context",
    "label_strategy": "Hard Label",
    "scheme_name": "ROI + 20% Context",
    "report_title": "ROI + 20% Context vs reference schemes (CEA five-class, N route)",
    "reference_schemes": [
        {"kind": "whole_image", "file": "metrics_step_05_N.json", "name": "Whole Image (Hard, step_05_N)"},
        {"kind": "sibling_step", "step_id": "step_13_N", "step_dir": "step_13_N", "name": "ROI Crop (Hard, step_13_N)"},
    ],
    "report_notes": [
        "Whole Image baseline referenced from step_05_N; ROI Crop referenced from step_13_N results.",
        "Context crop: bounding box expanded by 20% of its own width/height on all four sides, clamped to image bounds, then ResizePad to 384x384.",
    ],
}

if __name__ == "__main__":
    run_step(CONFIG)
