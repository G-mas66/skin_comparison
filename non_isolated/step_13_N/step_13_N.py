"""Step step_13_N: MR ROI benchmark (Patient Isolation = False).

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
    "experiment_name": "step_13_N",
    "benchmark_type": "roi_input_region",
    "description": "step_13_N: crop input, Hard Label, five-class, patient isolation = false",
    "input_variant": "crop",
    "label_strategy": "Hard Label",
    "scheme_name": "ROI Crop",
    "report_title": "ROI Crop vs reference schemes (CEA five-class, N route)",
    "reference_schemes": [
        {"kind": "whole_image", "file": "metrics_step_05_N.json", "name": "Whole Image (Hard, step_05_N)"},
    ],
    "report_notes": [
        "Whole Image baseline referenced from step_05_N MR records; not retrained.",
        "Input: joint bounding-box crop of all red polygons, restored to 384x384 via the distortion-free ResizePad; no mask, no mean fill.",
    ],
}

if __name__ == "__main__":
    run_step(CONFIG)
