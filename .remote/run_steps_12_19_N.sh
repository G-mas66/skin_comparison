#!/bin/bash
# Sequential runner for the MR ROI non-isolated route (Steps 12_N-19_N).
# Per PROTOCOL.md: smoke test first, then the full 15-run benchmark per step.
export PATH=/root/miniconda3/bin:$PATH
REPO=/root/autodl-tmp/skin_comparison
LOGDIR=$REPO/non_isolated/runner_logs
mkdir -p "$LOGDIR"

for step in step_12_N step_13_N step_14_N step_15_N step_16_N step_17_N step_18_N step_19_N; do
  cd "$REPO/non_isolated/$step" || exit 1
  echo "=== [$(date '+%F %T')] $step smoke test ==="
  if ! python "$step.py" --smoke > "$LOGDIR/${step}_smoke.log" 2>&1; then
    echo "SMOKE FAILED for $step - stopping" | tee -a "$LOGDIR/runner_status.log"
    tail -30 "$LOGDIR/${step}_smoke.log"
    exit 1
  fi
  tail -2 "$LOGDIR/${step}_smoke.log"
  echo "=== [$(date '+%F %T')] $step full run (3 seeds x 5 folds) ==="
  python "$step.py" > "$LOGDIR/${step}_full.log" 2>&1
  code=$?
  if [ $code -ne 0 ]; then
    echo "FULL RUN FAILED for $step (exit $code) - stopping" | tee -a "$LOGDIR/runner_status.log"
    tail -40 "$LOGDIR/${step}_full.log"
    exit 1
  fi
  grep -E '^\[' "$LOGDIR/${step}_full.log" | tail -4
  echo "=== [$(date '+%F %T')] $step complete ===" | tee -a "$LOGDIR/runner_status.log"
done
echo "ALL_ROI_N_STEPS_COMPLETE $(date '+%F %T')" | tee -a "$LOGDIR/runner_status.log"
