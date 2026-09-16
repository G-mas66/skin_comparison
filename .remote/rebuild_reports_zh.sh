export PATH=/root/miniconda3/bin:$PATH
DATA=/root/autodl-tmp/data_all_384
for step in step_01_I step_02_I step_03_I; do
  echo "=== rebuilding $step (Chinese notes) ==="
  cd /root/autodl-tmp/skin_comparison/isolated/$step
  python ${step}.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' --report-only 2>&1 | tail -1
done
echo "=== rebuilding step_04_I (Chinese notes) ==="
cd /root/autodl-tmp/skin_comparison/isolated/step_04_I
python step_04_I.py --report-only 2>&1 | tail -1
