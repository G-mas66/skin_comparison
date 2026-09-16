export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/skin_comparison/isolated/step_06_I
mkdir -p logs
DATA=/root/autodl-tmp/data_all_384
echo "=== verify step_05_I reference metrics present ==="
ls -la /root/autodl-tmp/skin_comparison/isolated/step_05_I/metrics_step_05_I.json
echo '=== smoke test (ALL5, 15-channel, 5-class) ==='
python step_06_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' --smoke 2>&1 | tail -6
smoke_code=${PIPESTATUS[0]}
if [ "$smoke_code" -ne 0 ]; then
  echo "SMOKE TEST FAILED (exit $smoke_code) - NOT launching full training"
  exit "$smoke_code"
fi
echo '=== launching full step_06_I (5 schemes x 3 seeds x 5 folds = 75 runs) ==='
nohup python step_06_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' > logs/nohup_full.out 2>&1 &
echo "full training launched with pid $!"
sleep 5
tail -3 logs/nohup_full.out
