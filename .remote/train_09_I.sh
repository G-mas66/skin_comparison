export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/skin_comparison/isolated/step_09_I
mkdir -p logs
ulimit -n 65535
DATA=/root/autodl-tmp/data_all_384
echo '=== verify step_08 soft metrics present ==='
ls -la /root/autodl-tmp/skin_comparison/isolated/step_08_I/metrics_step_08_I.json
echo '=== smoke test (ALL5, soft label) ==='
python step_09_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' --smoke 2>&1 | tail -5
smoke_code=${PIPESTATUS[0]}
if [ "$smoke_code" -ne 0 ]; then
  echo "SMOKE TEST FAILED (exit $smoke_code) - NOT launching full training"
  exit "$smoke_code"
fi
echo '=== launching full step_09_I (5 schemes x 3 seeds x 5 folds = 75 runs, soft label) ==='
nohup python step_09_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' > logs/nohup_full.out 2>&1 &
echo "full training launched with pid $!"
sleep 5
tail -3 logs/nohup_full.out
