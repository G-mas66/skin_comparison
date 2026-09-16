export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/skin_comparison/isolated/step_05_I
mkdir -p logs
DATA=/root/autodl-tmp/data_all_384
echo "=== verify dataset (no re-upload) ==="
ls "$DATA" | wc -l
echo '=== smoke test (channel M, 5-class) ==='
python step_05_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' --smoke 2>&1 | tail -6
smoke_code=${PIPESTATUS[0]}
if [ "$smoke_code" -ne 0 ]; then
  echo "SMOKE TEST FAILED (exit $smoke_code) - NOT launching full training"
  exit "$smoke_code"
fi
echo '=== launching full step_05_I (5 channels x 3 seeds x 5 folds = 75 runs) ==='
nohup python step_05_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' > logs/nohup_full.out 2>&1 &
echo "full training launched with pid $!"
sleep 5
tail -3 logs/nohup_full.out
