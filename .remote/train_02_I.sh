export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/skin_comparison/isolated/step_02_I
mkdir -p logs
DATA=/root/autodl-tmp/data_all_384
echo "=== verify dataset (no re-upload) ==="
ls "$DATA" | wc -l
echo '=== smoke test (channel MUV) ==='
python step_02_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' --smoke 2>&1 | tail -8
smoke_code=${PIPESTATUS[0]}
if [ "$smoke_code" -ne 0 ]; then
  echo "SMOKE TEST FAILED (exit $smoke_code) - NOT launching full training"
  exit "$smoke_code"
fi
echo '=== launching full step_02_I (4 channels x 3 seeds x 5 folds = 60 runs) ==='
nohup python step_02_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' > logs/nohup_full.out 2>&1 &
echo "full training launched with pid $!"
sleep 5
tail -5 logs/nohup_full.out
