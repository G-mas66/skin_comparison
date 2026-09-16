export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/skin_comparison/isolated/step_01_I
mkdir -p logs
DATA=/root/autodl-tmp/data_all_384
echo "=== precheck: dataset counts in $DATA ==="
ls "$DATA" | wc -l
ls "$DATA" | grep -c '^M[0-9]'
echo '=== smoke test ==='
python step_01_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' --smoke 2>&1 | tail -20
smoke_code=${PIPESTATUS[0]}
if [ "$smoke_code" -ne 0 ]; then
  echo "SMOKE TEST FAILED (exit $smoke_code) - NOT launching full training"
  exit "$smoke_code"
fi
echo '=== launching full 15-run training in background ==='
nohup python step_01_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' > logs/nohup_full.out 2>&1 &
echo "full training launched with pid $!"
sleep 5
tail -5 logs/nohup_full.out
