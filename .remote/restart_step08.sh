echo '=== killing stuck process tree ==='
PARENT=$(ps aux | grep 'python step_08_I.py' | grep -v grep | head -1 | awk '{print $2}')
kill -9 "$PARENT" 2>/dev/null
sleep 5
pkill -9 -f 'python step_08_I.py' 2>/dev/null
sleep 3
ps aux | grep 'python step_08' | grep -v grep || echo 'all step_08 processes gone'
echo '=== gpu after kill ==='
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo '=== incident log ==='
printf 'incident 2026-09-16 07:0x-08:05: run soft_MB/fold_03/soft_MB_seed_2026 stalled at epoch ~14 for >60 min; process alive, GPU 98%%, workers idle (cudnn.benchmark/channels_last suspected spin). Fix: kill + resume from status.json (23/75 done).\n' >> /root/autodl-tmp/skin_comparison/isolated/step_08_I/logs/incident_stall.txt
echo '=== relaunch with resume ==='
cd /root/autodl-tmp/skin_comparison/isolated/step_08_I
ulimit -n 65535
(nohup /root/miniconda3/bin/python step_08_I.py --manifest /root/autodl-tmp/data_all_384/manifest.csv --data-root /root/autodl-tmp/data_all_384 --cache-dir '' > logs/nohup_full2.out 2>&1 &)
sleep 20
tail -3 logs/nohup_full2.out
