echo '=== process ==='
ps aux | grep 'python step_08' | grep -v grep | head -3
echo '=== gpu ==='
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo '=== log tail ==='
tail -8 /root/autodl-tmp/skin_comparison/isolated/step_08_I/logs/nohup_full.out
echo '=== current run train.log ==='
tail -3 /root/autodl-tmp/skin_comparison/isolated/step_08_I/fold_03/soft_MB_seed_2026/train.log 2>/dev/null
echo '=== worker processes ==='
PARENT=$(ps aux | grep 'python step_08_I.py' | grep -v grep | head -1 | awk '{print $2}')
echo "parent pid: $PARENT"
ps --ppid "$PARENT" -o pid,stat,time 2>/dev/null | head -12
