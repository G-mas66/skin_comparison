echo '=== nproc / mem ==='
nproc
free -g | head -2
echo '=== train.log lines ==='
wc -l /root/autodl-tmp/skin_comparison/isolated/step_01_I/fold_01/seed_42/train.log 2>/dev/null
tail -3 /root/autodl-tmp/skin_comparison/isolated/step_01_I/fold_01/seed_42/train.log 2>/dev/null
echo '=== nohup tail ==='
tail -3 /root/autodl-tmp/skin_comparison/isolated/step_01_I/logs/nohup_full.out
echo '=== worker cpu now ==='
ps --ppid 5892 -o pid,stat,time,pcpu | head -12
echo '=== main cpu ==='
ps -p 5892 -o pid,stat,time,pcpu
