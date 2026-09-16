export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/skin_comparison/isolated/step_03_I
echo '=== stopping phase-1 (3-scheme) process ==='
pkill -f 'python step_03_I.py' && sleep 3
ps aux | grep step_03_I | grep -v grep || echo 'no step_03_I processes remain'
echo '=== completed runs preserved in status.json ==='
cat logs/status.json
echo '=== relaunching with the 5-scheme script (single final report) ==='
DATA=/root/autodl-tmp/data_all_384
nohup python step_03_I.py --manifest "$DATA/manifest.csv" --data-root "$DATA" --cache-dir '' > logs/nohup_full.out 2>&1 &
echo "relaunched with pid $!"
sleep 15
tail -8 logs/nohup_full.out
