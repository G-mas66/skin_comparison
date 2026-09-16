export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
pip install -q py-spy 2>&1 | tail -1
echo '=== py-spy dump main (5892) ==='
py-spy dump --pid 5892 2>&1 | head -50
echo '=== child processes ==='
ps --ppid 5892 -o pid,stat,time,cmd | head -15
echo '=== proc status ==='
cat /proc/5892/status | grep -E 'Threads|State'
