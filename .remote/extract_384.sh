set -e
cd /root/autodl-tmp/data_all_384
tar -xf /root/autodl-tmp/tar_384/chunk_0.tar
tar -xf /root/autodl-tmp/tar_384/chunk_1.tar
tar -xf /root/autodl-tmp/tar_384/chunk_2.tar
tar -xf /root/autodl-tmp/tar_384/chunk_3.tar
rm -f ./*.part*
echo "=== total entries ==="
ls | wc -l
echo "=== png count ==="
ls *.png | wc -l
echo "=== per modality ==="
echo -n "M: ";   ls | grep -c '^M[0-9][0-9]*\.png$'
echo -n "MB: ";  ls | grep -c '^MB[0-9][0-9]*\.png$'
echo -n "MP: ";  ls | grep -c '^MP[0-9][0-9]*\.png$'
echo -n "MR: ";  ls | grep -c '^MR[0-9][0-9]*\.png$'
echo -n "MUV: "; ls | grep -c '^MUV[0-9][0-9]*\.png$'
echo "=== manifest ==="
wc -l manifest.csv
rm -rf /root/autodl-tmp/tar_384
df -h /root/autodl-tmp | tail -1
