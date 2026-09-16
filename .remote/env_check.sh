export PATH=/root/miniconda3/bin:$PATH
python --version
python - <<'PYEOF'
import torch
print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available(), "| cuda ver:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(64, 64, device="cuda")
    print("gpu kernel ok:", float((x @ x).sum()) != 0.0)
PYEOF
echo '=== key pip packages ==='
pip list 2>/dev/null | grep -iE '^(torch|torchvision|timm|pandas|numpy|pillow|scikit-learn|matplotlib) '
echo '=== conda envs ==='
conda env list 2>/dev/null
echo '=== disk ==='
df -h /root/autodl-tmp | tail -1
echo '=== search dataset ==='
ls /root/autodl-tmp/ 2>/dev/null
find /root/autodl-tmp /root -maxdepth 3 -iname "*.csv" 2>/dev/null | grep -v conda | head -20
find / -maxdepth 4 -iname "M1.jpg" -o -maxdepth 4 -iname "MB1.jpg" 2>/dev/null | head -5
