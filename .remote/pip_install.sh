export PATH=/root/miniconda3/bin:$PATH
pip install -q pandas scikit-learn 2>&1 | tail -5
python -c "import pandas, sklearn; print('pandas', pandas.__version__, '| sklearn', sklearn.__version__)"
mkdir -p /root/autodl-tmp/skin_comparison/isolated/step_01_I
ls /root/autodl-tmp/data_all 2>/dev/null | head -3
