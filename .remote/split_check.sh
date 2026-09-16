export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/skin_comparison/isolated/step_01_I
python - <<'PYEOF'
import sys
sys.path.insert(0, "/root/autodl-tmp/skin_comparison")
import step_01_I as step
print("import OK; STEP_DIR =", step.STEP_DIR)

from dataset import make_five_folds
folded = make_five_folds(
    "/root/autodl-tmp/data_all/manifest.csv", patient_isolation=True, seed=42
)
print("total samples:", len(folded))
print(folded["fold"].value_counts().sort_index().to_string())
print("binary label distribution per fold:")
for fold in range(5):
    tr = folded.loc[folded["fold"] != fold, "label"].map(step.binary_target)
    ev = folded.loc[folded["fold"] == fold, "label"].map(step.binary_target)
    print(f"  fold {fold+1}: train 0/1 = {(tr==0).sum()}/{(tr==1).sum()}, eval 0/1 = {(ev==0).sum()}/{(ev==1).sum()}, eval_n = {len(ev)}")
PYEOF
