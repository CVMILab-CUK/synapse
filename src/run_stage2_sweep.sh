#!/bin/bash
# Chain: [1] fixed-pipeline (token0 IP-Adapter) re-eval  [2] CFG sweep  [3] retrieval-augmented
# [4] consolidate + backup.  All inference-only on the trained 50ep model (no retraining).
SRC=/home/work/models/synapse/src; cd "$SRC"
LOG=/tmp/stage2_sweep.log
CK=ckpt_dir/EEG_LDM_things2/checkpoint-120000
export SD21_MODEL_ID="philschmid/stable-diffusion-2-1"
export HF_HOME=$HOME/.cache/huggingface

echo "=== [1] FIXED pipeline (token0) re-eval @cfg7.5 (was mean-bug) ==="
python3 tools/eval_stage2_v2.py --ckpt $CK --mode standard --cfg 7.5 --steps 50 --n 200 --fid --tag fixed_c75 || echo "FAIL fixed"

echo "=== [2] CFG sweep (standard, fixed pipeline) ==="
for c in 3 5 9 12; do
  python3 tools/eval_stage2_v2.py --ckpt $CK --mode standard --cfg $c --steps 40 --n 200 --fid --tag cfg${c} || echo "FAIL cfg$c"
done

echo "=== [3] retrieval-augmented @cfg7.5 ==="
python3 tools/eval_stage2_v2.py --ckpt $CK --mode retrieval --cfg 7.5 --steps 50 --n 200 --fid --tag retrieval || echo "FAIL retrieval"

echo "=== [4] consolidate + backup ==="
TS=$(date +%Y%m%d_%H%M%S); BK="$HOME/models/synapse_backup/sweep_$TS"; mkdir -p "$BK/samples"
grep -aE "^RESULT" "$LOG" > "$BK/RESULTS.txt" 2>/dev/null
cp "$LOG" "$BK/sweep.log" 2>/dev/null
for d in output/v2_fixed_c75 output/v2_retrieval output/v2_cfg9; do
  cp "$d"/000[0-4].png "$BK/samples/$(basename $d)_"* 2>/dev/null
  for f in "$d"/000[0-4].png; do [ -f "$f" ] && cp "$f" "$BK/samples/$(basename $d)_$(basename $f)"; done
done
cp models/eeg_diffusion_pipeline.py tools/eval_stage2_v2.py "$BK/" 2>/dev/null
echo "===== SWEEP RESULTS ====="; cat "$BK/RESULTS.txt"
echo "=== SWEEP DONE -> $BK ==="
