#!/bin/bash
# THINGS-EEG2 EXPAND revision: learned 63->128 electrode projection (Linear) replaces zero-pad,
# removing the inter-channel attention "sink". Full chain:
# [1] Stage1 expand-AE -> [2] best encoder + retrieval eval -> [2b] attention map (sink check)
# -> [3] Stage2 LDM -> [4] gen-eval + FID -> [5] backup.  (no set -e: always reach backup)
SRC=/home/work/models/synapse/src
LOG=/tmp/things2_expand.log
cd "$SRC"
export SD21_MODEL_ID="philschmid/stable-diffusion-2-1"
export HF_HOME=$HOME/.cache/huggingface

AECFG=config/Train_AE_things2_expand.json
LDMCFG=config/Train_LDM_things2_expand.json
ENC=pretrain_models/EEGEncoder_things2_expand.pth

echo "=== [1/5] Stage1 expand-AE (real 63 -> learned 128, no zero-pad sink) ==="
rm -rf ckpt_dir/EEG_condition_things2_expand sharedfile
python3 train_ae.py -c $AECFG

echo "=== [2/5] select BEST encoder + retrieval eval ==="
python3 tools/select_best_encoder.py --ckpt_dir ckpt_dir/EEG_condition_things2_expand \
  --config $AECFG --out $ENC
python3 eval_things2_retrieval.py --encoder $ENC --config $AECFG --nway-repeats 200 || echo "retrieval eval failed (non-fatal)"

echo "=== [2b/5] attention map (sink check) ==="
python3 tools/dump_attention.py --config $AECFG --encoder $ENC --out output/attn_map_expand.png || echo "attn dump failed (non-fatal)"

echo "=== [3/5] Stage2 LDM (expand encoder, token0 -> IP-Adapter) ==="
rm -rf ckpt_dir/EEG_condition_things2_expand ckpt_dir/EEG_LDM_things2_expand sharedfile
python3 train_ldm.py -c $LDMCFG
echo "=== EXPAND STAGE2 DONE ==="

echo "=== [4/5] gen-eval (fixed pipeline, 2-way) + FID ==="
LDMCK=$(for d in ckpt_dir/EEG_LDM_things2_expand/checkpoint-*; do [ -d "$d/unet" ] && echo "$d"; done | sort -t- -k2 -n | tail -1)
echo "gen-eval on $LDMCK"
python3 tools/eval_stage2_v2.py --config $LDMCFG --ckpt "$LDMCK" --mode standard --cfg 7.5 --steps 50 --n 200 --fid --tag expand || echo "gen-eval failed (non-fatal)"
python3 tools/eval_fid_things2.py --config $LDMCFG --ckpt "$LDMCK" --n 2000 --steps 50 --cfg 7.5 || echo "FID2000 failed (non-fatal)"

echo "=== [5/5] backup ==="
TS=$(date +%Y%m%d_%H%M%S); BK="$HOME/models/synapse_backup/expand_$TS"; mkdir -p "$BK/samples"
grep -aE "IMAGE-space|TEXT-space|BEST ckpt|RESULT tag=expand|Stage-2 FID|saved attention" "$LOG" > "$BK/RESULTS.txt" 2>/dev/null
cp "$LOG" "$BK/expand.log" 2>/dev/null
cp $AECFG $LDMCFG "$BK/" 2>/dev/null
cp $ENC "$BK/" 2>/dev/null
cp output/attn_map_expand.png "$BK/" 2>/dev/null
cp models/eeg_AE.py "$BK/eeg_AE.py" 2>/dev/null
cp output/v2_expand/000[0-9].png "$BK/samples/" 2>/dev/null
du -sh "$BK" > "$BK/_size.txt" 2>/dev/null
echo "===== EXPAND RESULTS ====="; cat "$BK/RESULTS.txt"
echo "=== EXPAND BACKUP DONE -> $BK ==="
