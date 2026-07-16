#!/bin/bash
SRC=/home/work/models/synapse/src; LOG=/tmp/aug2.log; cd "$SRC"
export HF_HOME=$HOME/.cache/huggingface
CFG=config/Train_AE_things2_aug2.json; ENC=pretrain_models/EEGEncoder_things2_aug2.pth
echo "=== [1] train aug2-AE (FIXED temp, noise+scale only) ==="
rm -rf ckpt_dir/EEG_condition_things2_aug2 sharedfile
python3 train_ae.py -c $CFG
echo "=== [2] select best + retrieval eval ==="
python3 tools/select_best_encoder.py --ckpt_dir ckpt_dir/EEG_condition_things2_aug2 --config $CFG --out $ENC
python3 eval_things2_retrieval.py --encoder $ENC --config $CFG 2>/dev/null | grep -aE "IMAGE-space" | sed 's/^/[aug2 signal] /'
python3 eval_things2_retrieval.py --encoder $ENC --config $CFG --emb_avg 2>/dev/null | grep -aE "IMAGE-space" | sed 's/^/[aug2 emb_avg] /'
echo "=== [3] backup ==="
TS=$(date +%Y%m%d_%H%M%S); BK="$HOME/models/synapse_backup/aug2_$TS"; mkdir -p "$BK"
grep -aE "\[aug2 |BEST ckpt" "$LOG" > "$BK/RESULTS.txt" 2>/dev/null
cp "$LOG" "$BK/aug2.log"; cp $CFG $ENC "$BK/" 2>/dev/null
echo "==== AUG2 (baseline 0.195) ===="; cat "$BK/RESULTS.txt"; echo "=== AUG2 DONE -> $BK ==="
