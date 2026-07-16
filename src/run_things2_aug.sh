#!/bin/bash
# THINGS-EEG2 encoder top-1 improvement experiment:
#   augmentation (encoder-input only, denoising recon) + learnable InfoNCE temperature
#   + embedding-space eval averaging.  Baseline native-63 top-1 = 0.195 (signal-space avg).
SRC=/home/work/models/synapse/src
LOG=/tmp/things2_aug.log
cd "$SRC"
export HF_HOME=$HOME/.cache/huggingface

BASE=pretrain_models/EEGEncoder_things2.pth
BASECFG=config/Train_AE_things2.json
AUGCFG=config/Train_AE_things2_aug.json
AUGENC=pretrain_models/EEGEncoder_things2_aug.pth

echo "=== [0] BASELINE eval-fix isolation (native-63, NO retrain) ==="
python3 eval_things2_retrieval.py --encoder $BASE --config $BASECFG 2>/dev/null \
  | grep -aE "IMAGE-space" | sed 's/^/[base signal] /'
python3 eval_things2_retrieval.py --encoder $BASE --config $BASECFG --emb_avg 2>/dev/null \
  | grep -aE "IMAGE-space" | sed 's/^/[base emb_avg] /'

echo "=== [1] train aug-AE (augmentation + learnable temp) ==="
rm -rf ckpt_dir/EEG_condition_things2_aug sharedfile
python3 train_ae.py -c $AUGCFG

echo "=== [2] select best aug encoder + retrieval eval ==="
python3 tools/select_best_encoder.py --ckpt_dir ckpt_dir/EEG_condition_things2_aug --config $AUGCFG --out $AUGENC
python3 eval_things2_retrieval.py --encoder $AUGENC --config $AUGCFG 2>/dev/null \
  | grep -aE "IMAGE-space" | sed 's/^/[aug signal] /'
python3 eval_things2_retrieval.py --encoder $AUGENC --config $AUGCFG --emb_avg 2>/dev/null \
  | grep -aE "IMAGE-space" | sed 's/^/[aug emb_avg] /'

echo "=== [3] backup ==="
TS=$(date +%Y%m%d_%H%M%S); BK="$HOME/models/synapse_backup/aug_$TS"; mkdir -p "$BK"
grep -aE "\[base |\[aug |BEST ckpt" "$LOG" > "$BK/RESULTS.txt" 2>/dev/null
cp "$LOG" "$BK/aug.log" 2>/dev/null
cp $AUGCFG "$BK/" 2>/dev/null; cp $AUGENC "$BK/" 2>/dev/null
cp trainer/eeg_ae_SD2_blip_trainer.py eval_things2_retrieval.py "$BK/" 2>/dev/null
echo "===== AUG RESULTS (baseline 0.195) ====="; cat "$BK/RESULTS.txt"
echo "=== AUG DONE -> $BK ==="
