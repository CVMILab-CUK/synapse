#!/bin/bash
# Drives the remaining full-data test: wait for Stage-1 AE -> extract encoder -> Stage-2 LDM.
set -e
cd /home/work/models/synapse/src
export HF_HOME=$HOME/.cache/huggingface
export SD21_MODEL_ID="philschmid/stable-diffusion-2-1"

echo "=== waiting for AE full-test to finish ==="
while pgrep -f "train_ae.py -c config/Train_AE_fulltest.json" >/dev/null; do sleep 10; done
echo "=== AE DONE; checkpoints: ==="
ls -t ckpt_dir/EEG_condition_fulltest/*.pth

CKPT=$(ls -t ckpt_dir/EEG_condition_fulltest/*.pth | head -1)
echo "=== extracting encoder from $CKPT ==="
python3 make_pretrained_encoder.py \
  --ckpt "$CKPT" \
  --config config/Train_AE_fulltest.json \
  --out pretrain_models/EEGEncoder_fulltest.pth 2>&1 | grep -E "extracted|missing|unexpected|saved"

echo "=== ENCODER DONE; launching LDM full-test ==="
rm -f sharedfile
python3 train_ldm.py -c config/Train_LDM_fulltest.json 2>&1
echo "=== LDM DONE (exit $?) ==="
