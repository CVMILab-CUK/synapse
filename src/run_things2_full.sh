#!/bin/bash
# Full auto-chain for THINGS-EEG2 (multi-subject, subject-free, cached embeds, max batch):
#   wait for CLIP embed cache + all-10-subject preprocessing
#   -> Stage1 AE (batch 512, cached) -> extract encoder -> retrieval eval -> Stage2 LDM (batch 64)
set -e
cd /home/work/models/synapse/src

EMB=pretrain_models/clip_embeds_things2.pt
PP_LOG="$1"   # preprocessing task output log (for "done ->" completion)

echo "=== [1/5] waiting for CLIP embed cache ($EMB) ==="
until [ -f "$EMB" ]; do sleep 10; done
echo "    embed cache ready"

echo "=== [2/5] waiting for all-10-subject preprocessing to finish ==="
if [ -n "$PP_LOG" ]; then
  until grep -q "done ->" "$PP_LOG" 2>/dev/null; do sleep 30; done
fi
echo "    train samples: $(ls preprocessing_data 2>/dev/null; ls /home/work/data/things_eeg2/preprocessing_data/train | wc -l)"

echo "=== [3/5] Stage1 AE (multi-subject, cached embeds, batch 512) ==="
rm -f sharedfile
python3 train_ae.py -c config/Train_AE_things2.json

# pick highest-step checkpoint by the TRAILING number (path contains underscores, so
# sort -t_ on the full path is wrong — extract the step suffix and sort numerically)
LATEST=$(ls ckpt_dir/EEG_condition_things2/*.pth | sed -E 's/.*_([0-9]+)\.pth$/\1/' | sort -n | tail -1)
CKPT="ckpt_dir/EEG_condition_things2/EEG_condition_things2_${LATEST}.pth"
echo "=== [4/5] extract encoder from $CKPT + retrieval eval ==="
python3 make_pretrained_encoder.py --ckpt "$CKPT" --config config/Train_AE_things2.json \
  --out pretrain_models/EEGEncoder_things2.pth
python3 eval_things2_retrieval.py --encoder pretrain_models/EEGEncoder_things2.pth \
  --config config/Train_AE_things2.json --nway-repeats 200 || echo "eval failed (non-fatal)"

echo "=== [5/5] Stage2 LDM on THINGS-EEG2 (batch 64) ==="
rm -f sharedfile
SD21_MODEL_ID="philschmid/stable-diffusion-2-1" HF_HOME=$HOME/.cache/huggingface \
  python3 train_ldm.py -c config/Train_LDM_things2.json
echo "=== THINGS-EEG2 FULL PIPELINE DONE ==="
