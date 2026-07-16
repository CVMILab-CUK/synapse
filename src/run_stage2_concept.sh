#!/bin/bash
# Stage-2 only (encoder already trained/extracted): train LDM -> taxonomy-free gen-eval
# -> backup. Uses the file-system sharing fix in train_ldm.py to avoid shm exhaustion.
SRC=/home/work/models/synapse/src
LOG=/tmp/things2_concept.log
cd "$SRC"

echo "=== Stage2 LDM (concept-text encoder, token0 -> IP-Adapter) ==="
rm -f sharedfile
SD21_MODEL_ID="philschmid/stable-diffusion-2-1" HF_HOME=$HOME/.cache/huggingface \
  python3 train_ldm.py -c config/Train_LDM_things2.json
echo "=== STAGE2 DONE ==="

echo "=== gen-eval (taxonomy-free CLIP) ==="
LDMCK=$(for d in ckpt_dir/EEG_LDM_things2/checkpoint-*; do [ -d "$d/unet" ] && echo "$d"; done | sort -t- -k2 -n | tail -1)
echo "gen-eval on $LDMCK"
rm -rf output/stage2_eval_concept
SD21_MODEL_ID="philschmid/stable-diffusion-2-1" HF_HOME=$HOME/.cache/huggingface \
  python3 tools/eval_stage2_things2.py --ckpt "$LDMCK" --n 64 --steps 50 --cfg 7.5 \
  --out output/stage2_eval_concept || echo "gen-eval failed (non-fatal)"

echo "=== FID (perceptual quality) ==="
SD21_MODEL_ID="philschmid/stable-diffusion-2-1" HF_HOME=$HOME/.cache/huggingface \
  python3 tools/eval_fid_things2.py --ckpt "$LDMCK" --n 2000 --steps 50 --cfg 7.5 || echo "FID failed (non-fatal)"

echo "=== backup ==="
TS=$(date +%Y%m%d_%H%M%S); BK="$HOME/models/synapse_backup/concept_$TS"
mkdir -p "$BK/samples"
grep -aE "IMAGE-space|TEXT-space|BEST ckpt|STAGE2 GEN EVAL|gen->concept|2-way ident|mean CLIP|GA\(top_k\)" "$LOG" > "$BK/metrics.txt" 2>/dev/null
cp "$LOG" "$BK/run.log" 2>/dev/null
cp config/Train_AE_things2.json config/Train_LDM_things2.json pretrain_models/EEGEncoder_things2.pth "$BK/" 2>/dev/null
cp tools/build_concept_text_cache.py tools/eval_stage2_things2.py "$BK/" 2>/dev/null
cp output/stage2_eval_concept/0[0-2]*.png "$BK/samples/" 2>/dev/null
du -sh "$BK" > "$BK/_size.txt" 2>/dev/null
echo "=== CONCEPT BACKUP DONE -> $BK ==="
