#!/bin/bash
# THINGS-EEG2 v2 (CLIP-style token split): Stage1 AE (token0->image-contrastive,
# tok1:77->text) -> extract encoder -> dual retrieval eval -> Stage2 LDM (IP-Adapter
# fed by token0) -> auto-backup. Detached & self-contained.
# (no set -e: always reach the backup step even if a stage fails)
SRC=/home/work/models/synapse/src
LOG=/tmp/things2_v2.log
cd "$SRC"

echo "=== [1/4] Stage1 AE (token-split, batch 2048, RAM) ==="
rm -rf ckpt_dir/EEG_condition_things2 sharedfile
python3 train_ae.py -c config/Train_AE_things2.json

echo "=== [2/4] select BEST checkpoint (Stage1 overfits) + retrieval eval ==="
python3 tools/select_best_encoder.py --ckpt_dir ckpt_dir/EEG_condition_things2 \
  --config config/Train_AE_things2.json --out pretrain_models/EEGEncoder_things2.pth
python3 eval_things2_retrieval.py --encoder pretrain_models/EEGEncoder_things2.pth \
  --config config/Train_AE_things2.json --nway-repeats 200 || echo "eval failed (non-fatal)"

echo "=== [3/4] Stage2 LDM (token0 -> IP-Adapter) ==="
rm -rf ckpt_dir/EEG_condition_things2   # encoder already extracted -> free disk before Stage2
rm -f sharedfile
SD21_MODEL_ID="philschmid/stable-diffusion-2-1" HF_HOME=$HOME/.cache/huggingface \
  python3 train_ldm.py -c config/Train_LDM_things2.json
echo "=== V2 PIPELINE DONE ==="

echo "=== [3b/4] Stage2 generation eval (taxonomy-free CLIP) ==="
LDMCK=$(for d in ckpt_dir/EEG_LDM_things2/checkpoint-*; do [ -d "$d/unet" ] && echo "$d"; done | sort -t- -k2 -n | tail -1)
echo "gen-eval on $LDMCK"
rm -rf output/stage2_eval_concept
SD21_MODEL_ID="philschmid/stable-diffusion-2-1" HF_HOME=$HOME/.cache/huggingface \
  python3 tools/eval_stage2_things2.py --ckpt "$LDMCK" --n 64 --steps 50 --cfg 7.5 \
  --out output/stage2_eval_concept || echo "gen-eval failed (non-fatal)"

echo "=== [4/4] backup ==="
TS=$(date +%Y%m%d_%H%M%S); BK="$HOME/models/synapse_backup/v2_$TS"
mkdir -p "$BK/code" "$BK/samples" "$BK/conversation" "$BK/planning"
python3 tools/make_report.py --log "$LOG" --out "$BK/REPORT.md" 2>>"$BK/backup.err"
cp "$LOG" "$BK/v2.log" 2>/dev/null
cp config/Train_AE_things2.json config/Train_LDM_things2.json "$BK/" 2>/dev/null
cp ../REPRODUCE.md "$BK/" 2>/dev/null
cp datalibs/things_eeg2_preprocess.py datalibs/build_ram_cache.py precompute_clip_embeds.py \
   eval_things2_retrieval.py make_pretrained_encoder.py run_things2_v2.sh tools/probe_AB.py "$BK/code/" 2>/dev/null
cp pretrain_models/EEGEncoder_things2.pth "$BK/" 2>/dev/null
{ echo "== AE =="; ls -la ckpt_dir/EEG_condition_things2 2>/dev/null; echo "== LDM =="; find ckpt_dir/EEG_LDM_things2 -maxdepth 2 2>/dev/null; } > "$BK/checkpoints.txt"
find output ckpt_dir/EEG_LDM_things2 -name "*.png" 2>/dev/null | head -80 | while read f; do cp "$f" "$BK/samples/" 2>/dev/null; done
PROJ="$HOME/.claude/projects/-home-work-models-synapse"; SID="b591f5ec-7959-4d88-bdf4-b5764a41823f"
cp "$PROJ/$SID.jsonl" "$BK/conversation/transcript.jsonl" 2>/dev/null
python3 tools/extract_transcript.py --jsonl "$PROJ/$SID.jsonl" --out "$BK/conversation/transcript.md" 2>>"$BK/backup.err"
cp -r /home/work/models/synapse/.omc "$BK/planning/omc_repo" 2>/dev/null
du -sh "$BK" > "$BK/_size.txt" 2>/dev/null
echo "=== V2 BACKUP DONE -> $BK ==="
