#!/bin/bash
# Stage-2 (LDM) bounded run on THINGS-EEG2 using the (frozen) Stage-1 encoder, then
# auto-backup report+logs+samples+conversation+planning. Self-contained, no approval.
SRC=/home/work/models/synapse/src
LOG=/tmp/things2_stage2.log   # this script's stdout (nohup) lands here
cd "$SRC"

echo "=== Stage2 LDM training (bounded) ==="
rm -f sharedfile
SD21_MODEL_ID="philschmid/stable-diffusion-2-1" HF_HOME=$HOME/.cache/huggingface \
  python3 train_ldm.py -c config/Train_LDM_things2.json
echo "=== STAGE2 DONE ==="

# ---- backup ----
TS=$(date +%Y%m%d_%H%M%S)
BK="$HOME/models/synapse_backup/stage2_$TS"
mkdir -p "$BK/code" "$BK/samples" "$BK/conversation" "$BK/planning"
python3 tools/make_report.py --log "$LOG" --out "$BK/REPORT.md" 2>>"$BK/backup.err"
cp "$LOG" "$BK/stage2.log" 2>/dev/null
cp config/Train_AE_things2.json config/Train_LDM_things2.json "$BK/" 2>/dev/null
cp ../REPRODUCE.md "$BK/" 2>/dev/null
cp datalibs/things_eeg2_preprocess.py datalibs/build_ram_cache.py precompute_clip_embeds.py \
   eval_things2_retrieval.py make_pretrained_encoder.py run_things2_full.sh \
   run_things2_stage2.sh tools/probe_AB.py "$BK/code/" 2>/dev/null
cp pretrain_models/EEGEncoder_things2.pth "$BK/" 2>/dev/null
{ echo "== EEG_LDM_things2 =="; find ckpt_dir/EEG_LDM_things2 -maxdepth 2 2>/dev/null; } > "$BK/checkpoints.txt"
find output ckpt_dir/EEG_LDM_things2 -name "*.png" 2>/dev/null | head -80 \
  | while read f; do cp "$f" "$BK/samples/" 2>/dev/null; done
PROJ="$HOME/.claude/projects/-home-work-models-synapse"; SID="b591f5ec-7959-4d88-bdf4-b5764a41823f"
cp "$PROJ/$SID.jsonl" "$BK/conversation/transcript.jsonl" 2>/dev/null
python3 tools/extract_transcript.py --jsonl "$PROJ/$SID.jsonl" --out "$BK/conversation/transcript.md" 2>>"$BK/backup.err"
cp -r "$PROJ/memory" "$BK/conversation/memory" 2>/dev/null
cp -r /home/work/models/synapse/.omc "$BK/planning/omc_repo" 2>/dev/null
du -sh "$BK" > "$BK/_size.txt" 2>/dev/null
echo "=== STAGE2 BACKUP DONE -> $BK ==="
