# SYNAPSE — Reproduce / Dry-run Guide

This document explains how to get the SYNAPSE code training again from scratch on
this machine, including data download, a fast dry-run of each stage, and the
Stage-1 → encoder → Stage-2 hand-off.

> All data lives under `~/data` (hard requirement). Models/HF cache live under
> `~/.cache/huggingface`.

## 0. Environment

```bash
pip install diffusers==0.32.2 open_clip_torch==2.30.0 omegaconf k-diffusion kornia lightning
# (torch / transformers / accelerate / datasets are already present)
```

Notes specific to this environment:
- GPU: single NVIDIA H200 (143 GB). The trainers use `torch.multiprocessing.spawn`
  DDP, so they run with `world_size = 1` here — that is fine.
- `models/__init__.py` and `trainer/__init__.py` were patched to make the **legacy**
  DreamDiffusion modules (`eeg_LDM`, `eeg_ae_trainer`, `eeg_ldm_trainer`, which need
  the unpublished `dc_ldm`/`ldm`/`taming`/`clip` packages) **optional**. The active
  path — `eeg_ae_SD2_blip_trainer` (Stage 1) and `eeg_ldm2_ddp_trainer` (Stage 2) —
  does not need them.
- `stabilityai/stable-diffusion-2-1` is now **gated**. `models/eeg_LDM2.py` reads the
  model id from the `SD21_MODEL_ID` env var (default = the gated repo). Use a
  non-gated faithful mirror (v_prediction, cross-attention dim 1024), e.g.
  `philschmid/stable-diffusion-2-1`.

## 1. Data

Source (easy route): HuggingFace `luigi-s/EEG_Image_CVPR_ALL_subj` (parquet, bundles
the EEG signal as `conditioning_image` (440×128 = time×channels), the stimulus JPEG,
`caption`, `label_folder` (wnid), `label`, `subject`).

Convert it into the per-sample `.pth` + JPEG layout the trainers expect:

```bash
cd src
# full dataset  -> ~/data/eeg_cvpr   (train 7959 / val 1994 / test 1987)
python -m datalibs.hf_to_local --out ~/data/eeg_cvpr
# tiny subset for dry-runs -> ~/data/eeg_cvpr_dryrun (24 per split)
python -m datalibs.hf_to_local --out ~/data/eeg_cvpr_dryrun --limit 24
```

Layout produced:
```
~/data/eeg_cvpr/
  preprocessing_data/{train,val,test}/<wnid>_<idx>.pth   # {"eeg":(440,128), "image":"<wnid>_<idx>", "label":int, "subject":int}
  image/<wnid>/<wnid>_<idx>.JPEG
```
The original repo's `preprocessing.ipynb` produces the same layout from the raw
`eeg_5_95_std.pth` + ImageNet JPEGs; `datalibs/hf_to_local.py` is the HF-based shortcut.

## 2. Stage 1 — CLIP-aligned EEG AutoEncoder

```bash
cd src
python train_ae.py -c config/Train_AE_dryrun.json     # dry-run (subset, tiny iters)
# full training:
python train_ae.py -c config/Train_AE.json
```
Stage 1 downloads `open_clip ViT-H-14 (laion2b)` and `Salesforce/blip2-flan-t5-xl`
(BLIP2 generates the caption that is CLIP-text-encoded for the alignment loss).
Checkpoints: `src/ckpt_dir/<name>/<name>_<step>.pth`.

## 3. Extract the pretrained encoder (was `Make Prtrained Encoder.ipynb`)

Ported to `src/make_pretrained_encoder.py`:

```bash
cd src
python make_pretrained_encoder.py \
  --ckpt   ./ckpt_dir/EEG_condition_dryrun/EEG_condition_dryrun_0.pth \
  --config ./config/Train_AE_dryrun.json \
  --out    ./pretrain_models/EEGEncoder_dryrun.pth
```
It strips the `Encoder.` prefix from the AE checkpoint and saves `{"net": ...}`.
The encoder hyper-parameters are read from `--config`, so they stay consistent with
both Stage-1 training and the Stage-2 `cond_stage_model2`. **Important:** Stage-1 and
Stage-2 must agree on encoder geometry (`n_layer`, `n_head`, `dims`, `stride`,
`skip_mode`, `pos_mode`, `global_attn`), otherwise the encoder will not load into the
LDM cond stage (shape-affecting params) or will run attention with a different head
split than it was trained with (`n_head`).

Reconciliation applied: the upstream repo's `Train_AE.json` (`n_layer=3`, `n_head=16`)
disagreed with the extraction notebook + LDM contract (`n_layer=2`, `n_head=64`,
`global_attn=True`) — a full run would crash at encoder load. `Train_AE.json`,
`Train_AE_dryrun.json`, and `Train_LDM*` are now all aligned at `n_layer=2`,
`n_head=64`. (Note: `eeg_ldm2_ddp_trainer` does not thread `global_attn` from config —
it relies on `eLDM2`'s default `global_attn=True`, which matches the AE configs here.)

## 4. Stage 2 — EEG-conditioned Stable Diffusion (LDM)

```bash
cd src
SD21_MODEL_ID="philschmid/stable-diffusion-2-1" \
  python train_ldm.py -c config/Train_LDM_dryrun.json -m ddp    # dry-run
# full training (after pointing eeg_pretrian_path at your extracted encoder):
SD21_MODEL_ID="philschmid/stable-diffusion-2-1" \
  python train_ldm.py -c config/Train_LDM.json -m ddp
```
`config/Train_LDM_dryrun.json` points `eeg_pretrian_path` at the extracted encoder,
uses SD2.1 (vae+unet+scheduler, v_prediction), IP-Adapter on, EMA off (disk), and a
tiny step budget.
