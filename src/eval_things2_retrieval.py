"""
THINGS-EEG2 semantic evaluation for the Stage-1 CLIP-aligned EEG encoder.

This directly answers the CVPR reviewers' core criticism (semantic fidelity / low GA):
instead of 40-way classification GA, it reports the metrics standard in the
THINGS-EEG2 literature (NICE, Song et al.):
  - zero-shot top-1 / top-5 image retrieval over the 200 test concepts
  - 2-way and 10-way identification accuracy (averaged over random distractor draws)

EEG embedding = mean-pooled encoder latent (B,1024); image embedding = open_clip
ViT-H-14 image features (1024). Both L2-normalized; cosine similarity used.

Usage:
    python eval_things2_retrieval.py \
        --encoder ./pretrain_models/EEGEncoder_things2.pth \
        --config  ./config/Train_AE_things2.json \
        [--subject 1]            # restrict to one subject's test EEG (default: all)
"""
import os
import glob
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from models.eeg_AE import eeg_encoder as Encoder
import open_clip


def build_encoder(cfg, device):
    enc = Encoder(
        in_seq=cfg["in_seq"], in_channels=cfg["in_channels"], real_channels=cfg.get("real_channels", None), out_channels=cfg["z_channels"],
        out_seq=cfg["out_seq"], dims=cfg["dims"], shortcut=bool(cfg["shortcut"]), dropout=0.0,
        groups=cfg["groups"], layer_mode=cfg["layer_mode"], block_mode=cfg["block_mode"],
        down_mode=cfg["down_mode"], pos_mode=cfg["pos_mode"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], dff_factor=cfg["dff_factor"], stride=cfg["stride"],
        skip_mode=cfg["skip_mode"], global_attn=bool(cfg.get("global_attn", 1)),
    ).to(device).eval()
    sd = torch.load(cfg["_encoder_path"], map_location="cpu", weights_only=False)["net"]
    enc.load_state_dict(sd, strict=True)
    return enc


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--subject", type=int, default=None)
    p.add_argument("--emb_avg", action="store_true", help="average trials in embedding space (not raw signal)")
    p.add_argument("--nway-repeats", type=int, default=1000)
    args = p.parse_args()
    device = "cuda"

    cfg = json.load(open(args.config)); cfg["_encoder_path"] = args.encoder
    enc = build_encoder(cfg, device)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", force_custom_text=True)
    clip_model = clip_model.to(device).eval()

    test_dir = cfg["eeg_test_path"]; img_root = cfg["img_path"]
    files = sorted(glob.glob(os.path.join(test_dir, "*.pth")))

    # group test EEG by image (one ground-truth image per concept); average across subjects
    by_img = {}
    for f in files:
        d = torch.load(f, weights_only=False)
        if args.subject is not None and d["subject"] != args.subject:
            continue
        by_img.setdefault(d["image"], []).append(np.asarray(d["eeg"], dtype=np.float32))
    img_names = sorted(by_img.keys())
    print(f"test concepts: {len(img_names)} | subject filter: {args.subject}")

    # EEG embeddings (mean over repeats/subjects per image, then encode).
    # Keep both: mean-pooled (for image space) and full latent (for text space).
    eeg_img_emb, eeg_txt_emb = [], []
    for name in img_names:
        if args.emb_avg:
            # encode each repeat/subject trial separately, then average in EMBEDDING space
            # (avoids cancelling mis-aligned across-subject signals in raw space).
            batch = torch.from_numpy(np.stack(by_img[name])).to(device)     # (K,time,ch)
            lat, _ = enc(batch); lat = lat.permute(0, 2, 1)                 # (K,77,1024)
            im = F.normalize(F.normalize(lat[:, 0, :], dim=-1).mean(0, keepdim=True), dim=-1)
            tx = F.normalize(F.normalize(lat[:, 1:, :].mean(1), dim=-1).mean(0, keepdim=True), dim=-1)
        else:
            eeg = torch.from_numpy(np.mean(by_img[name], axis=0)).unsqueeze(0).to(device)  # (1,time,ch)
            lat, _ = enc(eeg); lat = lat.permute(0, 2, 1)                   # (1,77,1024)
            im = F.normalize(lat[:, 0, :], dim=-1)
            tx = F.normalize(lat[:, 1:, :].mean(1), dim=-1)
        eeg_img_emb.append(im)            # token0 -> image-contrastive space
        eeg_txt_emb.append(tx)            # tokens 1:77 -> text-aligned space
    eeg_img_emb = torch.cat(eeg_img_emb, 0)
    eeg_txt_emb = torch.cat(eeg_txt_emb, 0)

    # image embeddings via CLIP (image-projected space)
    img_emb = []
    for name in img_names:
        cid = name.split("_")[0]
        img = cv2.cvtColor(cv2.imread(os.path.join(img_root, cid, name + ".JPEG")), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073]); std = np.array([0.26862954, 0.26130258, 0.27577711])
        img = (img - mean) / std
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img_emb.append(F.normalize(clip_model.encode_image(t).float(), dim=-1))
    img_emb = torch.cat(img_emb, 0)                # (N,1024)

    # text targets (mean-pooled caption penultimate) from the precomputed cache — the
    # space the SYNAPSE Stage-1 actually aligns to (ALIGN/COSINE). This reveals whether
    # the encoder learned, independent of the image-space mismatch.
    txt_emb = None
    cache_path = cfg.get("cached_embed_path")
    if cache_path and os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        txt_emb = F.normalize(torch.stack([cache[n]["text"].float().mean(0) for n in img_names]).to(device), dim=-1)

    N = len(img_names)
    gt = torch.arange(N, device=device)

    def report(tag, sim):
        ranks = (sim.argsort(dim=1, descending=True) == gt[:, None]).float().argmax(1)
        top1 = (ranks == 0).float().mean().item(); top5 = (ranks < 5).float().mean().item()
        g = torch.Generator(device=device).manual_seed(0)
        def nway(n):
            acc = 0.0
            for i in range(N):
                wins = 0
                for _ in range(args.nway_repeats):
                    distract = torch.randperm(N, generator=g, device=device)
                    distract = distract[distract != i][: n - 1]
                    cand = torch.cat([gt[i:i+1], distract])
                    wins += int(sim[i, cand].argmax().item() == 0)
                acc += wins / args.nway_repeats
            return acc / N
        print(f"[{tag}] top-1 {top1:.4f} | top-5 {top5:.4f} | 2-way {nway(2):.4f} | 10-way {nway(10):.4f}  (chance top1={1/N:.4f})")

    report("IMAGE-space (EEG token0 vs CLIP image)", eeg_img_emb @ img_emb.t())
    if txt_emb is not None:
        report("TEXT-space  (EEG tok1: mean vs CLIP caption)", eeg_txt_emb @ txt_emb.t())


if __name__ == "__main__":
    main()
