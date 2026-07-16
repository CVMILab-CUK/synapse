"""
Precompute per-unique-image CLIP-text (via BLIP2 caption) and CLIP-image embeddings
for the Stage-1 alignment loss, so AE training does NOT re-run BLIP2 every step.

At THINGS-EEG2 multi-subject scale each image is seen 10x/epoch (and again every
epoch); live BLIP2 captioning dominates runtime. Caching the deterministic
(do_sample=False) embeddings once gives a large speedup and frees ~11GB VRAM
(BLIP2 + open_clip no longer loaded during training), enabling much larger batches.

Output: torch file {image_name: {"text": fp16 (77,1024), "image": fp16 (1024,)}}

Usage:
    python precompute_clip_embeds.py --config config/Train_AE_things2.json \
        --out pretrain_models/clip_embeds_things2.pt
"""
import os
import glob
import json
import argparse

import cv2
import numpy as np
import torch

from datalibs.compose import Resize, Normalization, Scaling
from torchvision import transforms
from models import Frozen_CLIPImage2TextEmbedder as ImageClip


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()
    cfg = json.load(open(args.config))
    img_root = cfg["img_path"]
    device = "cuda"

    # unique image names referenced by any split
    names = set()
    for split in ["eeg_train_path", "eeg_val_path", "eeg_test_path"]:
        for f in glob.glob(os.path.join(cfg[split], "*.pth")):
            names.add(torch.load(f, weights_only=False)["image"])
    names = sorted(names)
    print(f"unique images: {len(names)}")

    tf = transforms.Compose([Resize((args.img_size, args.img_size)), Normalization(), Scaling(0, 1)])
    resize = Resize((args.img_size, args.img_size))
    to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1).float()

    clip = ImageClip()
    clip.model.to(device); clip.blip_model.to(device)

    from PIL import Image
    import torch.nn.functional as F

    cache = {}
    buf_img, buf_pil, buf_names = [], [], []
    dbg = []

    def flush():
        if not buf_names:
            return
        img = torch.stack(buf_img).to(device)
        # --- BLIP2 caption -> CLIP text embedding, PER IMAGE.
        #     FIX: BLIP2 must receive proper uint8/PIL images. Feeding a [0,1] float
        #     tensor corrupted the input -> degenerate captions ("a black and white
        #     image of...") -> collapsed text targets. repetition_penalty kills loops.
        blip_inputs = clip.blip_processor(images=buf_pil, return_tensors="pt").to(device, torch.float16)
        caption_ids = clip.blip_model.generate(**blip_inputs, max_new_tokens=40, num_beams=3,
                                               repetition_penalty=1.5, do_sample=False)
        captions = clip.blip_processor.batch_decode(caption_ids, skip_special_tokens=True)
        tokens = clip.tokenizer([c.strip() for c in captions]).to(device)   # (B,77)
        x = clip.model.text.token_embedding(tokens)
        x = x + clip.model.text.positional_embedding
        x = clip.model.text.transformer(x)
        text_embed = clip.model.text.ln_final(x)                           # (B,77,1024)
        image_embed = clip.model.encode_image(F.interpolate(img, size=224))  # (B,1024)
        for i, nm in enumerate(buf_names):
            cache[nm] = {"text": text_embed[i].half().cpu(), "image": image_embed[i].half().cpu()}
        if len(dbg) < 8:
            dbg.extend(captions[:8])
        buf_img.clear(); buf_pil.clear(); buf_names.clear()

    for k, nm in enumerate(names):
        cid = nm.split("_")[0]
        path = os.path.join(img_root, cid, nm + ".JPEG")
        raw = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)              # uint8 RGB
        buf_pil.append(Image.fromarray(raw))                               # proper input for BLIP2
        buf_img.append(to_t(tf(raw / 255.)))                              # normalized for CLIP image
        buf_names.append(nm)
        if len(buf_names) >= args.batch:
            flush()
            if (k + 1) % (args.batch * 10) == 0:
                print(f"  {k+1}/{len(names)}  e.g. {dbg[:2]}")
    flush()
    print("sample captions:", dbg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(cache, args.out)
    print("saved", len(cache), "embeddings ->", args.out)


if __name__ == "__main__":
    main()
