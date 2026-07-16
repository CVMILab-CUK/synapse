"""
Convert the HuggingFace dataset `luigi-s/EEG_Image_CVPR_ALL_subj` into the
per-sample `.pth` + JPEG layout that SYNAPSE's trainers expect.

HF parquet schema:
    image              : {bytes, path}   stimulus image (JPEG bytes)
    conditioning_image : List[List[float]]  EEG signal, shape (440, 128) = (time, channels)
                         This is the model's expected input layout (B, 440, 128); stored as-is.
    caption            : str
    label_folder       : str   ImageNet wnid (e.g. n02106662)
    label              : int
    subject            : int

Output layout (always under ~/data):
    <out_root>/preprocessing_data/{train,val,test}/<wnid>_<idx>.pth
    <out_root>/image/<wnid>/<wnid>_<idx>.JPEG

Each .pth holds: {"eeg": float32 (440,128), "image": "<wnid>_<idx>", "label": int, "subject": int}
The trainer reconstructs the JPEG path as image_path/<wnid>/<name>.JPEG and splits
the name on "_" (exactly one underscore -> wnid has none, so name = "<wnid>_<idx>").

Usage:
    python -m datalibs.hf_to_local --out ~/data/eeg_cvpr            # full dataset
    python -m datalibs.hf_to_local --out ~/data/eeg_cvpr --limit 24 # small subset per split
"""
import os
import io
import argparse

import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download

REPO = "luigi-s/EEG_Image_CVPR_ALL_subj"
SHARDS = {
    "train": [f"data/train-0000{i}-of-00006.parquet" for i in range(6)],
    "val":   [f"data/validation-0000{i}-of-00002.parquet" for i in range(2)],
    "test":  [f"data/test-0000{i}-of-00002.parquet" for i in range(2)],
}
COLS = ["conditioning_image", "label_folder", "label", "subject", "image"]


def convert(out_root, limit=None):
    pre_root = os.path.join(out_root, "preprocessing_data")
    img_root = os.path.join(out_root, "image")
    for split in SHARDS:
        os.makedirs(os.path.join(pre_root, split), exist_ok=True)
    os.makedirs(img_root, exist_ok=True)

    for split, shards in SHARDS.items():
        idx = 0
        for shard in shards:
            path = hf_hub_download(REPO, shard, repo_type="dataset")
            table = pq.read_table(path, columns=COLS)
            for row in table.to_pylist():
                if limit is not None and idx >= limit:
                    break
                wnid = row["label_folder"]
                name = f"{wnid}_{idx}"          # exactly one underscore (wnid has none)

                eeg = np.asarray(row["conditioning_image"], dtype=np.float32)  # (440, 128)
                assert eeg.shape == (440, 128), f"unexpected eeg shape {eeg.shape}"

                cls_dir = os.path.join(img_root, wnid)
                os.makedirs(cls_dir, exist_ok=True)
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                img.save(os.path.join(cls_dir, f"{name}.JPEG"), "JPEG")

                torch.save(
                    {"eeg": eeg, "image": name, "label": int(row["label"]), "subject": int(row["subject"])},
                    os.path.join(pre_root, split, f"{name}.pth"),
                )
                idx += 1
            if limit is not None and idx >= limit:
                break
        print(f"[{split}] wrote {idx} samples")
    print("done ->", out_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output root (must be under ~/data)")
    p.add_argument("--limit", type=int, default=None, help="max samples per split (for subsets)")
    args = p.parse_args()
    out = os.path.abspath(os.path.expanduser(args.out))
    home_data = os.path.abspath(os.path.expanduser("~/data"))
    assert out == home_data or out.startswith(home_data + os.sep), \
        f"output must live under ~/data, got {out}"
    convert(out, args.limit)
