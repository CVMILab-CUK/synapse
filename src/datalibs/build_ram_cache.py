"""
Consolidate per-sample THINGS-EEG2 .pth files into one tensor blob per split, so
training can load the whole set into RAM once (EEGRamDataset) instead of doing
165k per-sample torch.load calls — removes the dataloader bottleneck at big batch.

Output: <out>/ram_<split>.pt = {"eeg": float32 (N,T,C), "names": list, "labels": (N,), "subjects": (N,)}

Usage:
    python -m datalibs.build_ram_cache --root ~/data/things_eeg2 --splits train test
"""
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def _load(f):
    d = torch.load(f, weights_only=False)
    return (np.asarray(d["eeg"], dtype=np.float32), d["image"], int(d["label"]), int(d.get("subject", 0)))


def build_split(root, split, workers=32):
    pre = os.path.join(root, "preprocessing_data", split)
    out = os.path.join(root, f"ram_{split}.pt")
    files = sorted(glob.glob(os.path.join(pre, "*.pth")))
    print(f"[{split}] {len(files)} files -> {out}")
    eeg, names, labels, subs = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, (e, nm, lb, sb) in enumerate(ex.map(_load, files, chunksize=64)):
            eeg.append(e); names.append(nm); labels.append(lb); subs.append(sb)
            if (k + 1) % 20000 == 0:
                print(f"  {k+1}/{len(files)}")
    blob = {
        "eeg": torch.from_numpy(np.stack(eeg)),            # (N,T,C) float32
        "names": names,
        "labels": torch.tensor(labels, dtype=torch.long),
        "subjects": torch.tensor(subs, dtype=torch.long),
    }
    torch.save(blob, out)
    print(f"[{split}] saved {blob['eeg'].shape} -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    for s in args.splits:
        build_split(root, s, args.workers)
