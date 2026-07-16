"""
Preprocess the raw THINGS-EEG2 dataset (HF: gasparyanartur/things-eeg2) into the
per-sample `.pth` + JPEG layout that SYNAPSE's `EEGPrepDataset` expects.

Raw format (per subject/session .npy = dict):
    raw_eeg_data : float64 (64, n_samples)   # 63 EEG + 1 'stim' channel (index 63)
    ch_names     : 64 names (last = 'stim')
    ch_types     : 63 'eeg' + 1 'stim'
    sfreq        : 1000 (Hz)

Standard Gifford 2022 preprocessing applied here (parametrized):
    - events  : rising edges on the stim channel; code 99999 = block marker (dropped)
    - epoch   : [-pre_ms, +post_ms] around onset (default -200..800 ms, 1000 ms)
    - baseline: subtract per-channel mean over the pre-stimulus window
    - resample: decimate 1000 Hz -> resample_hz (default 100 Hz)  => time = (pre+post)/1000*hz
    - channels: 63 EEG (stim dropped)
    - reps    : averaged across repetitions & sessions per image condition (improves SNR;
                standard for THINGS-EEG2 retrieval/generation)

Disk strategy: 148 GB of raw EEG cannot all be stored. Each (subject, session) raw
file is downloaded, epoched, then DELETED from the HF cache before the next one.

Output (always under ~/data):
    <out>/preprocessing_data/{train,val,test}/<cid>_<gidx>.pth
        {"eeg": float32 (time, 63), "image": "<cid>_<imgidx>", "label": <concept_int>, "subject": <int>}
    <out>/image/<cid>/<cid>_<imgidx>.JPEG     (stimulus image, stored once, shared across subjects)

`<cid>` is a 5-digit concept id (no underscore) so EEGPrepDataset's `name.split("_")`
path logic keeps working unchanged. EEG layout (time, channels) matches the CVPR40
convention the encoder was validated on.
"""
import os
import io
import gc
import zipfile
import argparse

import numpy as np
import torch
from PIL import Image
from scipy.signal import resample as scipy_resample
from huggingface_hub import hf_hub_download

REPO = "gasparyanartur/things-eeg2"
BLOCK_MARKER = 99999


def parse_subjects(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def extract_images(out_root):
    """Extract train/test image zips once; return {split: [stored_name per condition index]}.

    THINGS image_metadata lists files in condition order. We store each image as
    image/<cid>/<cid>_<imgidx>.JPEG where cid is the 5-digit concept id.
    """
    img_root = os.path.join(out_root, "image")
    os.makedirs(img_root, exist_ok=True)
    meta_path = hf_hub_download(REPO, "image_metadata.npy", repo_type="dataset")
    meta = np.load(meta_path, allow_pickle=True).item()

    name_maps = {}
    for split, files_key, concepts_key, zip_name, inner in [
        ("train", "train_img_files", "train_img_concepts", "imgs/training_images.zip", "training_images"),
        ("test",  "test_img_files",  "test_img_concepts",  "imgs/test_images.zip",     "test_images"),
    ]:
        files = meta[files_key]
        concepts = meta[concepts_key]   # e.g. "00001_aardvark"
        names = []
        marker = os.path.join(img_root, f".{split}_extracted")
        zpath = hf_hub_download(REPO, zip_name, repo_type="dataset")
        zf = zipfile.ZipFile(zpath) if not os.path.exists(marker) else None
        # build a lookup of zip members by basename for robust extraction
        members = {}
        if zf is not None:
            for m in zf.namelist():
                members[os.path.basename(m)] = m
        for idx, (fn, concept) in enumerate(zip(files, concepts)):
            cid = concept.split("_")[0]                  # 5-digit id, no underscore
            stored = f"{cid}_{idx}"                       # exactly one underscore
            names.append(stored)
            dst_dir = os.path.join(img_root, cid)
            dst = os.path.join(dst_dir, stored + ".JPEG")
            if zf is not None and not os.path.exists(dst):
                os.makedirs(dst_dir, exist_ok=True)
                member = members.get(os.path.basename(fn))
                if member is None:
                    raise RuntimeError(f"image {fn} not found in {zip_name}")
                Image.open(io.BytesIO(zf.read(member))).convert("RGB").save(dst, "JPEG")
        if zf is not None:
            zf.close(); open(marker, "w").close()
        name_maps[split] = names
        print(f"[images/{split}] {len(names)} conditions ready")
    return name_maps, meta


def epoch_session(npy_path, pre, post, resample_hz):
    """Return (epochs (n_trials, time, 63), codes (n_trials,)) for one session file."""
    d = np.load(npy_path, allow_pickle=True).item()
    data = d["raw_eeg_data"]                 # (64, n_samples)
    sf = int(d["sfreq"])
    eeg = data[:63]                          # drop stim
    stim = data[63].astype(np.int64)
    pre_s, post_s = int(pre / 1000 * sf), int(post / 1000 * sf)
    win = pre_s + post_s
    out_time = int((pre + post) / 1000 * resample_hz)

    onsets = np.where((stim != 0) & (np.concatenate([[0], stim[:-1]]) == 0))[0]
    epochs, codes = [], []
    n = data.shape[1]
    for o in onsets:
        code = int(stim[o])
        if code == BLOCK_MARKER:
            continue
        s0, s1 = o - pre_s, o + post_s
        if s0 < 0 or s1 > n:
            continue
        seg = eeg[:, s0:s1].astype(np.float32)          # (63, win)
        seg = seg - seg[:, :pre_s].mean(axis=1, keepdims=True)   # baseline correct
        if win != out_time:
            seg = scipy_resample(seg, out_time, axis=1).astype(np.float32)
        epochs.append(seg.T)                            # (time, 63)
        codes.append(code)
    del d, data, eeg, stim; gc.collect()
    return np.asarray(epochs, dtype=np.float32), np.asarray(codes, dtype=np.int64)


def _avg_conditions(subj, raw_split, pre, post, resample_hz, keep_raw):
    """Epoch all 4 sessions and rep-average per condition code. Returns {code: (time,63) float32}."""
    acc, cnt = {}, {}
    for ses in range(1, 5):
        rel = f"raw-eeg/sub-{subj:02d}/ses-{ses:02d}/raw_eeg_{raw_split}.npy"
        path = hf_hub_download(REPO, rel, repo_type="dataset")
        ep, codes = epoch_session(path, pre, post, resample_hz)
        for e, c in zip(ep, codes):
            if c not in acc:
                acc[c] = e.astype(np.float64); cnt[c] = 1
            else:
                acc[c] += e; cnt[c] += 1
        del ep, codes; gc.collect()
        if not keep_raw:
            for q in (os.path.realpath(path), path):
                try: os.remove(q)
                except OSError: pass
        print(f"  sub-{subj:02d} {raw_split} ses-{ses:02d}: {len(acc)} conditions so far")
    return {c: (acc[c] / cnt[c]).astype(np.float32) for c in acc}


def process_subject(subj, out_root, name_maps, meta, split_map, pre, post, resample_hz, keep_raw):
    pre_root = os.path.join(out_root, "preprocessing_data")
    for s in split_map.values():
        os.makedirs(os.path.join(pre_root, s), exist_ok=True)

    # 1) rep-average each split's conditions
    train_avg = _avg_conditions(subj, "training", pre, post, resample_hz, keep_raw)
    test_avg = _avg_conditions(subj, "test", pre, post, resample_hz, keep_raw)

    # 2) per-channel z-score stats from TRAIN only (raw EEG is in volts ~1e-5; standardize like CVPR40)
    stack = np.stack(list(train_avg.values()), 0)          # (n_train, time, 63)
    ch_mean = stack.mean(axis=(0, 1), keepdims=True)       # (1,1,63)
    ch_std = stack.std(axis=(0, 1), keepdims=True) + 1e-8
    del stack; gc.collect()
    np.save(os.path.join(out_root, f"zscore_sub-{subj:02d}.npy"),
            {"mean": ch_mean.squeeze(), "std": ch_std.squeeze()}, allow_pickle=True)

    # 3) standardize + write both splits
    for raw_split, out_split, concepts_key in [
        ("training", split_map["train"], "train_img_concepts"),
        ("test",     split_map["test"],  "test_img_concepts"),
    ]:
        concepts = meta[concepts_key]
        names = name_maps["train" if raw_split == "training" else "test"]
        avg = train_avg if raw_split == "training" else test_avg
        written = 0
        for code, eeg in avg.items():
            cond_idx = code - 1                            # stim codes are 1-based
            if cond_idx < 0 or cond_idx >= len(names):
                continue
            eeg = ((eeg - ch_mean[0]) / ch_std[0]).astype(np.float32)  # (time,63) standardized
            img_name = names[cond_idx]
            label = int(concepts[cond_idx].split("_")[0])
            stored = f"{img_name.split('_')[0]}_{subj:02d}{cond_idx:06d}"  # unique per subj+cond, 1 underscore
            torch.save(
                {"eeg": eeg, "image": img_name, "label": label, "subject": subj},
                os.path.join(pre_root, out_split, stored + ".pth"),
            )
            written += 1
        print(f"[sub-{subj:02d}/{out_split}] wrote {written} samples")
    del train_avg, test_avg; gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--subjects", default="1-10")
    p.add_argument("--pre-ms", type=float, default=200)
    p.add_argument("--post-ms", type=float, default=800)
    p.add_argument("--resample-hz", type=int, default=100)
    p.add_argument("--keep-raw", action="store_true", help="don't delete raw .npy after processing")
    p.add_argument("--train-out", default="train", help="output split dir for THINGS training EEG")
    p.add_argument("--test-out", default="test", help="output split dir for THINGS test EEG")
    args = p.parse_args()

    out = os.path.abspath(os.path.expanduser(args.out))
    home_data = os.path.abspath(os.path.expanduser("~/data"))
    assert out == home_data or out.startswith(home_data + os.sep), f"output must be under ~/data, got {out}"

    split_map = {"train": args.train_out, "test": args.test_out}
    name_maps, meta = extract_images(out)
    for subj in parse_subjects(args.subjects):
        print(f"=== subject {subj} ===")
        process_subject(subj, out, name_maps, meta, split_map, args.pre_ms, args.post_ms,
                        args.resample_hz, args.keep_raw)
    print("done ->", out)


if __name__ == "__main__":
    main()
