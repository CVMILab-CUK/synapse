"""
Extract the encoder-only weights from a trained EEG AutoEncoder checkpoint.

This is the Python port of `Make Prtrained Encoder.ipynb`. Stage-1 training
(`train_ae.py`) saves a full autoencoder checkpoint whose state_dict keys are
prefixed with ``Encoder.`` / ``Decoder.`` (the AE wraps an ``eeg_encoder`` as
``self.Encoder``). Stage-2 (`train_ldm.py`) only needs the encoder, loaded into a
bare ``eeg_encoder``. This script strips the ``Encoder.`` prefix and saves
``{"net": encoder_state_dict}`` to ``--out``.

The encoder hyper-parameters MUST match those used during AE training *and* the
ones the LDM's ``cond_stage_model2`` builds, otherwise the state_dict will not
load cleanly. By default they are read from the AE training config JSON so the
three stages stay consistent.

Usage:
    python make_pretrained_encoder.py \
        --ckpt ./ckpt_dir/EEG_condition_dryrun/EEG_condition_dryrun_0.pth \
        --config ./config/Train_AE_dryrun.json \
        --out ./pretrain_models/EEGEncoder_dryrun.pth
"""
import os
import json
import argparse

import torch

from models.eeg_AE import eeg_encoder as Encoder


def build_encoder(cfg):
    return Encoder(
        in_seq=cfg["in_seq"],
        in_channels=cfg["in_channels"],
        real_channels=cfg.get("real_channels", None),
        out_channels=cfg["z_channels"],   # AE z_channels == encoder out_channels (77)
        out_seq=cfg["out_seq"],
        dims=cfg["dims"],
        shortcut=bool(cfg["shortcut"]),
        dropout=cfg["dropout"],
        groups=cfg["groups"],
        layer_mode=cfg["layer_mode"],
        block_mode=cfg["block_mode"],
        down_mode=cfg["down_mode"],
        pos_mode=cfg["pos_mode"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        dff_factor=cfg["dff_factor"],
        stride=cfg["stride"],
        skip_mode=cfg["skip_mode"],
        global_attn=bool(cfg.get("global_attn", 1)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="trained AE checkpoint (.pth with {'net': ...})")
    p.add_argument("--config", required=True, help="AE training config JSON (for encoder hyper-params)")
    p.add_argument("--out", required=True, help="output path for encoder-only weights (.pth)")
    args = p.parse_args()

    cfg = json.load(open(args.config))

    # Build a reference encoder so we can validate the extracted keys load cleanly.
    encoder = build_encoder(cfg)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    full_state = ckpt["net"]

    # Strip the "Encoder." prefix (len 8) from autoencoder keys.
    encoder_state_dict = {n[len("Encoder."):]: v for n, v in full_state.items() if n.startswith("Encoder.")}
    if not encoder_state_dict:
        raise RuntimeError("No 'Encoder.*' keys found in checkpoint; is this an AutoEncoder checkpoint?")

    missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
    print(f"extracted {len(encoder_state_dict)} encoder tensors")
    print(f"missing keys ({len(missing)}):", missing[:8], "..." if len(missing) > 8 else "")
    print(f"unexpected keys ({len(unexpected)}):", unexpected[:8], "..." if len(unexpected) > 8 else "")
    if missing or unexpected:
        print("WARNING: encoder hyper-params may not match the AE checkpoint "
              "(check n_layer / dims / stride / global_attn in --config).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"net": encoder_state_dict}, args.out)
    print("saved encoder ->", args.out)


if __name__ == "__main__":
    main()
