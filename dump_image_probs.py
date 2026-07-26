"""
dump_image_probs.py

Run a trained image-classifier checkpoint (from train_image_classifier.py) over
a manifest and save per-sample class probabilities in manifest order, so
fuse_predictions.py can late-fuse them with PULSE's votes and/or the text path.

Usage
-----
    python dump_image_probs.py --resume runs/ptbxl_image_clf/best.pt \
        --manifest data/ptbxl_images/val/manifest.csv \
        --out runs/ptbxl_image_clf/val_probs.npz

    python dump_image_probs.py --resume runs/ptbxl_image_clf/best.pt \
        --manifest data/ptbxl_images/test/manifest.csv \
        --out runs/ptbxl_image_clf/test_probs.npz

Saves an .npz with:
    probs   : float32 [N, C]  softmax probabilities, manifest row order
    labels  : int64   [N]     ground-truth labels
    paths   : str     [N]     image paths (for path-based alignment)
    classes : str     [C]     class names in column order
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_image_classifier import (
    SignalImageDataset, build_model, build_transforms, TrainConfig,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True, help="path to best.pt from train_image_classifier.py")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.resume, map_location=device)
    class_names = ckpt["class_names"]
    n_classes = len(class_names)

    saved_cfg = ckpt.get("cfg", {})
    cfg = TrainConfig(**{k: v for k, v in saved_cfg.items()
                         if k in TrainConfig.__dataclass_fields__})

    model, data_cfg = build_model(cfg, n_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)

    eval_tf = build_transforms(cfg.img_size, data_cfg["mean"], data_cfg["std"], train=False)
    ds = SignalImageDataset(args.manifest, transform=eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_probs, all_labels = [], []
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            if cfg.channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=torch.bfloat16, enabled=True):
                logits = model(imgs)
            all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int64)
    paths = pd.read_csv(args.manifest)["path"].tolist()
    assert len(paths) == len(probs) == len(labels), "manifest / probs length mismatch"

    np.savez(args.out, probs=probs, labels=labels,
             paths=np.array(paths, dtype=object), classes=np.array(class_names, dtype=object))
    acc = (probs.argmax(1) == labels).mean()
    print(f"[dump_image_probs] wrote {args.out}  (N={len(probs)}, vision-only acc={acc:.4f})")


if __name__ == "__main__":
    main()
