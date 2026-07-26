"""
fuse_predictions.py

Late-fuse the vision classifier's probabilities with PULSE-7B's own zero-shot
5-superclass votes, choosing the fusion weight and a vision temperature on the
VALIDATION split only, then reporting the fused TEST metrics.

Why this helps
--------------
PULSE-7B is state-of-the-art on ECGBench's PTB-XL split, so its guess carries
real diagnostic signal that a from-scratch image CNN on rendered plots does not
fully capture -- and vice versa (the CNN sees the exact rendered morphology).
Averaging two decorrelated, individually-decent predictors is one of the most
reliable accuracy wins in ML. We keep PULSE's label OUT of the training text
(no leakage) and only combine it here, at decision time, under a weight fit on
held-out validation data.

Inputs
------
  * vision probs (.npz from dump_image_probs.py) for val and test
  * PULSE descriptions.csv (with pulse_superclass[, pulse_conf]) for val and test
    -- written by generate_descriptions_pulse.py

Optional third branch
----------------------
  * --val-text-probs / --test-text-probs : .npz with the same schema, if you also
    dumped MedTsLLM/BiomedCoOp text-path probabilities. Fusion then searches a
    2-simplex of weights over {vision, text, pulse}.

Usage
-----
    python fuse_predictions.py \
        --val-probs  runs/ptbxl_image_clf/val_probs.npz \
        --test-probs runs/ptbxl_image_clf/test_probs.npz \
        --val-desc   data/ptbxl_images_pulse/val/descriptions.csv \
        --test-desc  data/ptbxl_images_pulse/test/descriptions.csv \
        --out runs/ptbxl_image_clf/fused_test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SUPERCLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]

# Probability mass placed on PULSE's chosen class by stated confidence; the
# remainder is spread uniformly over the other classes. Missing vote -> uniform.
CONF_MASS = {"high": 0.85, "medium": 0.70, "low": 0.55, "": 0.70}


def load_probs(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    classes = [str(c) for c in d["classes"].tolist()]
    return d["probs"].astype(np.float64), d["labels"].astype(np.int64), \
        [str(p) for p in d["paths"].tolist()], classes


def class_index_map(classes):
    """Map a superclass name -> column index in the probability array."""
    upper = [c.upper() for c in classes]
    if set(SUPERCLASS_ORDER).issubset(set(upper)):
        return {c: upper.index(c) for c in SUPERCLASS_ORDER}
    # Vision classifier stored numeric labels; assume SUPERCLASS_ORDER indexing.
    return {c: i for i, c in enumerate(SUPERCLASS_ORDER)}


def pulse_distribution(desc_csv, paths, n_classes, cls_idx, smoothing_floor=0.02):
    """Build [N, C] PULSE vote distributions aligned to `paths` order."""
    df = pd.read_csv(desc_csv)
    by_path = {str(r["path"]): r for _, r in df.iterrows()}
    dist = np.full((len(paths), n_classes), 1.0 / n_classes, dtype=np.float64)

    matched = 0
    for i, p in enumerate(paths):
        row = by_path.get(p)
        if row is None:
            continue
        vote = str(row.get("pulse_superclass", "") or "").strip().upper()
        if vote not in cls_idx:
            continue
        conf = str(row.get("pulse_conf", "") or "").strip().lower()
        mass = CONF_MASS.get(conf, CONF_MASS[""])
        d = np.full(n_classes, (1.0 - mass) / (n_classes - 1), dtype=np.float64)
        d[cls_idx[vote]] = mass
        # small floor keeps things well-conditioned in log space
        d = d + smoothing_floor
        dist[i] = d / d.sum()
        matched += 1

    print(f"[fuse] {desc_csv}: matched PULSE votes for {matched}/{len(paths)} samples")
    return dist


def temper(probs, T):
    """Temperature-scale a probability array in log space."""
    logp = np.log(np.clip(probs, 1e-9, 1.0)) / T
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(pred, labels, n_classes):
    avg = "binary" if n_classes == 2 else "macro"
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average=avg, zero_division=0)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-probs", required=True)
    p.add_argument("--test-probs", required=True)
    p.add_argument("--val-desc", required=True)
    p.add_argument("--test-desc", required=True)
    p.add_argument("--val-text-probs", default=None)
    p.add_argument("--test-text-probs", default=None)
    p.add_argument("--out", default="fused_test_metrics.json")
    args = p.parse_args()

    v_probs, v_labels, v_paths, classes = load_probs(args.val_probs)
    t_probs, t_labels, t_paths, _ = load_probs(args.test_probs)
    n_classes = v_probs.shape[1]
    cls_idx = class_index_map(classes)

    v_pulse = pulse_distribution(args.val_desc, v_paths, n_classes, cls_idx)
    t_pulse = pulse_distribution(args.test_desc, t_paths, n_classes, cls_idx)

    # Optional text branch.
    v_text = t_text = None
    if args.val_text_probs and args.test_text_probs:
        v_text, _, vt_paths, _ = load_probs(args.val_text_probs)
        t_text, _, tt_paths, _ = load_probs(args.test_text_probs)
        assert vt_paths == v_paths and tt_paths == t_paths, \
            "text-probs must be in the same manifest order as vision probs"

    print("\n[fuse] individual val accuracy:")
    print(f"   vision : {accuracy_score(v_labels, v_probs.argmax(1)):.4f}")
    print(f"   pulse  : {accuracy_score(v_labels, v_pulse.argmax(1)):.4f}")
    if v_text is not None:
        print(f"   text   : {accuracy_score(v_labels, v_text.argmax(1)):.4f}")

    temps = [0.5, 0.75, 1.0, 1.5, 2.0]
    best = {"val_acc": -1.0}

    if v_text is None:
        # 2-branch: fused = a*vision(T) + (1-a)*pulse
        for T in temps:
            vt = temper(v_probs, T)
            for a in np.linspace(0.0, 1.0, 51):
                fused = a * vt + (1.0 - a) * v_pulse
                acc = accuracy_score(v_labels, fused.argmax(1))
                if acc > best["val_acc"]:
                    best = {"val_acc": acc, "T": float(T), "w": [float(a), float(1 - a)]}
        vt = temper(t_probs, best["T"])
        fused_test = best["w"][0] * vt + best["w"][1] * t_pulse
    else:
        # 3-branch: search weight simplex over {vision(T), text, pulse}
        grid = np.linspace(0.0, 1.0, 21)
        for T in temps:
            vt = temper(v_probs, T)
            for wv, wx in itertools.product(grid, grid):
                if wv + wx > 1.0:
                    continue
                wp = 1.0 - wv - wx
                fused = wv * vt + wx * v_text + wp * v_pulse
                acc = accuracy_score(v_labels, fused.argmax(1))
                if acc > best["val_acc"]:
                    best = {"val_acc": acc, "T": float(T),
                            "w": [float(wv), float(wx), float(wp)]}
        vt = temper(t_probs, best["T"])
        fused_test = best["w"][0] * vt + best["w"][1] * t_text + best["w"][2] * t_pulse

    vision_test = evaluate(t_probs.argmax(1), t_labels, n_classes)
    fused_metrics = evaluate(fused_test.argmax(1), t_labels, n_classes)

    result = {
        "best_val_acc": best["val_acc"],
        "vision_temperature": best["T"],
        "fusion_weights": best["w"],
        "weight_order": ["vision", "pulse"] if v_text is None
        else ["vision", "text", "pulse"],
        "test_vision_only": vision_test,
        "test_fused": fused_metrics,
    }
    print("\n[fuse] chosen on val:", {k: result[k] for k in
          ("best_val_acc", "vision_temperature", "fusion_weights", "weight_order")})
    print("[fuse] TEST vision-only:", vision_test)
    print("[fuse] TEST fused      :", fused_metrics)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fuse] wrote {args.out}")


if __name__ == "__main__":
    main()
