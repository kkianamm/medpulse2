# Swapping Qwen → PULSE-7B for PTB-XL classification

PULSE-7B (`PULSE-ECG/PULSE-7B`, arXiv:2410.19008) is a LLaVA-1.6/Vicuna-7B model
instruction-tuned on ~1M ECG-image instructions (ECGInstruct) and is SOTA on
ECGBench, **including the PTB-XL test split**. It replaces the general-purpose
Qwen captioner. Two accuracy levers ship together here:

1. **Better text modality** — ECG-literate reports instead of generic captions,
   fed through the *existing* MedTsLLM text pathway unchanged.
2. **Decision-time late fusion** — PULSE's own zero-shot 5-class vote is combined
   with the vision branch under a validation-tuned weight (no label leakage into
   the training text).

## 0. Install PULSE's LLaVA (one time)

PULSE is a `llava_llama` model and does **not** load through Qwen's
`AutoModelForImageTextToText`. Install the LLaVA fork it ships with:

```bash
git clone https://github.com/AIMedLab/PULSE.git
cd PULSE/LLaVA && pip install -e ".[train]"
pip install flash-attn --no-build-isolation   # optional
```

Run the generator with that LLaVA checkout on `PYTHONPATH`.

## 1. Render images PULSE expects (grid layout, not stacked)

PULSE was trained on standard clinical 12-lead layouts. Render with
`layout="grid"` (3×4 panels), **not** the default vertical `layout="stack"`.
In `prepare_ptbxl_images.py` the `method_kwargs` passed to
`precompute_dataset_images(...)` should include `layout="grid"` for the PULSE
image set (keep a separate `--out-dir data/ptbxl_images_pulse`). Everything else
(mV inversion, manifest schema) is identical.

## 2. Generate PULSE descriptions (drop-in for generate_descriptions.py)

```bash
for split in train val test; do
  python generate_descriptions_pulse.py \
    --manifest data/ptbxl_images_pulse/$split/manifest.csv \
    --out      data/ptbxl_images_pulse/$split/descriptions.csv \
    --style structured
done
```

Output CSV is `path,description,pulse_superclass,pulse_conf`. The first two
columns are byte-compatible with the Qwen pipeline; the vote columns are only
used by fusion (step 4).

## 3. Train the text pathway on PULSE reports

```toml
[data]
dataset = "PTB-XL-PULSE"

[datasets.PTB-XL-PULSE]
pulse_descriptions_dir = "data/ptbxl_images_pulse"
merge_mode = "append"
drop_sections = ["Impression"]   # keep findings, drop PULSE's own dx sentence
max_chars = 600
desc_dropout = 0.5               # train-only modality dropout
```

```bash
python3 train.py configs/your_pulse_config.toml
```

## 4. Late-fuse vision branch + PULSE votes

```bash
# a) train / already-have the vision branch
python train_image_classifier.py \
  --train-manifest data/ptbxl_images_pulse/train/manifest.csv \
  --val-manifest   data/ptbxl_images_pulse/val/manifest.csv \
  --test-manifest  data/ptbxl_images_pulse/test/manifest.csv \
  --backbone convnext_tiny --out-dir runs/ptbxl_image_clf

# b) dump its probabilities in manifest order
python dump_image_probs.py --resume runs/ptbxl_image_clf/best.pt \
  --manifest data/ptbxl_images_pulse/val/manifest.csv  --out runs/ptbxl_image_clf/val_probs.npz
python dump_image_probs.py --resume runs/ptbxl_image_clf/best.pt \
  --manifest data/ptbxl_images_pulse/test/manifest.csv --out runs/ptbxl_image_clf/test_probs.npz

# c) fuse (weight + temperature tuned on val, reported on test)
python fuse_predictions.py \
  --val-probs  runs/ptbxl_image_clf/val_probs.npz \
  --test-probs runs/ptbxl_image_clf/test_probs.npz \
  --val-desc   data/ptbxl_images_pulse/val/descriptions.csv \
  --test-desc  data/ptbxl_images_pulse/test/descriptions.csv \
  --out runs/ptbxl_image_clf/fused_test_metrics.json
```

Add `--val-text-probs/--test-text-probs` (same `.npz` schema) to also fold the
MedTsLLM text-path probabilities into a 3-way fusion.

## Files added

| File | Purpose |
|------|---------|
| `generate_descriptions_pulse.py` | PULSE captioner, drop-in for the Qwen one |
| `datasets/ptbxl_pulse.py` | `PTB-XL-PULSE` dataset (subclass of the Qwen wrapper) |
| `dump_image_probs.py` | Save vision-branch probabilities in manifest order |
| `fuse_predictions.py` | Validation-tuned late fusion of vision + PULSE (+ text) |
