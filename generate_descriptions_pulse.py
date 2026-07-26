"""
generate_descriptions_pulse.py

Drop-in replacement for generate_descriptions.py that uses PULSE-7B
(PULSE-ECG/PULSE-7B, arXiv:2410.19008) instead of Qwen to caption the
rendered PTB-XL ECG images.

Why PULSE instead of Qwen
-------------------------
Qwen is a *general* vision-language model. It has never been trained to read a
12-lead ECG, so its "descriptions" are frequently hedged, generic, or wrong
(hence all the anti-hallucination regex scrubbing in generate_descriptions.py).
PULSE-7B is a LLaVA-1.6/Vicuna-7B model instruction-tuned on ECGInstruct (~1M
ECG-image instructions) and is state-of-the-art on ECGBench, *including the
PTB-XL test split*. Feeding the downstream MedTsLLM text pathway an ECG-literate
report instead of a generic caption is the single biggest quality lever here.

Interface difference you must know about
----------------------------------------
PULSE is a `llava_llama` architecture model. It does NOT load through
transformers' AutoModelForImageTextToText (the interface Qwen uses in
generate_descriptions.py). It must be loaded with the LLaVA library that PULSE
ships in its repo. Install it once:

    git clone https://github.com/AIMedLab/PULSE.git
    cd PULSE/LLaVA
    pip install -e ".[train]"
    pip install flash-attn --no-build-isolation   # optional, faster

Then point PYTHONPATH at that LLaVA checkout (or run from inside it), and this
script will import it automatically.

Output schema (backward compatible)
-----------------------------------
Writes <out> as a CSV with columns:

    path, description[, pulse_superclass, pulse_conf]

- `path` / `description` are byte-for-byte compatible with what
  datasets/ptbxl_qwen.py and datasets/ptbxl_pulse.py expect, so the rest of the
  pipeline is unchanged.
- `pulse_superclass` / `pulse_conf` are PULSE's *own* zero-shot 5-class guess
  (NORM/MI/STTC/CD/HYP) parsed from a trailing classification line. These are
  deliberately kept OUT of the `description` text (to avoid the classifier just
  copying PULSE's label) and are instead consumed by fuse_predictions.py, where
  they are combined with the vision/text branches under a validation-tuned
  weight. That gives you PULSE's SOTA classification signal without naive label
  leakage into the text modality.

Rendering note (important for accuracy)
---------------------------------------
PULSE was trained on standard clinical 12-lead layouts. Render the images that
you feed to PULSE with signal_to_image.py's `layout="grid"` (3x4 panel layout),
NOT the default vertical `layout="stack"`. A stacked one-row-per-lead plot is
out of PULSE's training distribution and measurably degrades its reading. See
prepare_ptbxl_images_pulse notes in the README section at the bottom of this file.

Usage
-----
    # same manifest.csv that prepare_ptbxl_images.py already writes
    python generate_descriptions_pulse.py \
        --manifest data/ptbxl_images/train/manifest.csv \
        --out      data/ptbxl_images/train/descriptions.csv \
        --style structured

    # repeat for val / test manifests
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import pandas as pd
from tqdm import tqdm

# Reuse the exact resumable orchestration + label-safe conventions already used
# for Qwen so behaviour (resume, incremental flush) is identical.
from generate_descriptions import already_done


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_LONG = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}


# ============================================================================
# Prompts (tuned for PULSE / ECGInstruct style)
# ============================================================================
# PULSE responds best to direct ECG-interpretation questions phrased the way
# ECGInstruct was built. We ask for a structured report *and* a final,
# machine-parseable superclass line that we peel off into its own column.

_SUPERCLASS_MENU = "; ".join(f"{k} ({SUPERCLASS_LONG[k]})" for k in SUPERCLASSES)

STRUCTURED_PROMPT = f"""You are reading a printed 12-lead ECG. Write a concise clinical report using exactly these labeled sections, each on its own line:

Rhythm/Rate:
Axis:
Intervals:
QRS/R-wave progression:
ST-T:
Impression:

Then, on a final separate line, output your single best diagnostic category for this ECG from this list: {_SUPERCLASS_MENU}. Use exactly this format:

Superclass: <ONE OF: {", ".join(SUPERCLASSES)}> | confidence: <low|medium|high>

Base every statement only on what is visible in the tracing. Use hedged language when a finding is unclear. Do not invent exact numeric values unless they are clearly readable."""

PARAGRAPH_PROMPT = f"""You are reading a printed 12-lead ECG. In 3-5 concise clinical sentences, describe the visible rhythm and rate, QRS morphology and R-wave progression, and any ST-segment or T-wave abnormalities, ending with a brief overall impression. Base everything only on what is visible; hedge when unclear; do not fabricate numbers.

Then, on a final separate line, output your single best diagnostic category from this list: {_SUPERCLASS_MENU}. Use exactly this format:

Superclass: <ONE OF: {", ".join(SUPERCLASSES)}> | confidence: <low|medium|high>"""

STYLE_PROMPTS = {"structured": STRUCTURED_PROMPT, "paragraph": PARAGRAPH_PROMPT}


# ============================================================================
# Output parsing
# ============================================================================
_SUPERCLASS_LINE_RE = re.compile(
    r"superclass\s*:\s*\*{0,2}\s*(NORM|MI|STTC|CD|HYP)\b"
    r"(?:.*?confidence\s*:\s*\*{0,2}\s*(low|medium|high))?",
    re.IGNORECASE | re.DOTALL,
)


def split_report_and_vote(text: str) -> tuple[str, str, str]:
    """Separate the clinical report body from PULSE's trailing superclass vote.

    Returns (description_body, superclass_or_empty, confidence_or_empty).
    The superclass line is removed from the body so the label does not leak into
    the text modality; it is surfaced only via fuse_predictions.py.
    """
    if not text:
        return "", "", ""
    text = str(text).strip()

    superclass, conf = "", ""
    m = _SUPERCLASS_LINE_RE.search(text)
    if m:
        superclass = m.group(1).upper()
        conf = (m.group(2) or "").lower()

    # Drop any line that mentions the machine-readable vote from the body.
    body_lines = []
    for line in text.splitlines():
        if re.match(r"\s*\*{0,2}superclass\s*:", line, re.IGNORECASE):
            continue
        body_lines.append(line)
    body = "\n".join(body_lines).strip()

    # Light cleanup: strip common assistant preambles / markdown bold.
    body = re.sub(r"^\s*(?:sure|certainly|here is|here's)\b.*?:\s*", "", body,
                  flags=re.IGNORECASE | re.DOTALL)
    body = body.replace("**", "").strip()
    return body, superclass, conf


# ============================================================================
# PULSE (LLaVA-1.6) model
# ============================================================================
class PulseCaptioner:
    """Thin wrapper around PULSE-7B using the LLaVA-1.6 inference path.

    Kept per-image (not batched): LLaVA-1.6 uses AnyRes tiling, so different
    images produce different numbers of visual tokens, which makes naive
    batching incorrect. For the ~21k PTB-XL records this is fine on one H100;
    if you need it faster, serve PULSE with vLLM (it supports llava-1.6) and
    swap generate() for an HTTP call -- the parsing above is unchanged.
    """

    def __init__(self, model_path: str = "PULSE-ECG/PULSE-7B",
                 conv_mode: str = "llava_v1", dtype: str = "float16",
                 load_8bit: bool = False, load_4bit: bool = False):
        try:
            import torch  # noqa: F401
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except ImportError as e:
            raise ImportError(
                "PULSE needs the LLaVA library it ships with. Install it:\n"
                "    git clone https://github.com/AIMedLab/PULSE.git\n"
                "    cd PULSE/LLaVA && pip install -e '.[train]'\n"
                "then run this script with that LLaVA checkout on PYTHONPATH."
            ) from e

        self.conv_mode = conv_mode
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                load_8bit=load_8bit,
                load_4bit=load_4bit,
            )
        )
        self.model.eval()

    def caption(self, image_path: str, prompt_text: str,
                max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        import torch
        from PIL import Image
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(self.model.device, dtype=self.model.dtype) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=self.model.dtype)

        # Build the image-conditioned prompt with the right special tokens.
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            image_token = DEFAULT_IMAGE_TOKEN
        qs = image_token + "\n" + prompt_text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                  return_tensors="pt")
            .unsqueeze(0).to(self.model.device)
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor if isinstance(image_tensor, list) else image_tensor,
                image_sizes=[image.size],
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        out = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return out.strip()


# ============================================================================
# Resumable orchestration (writes extra vote columns alongside description)
# ============================================================================
def generate_for_manifest(manifest_csv, out_csv, captioner, prompt_text,
                          max_new_tokens=512, temperature=0.0, limit=None):
    df = pd.read_csv(manifest_csv)
    if limit:
        df = df.iloc[:limit]

    done = already_done(out_csv)
    todo = df[~df["path"].isin(done)]
    print(f"[pulse] {len(done)} already done, {len(todo)} remaining (of {len(df)} total)")

    mode = "a" if done else "w"
    with open(out_csv, mode, newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["path", "description", "pulse_superclass", "pulse_conf"])

        for path in tqdm(todo["path"].tolist()):
            try:
                raw = captioner.caption(path, prompt_text,
                                        max_new_tokens=max_new_tokens,
                                        temperature=temperature)
            except Exception as e:
                print(f"[warn] failed on {path}: {e}")
                raw = ""
            body, superclass, conf = split_report_and_vote(raw)
            if not body:
                print(f"[warn] empty description for {path}")
            writer.writerow([path, body, superclass, conf])
            f.flush()

    print(f"[pulse] wrote {out_csv}")


# ============================================================================
# CLI
# ============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-path", default="PULSE-ECG/PULSE-7B")
    p.add_argument("--conv-mode", default="llava_v1")
    p.add_argument("--style", choices=["structured", "paragraph"], default="structured")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy/deterministic (recommended for a fixed dataset).")
    p.add_argument("--load-4bit", action="store_true", help="fit PULSE-7B on a smaller GPU")
    p.add_argument("--load-8bit", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke-test on the first N rows")
    args = p.parse_args()

    captioner = PulseCaptioner(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
    )
    generate_for_manifest(
        args.manifest, args.out, captioner,
        prompt_text=STYLE_PROMPTS[args.style],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
