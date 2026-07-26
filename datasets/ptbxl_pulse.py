"""
datasets/ptbxl_pulse.py

PTB-XL classification dataset augmented with PULSE-7B ECG reports.

This subclasses PTBXLQwenClassificationDataset and inherits ALL of its
alignment, section-filtering, length-capping, label cross-checking and
train-only modality-dropout logic unchanged -- because PULSE's descriptions.csv
uses the exact same {path, description} schema the Qwen wrapper already reads
(the extra pulse_superclass/pulse_conf columns are simply ignored here; they are
consumed separately by fuse_predictions.py).

The only differences are cosmetic-but-useful:
  * a distinct dataset key ("PTB-XL-PULSE") so your TOML config is unambiguous;
  * it reads `pulse_descriptions_dir` from [datasets.PTB-XL-PULSE], falling back
    to `qwen_descriptions_dir` if you reused the old key name.

Example config
--------------
    [data]
    dataset = "PTB-XL-PULSE"

    [datasets.PTB-XL-PULSE]
    pulse_descriptions_dir = "data/ptbxl_images_pulse"
    merge_mode = "append"
    drop_sections = ["Impression"]   # keep the findings, drop PULSE's own dx text
    max_chars = 600
    desc_dropout = 0.5               # train-only modality dropout, regularizes text over-trust
"""

from __future__ import annotations

from typing import Any

from .ptbxl_qwen import PTBXLQwenClassificationDataset, _cfg_get


class PTBXLPulseClassificationDataset(PTBXLQwenClassificationDataset):
    """PTB-XL classification dataset augmented with PULSE-7B ECG reports."""

    # Sensible PULSE-specific default output directory.
    qwen_descriptions_dir = "data/ptbxl_images_pulse"

    def _refresh_qwen_config(self) -> None:
        # First let the parent load everything (merge_mode, drop_sections, etc.).
        super()._refresh_qwen_config()

        # Then allow a PULSE-named key to override the descriptions directory,
        # so configs read naturally without forcing the "qwen_" prefix.
        dc = self._get_qwen_dataset_config()
        pulse_dir = _cfg_get(dc, "pulse_descriptions_dir", None)
        if pulse_dir is not None:
            self.qwen_descriptions_dir = str(pulse_dir)


ptbxl_pulse_datasets = {
    "classification": PTBXLPulseClassificationDataset,
}
