#!/usr/bin/env python3
"""
Utility to inspect and update regression border plots offline.

Given an analysis name, the script reloads the cached autocorrelation data saved
by `C1_dRSA_run.py`, overlays the original border plus a user-configured
threshold (edited directly in this file), and writes comparison figures to a
dedicated inspection folder.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Customize these values to try different thresholds per model label.
# Any label absent from the dictionary falls back to DEFAULT_NEW_BORDER.
# ---------------------------------------------------------------------------
DEFAULT_NEW_BORDER = 0.05
MODEL_THRESHOLD_OVERRIDES = {
    "Envelope": 0.05,
    "Phoneme Voicing": 0.08,
    "Word Frequency": 0.05,
    "GloVe": 0.1,
    "GloVe Norm": 0.1, 
    "GPT Next-Token": 0.25,
    "GPT Surprisal": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot dRSA regression borders with custom thresholds."
    )
    parser.add_argument("analysis_name", help="Name of the analysis folder under results/.")
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing analysis folders (default: results/).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional explicit path to a *_metadata.json file. "
        "When omitted, the script searches single_subjects/ for a unique metadata file.",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=DEFAULT_NEW_BORDER,
        help="Override DEFAULT_NEW_BORDER without editing the file (optional).",
    )
    return parser.parse_args()


def slugify(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "model"


def locate_metadata(analysis_root: Path, metadata_override: Path | None) -> Path:
    if metadata_override:
        return metadata_override
    metadata_files = sorted((analysis_root / "single_subjects").glob("*_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(
            f"No metadata files found under {analysis_root / 'single_subjects'}."
        )
    if len(metadata_files) > 1:
        raise RuntimeError(
            "Multiple metadata files found. "
            "Please specify one via --metadata:\n" + "\n".join(str(p) for p in metadata_files)
        )
    return metadata_files[0]


def compute_border(lags: np.ndarray, lag_corr: np.ndarray, threshold: float) -> int:
    mask = np.abs(lag_corr) >= threshold
    if not np.any(mask):
        return 0
    return int(np.max(np.abs(lags[mask])))


def replot_border(
    lags: np.ndarray,
    lag_corr: np.ndarray,
    old_threshold: float,
    old_border: int,
    new_threshold: float,
    new_border: int,
    label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lags, lag_corr, color="tab:blue", linewidth=2, label="Autocorrelation")
    ax.axhline(old_threshold, color="tab:orange", linestyle="--", label=f"Old τ={old_threshold}")
    ax.axhline(new_threshold, color="tab:purple", linestyle="-.", label=f"New τ={new_threshold}")
    ax.axvline(old_border, color="tab:green", linestyle=":", label=f"Old border ±{old_border}")
    ax.axvline(-old_border, color="tab:green", linestyle=":")
    ax.axvline(new_border, color="tab:red", linestyle="--", label=f"New border ±{new_border}")
    ax.axvline(-new_border, color="tab:red", linestyle="--")
    ax.set_title(f"Regression Border Update | {label}")
    ax.set_xlabel("Lag (time points)")
    ax.set_ylabel("Mean correlation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    analysis_root = results_root / args.analysis_name
    if not analysis_root.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_root}")

    metadata_path = locate_metadata(analysis_root, args.metadata)
    with metadata_path.open() as f:
        metadata = json.load(f)

    regression_info = metadata.get("regression") or {}
    border_plots = regression_info.get("border_plots")
    if not border_plots:
        raise RuntimeError(
            "Metadata does not contain regression border plots. "
            "Rerun C1 with --plot-regression-borders enabled."
        )

    print(f"Loaded metadata: {metadata_path}")
    label_thresholds = {
        entry["label"]: MODEL_THRESHOLD_OVERRIDES.get(entry["label"], args.default_threshold)
        for entry in border_plots
    }
    suffix = "_".join(f"{slugify(label)}-{val:.3f}" for label, val in label_thresholds.items())

    for entry in border_plots:
        label = entry["label"]
        new_threshold = label_thresholds[label]
        png_path = Path(entry["path"])
        data_path = Path(entry.get("data_path") or png_path.with_suffix(".npz"))
        if not data_path.exists():
            print(f"[skip] Missing data file for {label}: {data_path}")
            continue
        data = np.load(data_path)
        lags = data["lags"]
        lag_corr = data["lag_corr"]
        old_threshold = float(data["threshold"])
        old_border = int(data["border_lag"])
        new_border = compute_border(lags, lag_corr, new_threshold)
        inspect_dir = png_path.parent / f"inspect_borders_{suffix}"
        inspect_dir.mkdir(parents=True, exist_ok=True)
        output_path = inspect_dir / f"{png_path.stem}_new_{new_threshold:.3f}.png"
        replot_border(
            lags,
            lag_corr,
            old_threshold,
            old_border,
            new_threshold,
            new_border,
            label,
            output_path,
        )
        print(
            f"[done] {label}: old τ={old_threshold}, old border={old_border}, "
            f"new τ={new_threshold}, new border={new_border} → {output_path}"
        )


if __name__ == "__main__":
    main()
