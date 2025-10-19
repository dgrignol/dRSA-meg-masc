"""
Utility script to inspect a specific time window of the resampled sentence and
concatenation masks.

The main resampling CLI produces a single diagnostic figure, but it is often
useful to zoom into other regions without re-running the heavy resampling step.
This script shows how to do that: it loads the already saved masks, selects an
interval, and delegates the plotting to `plot_masks_comparison`.
"""

from pathlib import Path

import numpy as np

from functions.generic_helpers import read_repository_root
from resample_concatenated_data import plot_masks_comparison


def main() -> None:
    """Load masks for a subject and export a zoomed-in comparison plot."""

    repo_root = Path(read_repository_root())

    # Keep the subject token in a single variable so changing it is trivial.
    subject = "sub-01"
    concat_dir = repo_root / "derivatives" / "preprocessed" / subject / "concatenated"

    # Load the original and resampled sentence/concatenation masks.
    sentence_orig = np.load(concat_dir / f"{subject}_concatenated_sentence_mask.npy")
    sentence_res = np.load(concat_dir / f"{subject}_concatenated_sentence_mask_100Hz.npy")
    boundary_orig = np.load(concat_dir / f"{subject}_concatenation_boundaries_mask.npy")
    boundary_res = np.load(concat_dir / f"{subject}_concatenation_boundaries_mask_100Hz.npy")

    # Window to visualise; adjust to inspect different portions.
    start_sec = 100
    end_sec = 1000

    # Create a comparison figure showing the chosen interval.
    plot_masks_comparison(
        sentence_orig,
        sentence_res,
        boundary_orig,
        boundary_res,
        original_rate=1000,
        target_rate=100,
        start_sec=start_sec,
        end_sec=end_sec,
        output_path=Path(f"results/{subject}_mask_resampling_{start_sec}-{end_sec}s.png"),
        title_prefix=f"{subject} ",
    )


if __name__ == "__main__":
    main()
