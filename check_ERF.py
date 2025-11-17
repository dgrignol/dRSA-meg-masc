"""
Plot the median (across subjects) magnetometer evoked response to all words.

The script reuses the epoching pipeline from ``check_decoding.py`` to segment
each dataset, averages all word epochs per subject, then aggregates them by
taking the median across subjects. The resulting evoked response is visualised
with spatially-coloured sensor traces, topographic maps at selected time
points, and a shaded area indicating the global field power (GFP).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np

import check_decoding as decoding


DEFAULT_TOPOMAP_TIMES = (-0.10, 0.0, 0.10, 0.20, 0.30, 0.40)
UNIT_SCALE_FT = 1e15  # Tesla -> femtoTesla
UNIT_LABEL = "fT"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the median magnetometer response to all word epochs."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        metavar="SUB",
        help=(
            "Subject identifiers to include (e.g., 0001 0002). "
            "Defaults to all entries in participants.tsv."
        ),
    )
    parser.add_argument(
        "--channel-type",
        default="mag",
        choices=["mag"],
        help="Sensor type to plot (currently only magnetometers are supported).",
    )
    parser.add_argument(
        "--topomap-times",
        nargs="*",
        type=float,
        metavar="TIME",
        help="Time points (in seconds) for the topographic snapshots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/investigate_envelope/median_ERF.png"),
        help="Path where the figure will be saved.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution of the saved figure.",
    )
    parser.add_argument(
        "--evoked-out",
        type=Path,
        default=None,
        help="Optional path to save the median evoked (FIF format).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    return parser.parse_args()


def _subject_word_evoked(
    subject: str, ch_type: str
) -> Optional[Tuple[mne.Evoked, int]]:
    """Return the per-subject word evoked and the number of contributing epochs."""
    epochs = decoding._get_epochs(subject)
    if epochs is None:
        return None

    words = epochs["is_word"]
    if len(words) == 0:
        return None

    picks = mne.pick_types(words.info, meg=ch_type, eeg=False, eog=False, ecg=False)
    if len(picks) == 0:
        return None

    words = words.copy()
    words.pick(picks)
    words.info["bads"] = []  # keep channel set identical across subjects

    evoked = words.average()
    evoked.comment = f"{subject}_words"
    return evoked, len(words)


def _median_evoked(evokeds: Sequence[mne.Evoked], nave: int) -> mne.Evoked:
    """Median-combine evoked objects that share the same metadata."""
    reference = evokeds[0]
    stacked = list()
    for evoked in evokeds:
        if evoked.ch_names != reference.ch_names:
            evoked = evoked.copy().reorder_channels(reference.ch_names)
        stacked.append(evoked.data)
    stacked = np.stack(stacked, axis=0)
    median = np.median(stacked, axis=0)
    info = reference.info.copy()
    comment = "Median across subjects (word epochs)"
    median_evoked = mne.EvokedArray(
        median,
        info=info,
        tmin=reference.times[0],
        nave=nave,
        comment=comment,
    )
    return median_evoked


def _sensor_colors(evoked: mne.Evoked) -> List:
    """Color sensors based on their planar angle to mimic spatial coloring."""
    layout = mne.channels.find_layout(evoked.info, ch_type="mag")
    if layout is None:
        return ["#1f77b4"] * len(evoked.ch_names)

    positions = layout.pos[:, :2]
    center = positions.mean(axis=0)
    angles = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
    angles = (angles - angles.min()) / (angles.ptp() if angles.ptp() else 1.0)
    cmap = plt.cm.hsv
    color_lookup = {name: cmap(angle) for name, angle in zip(layout.names, angles)}
    return [color_lookup.get(ch, "#444444") for ch in evoked.ch_names]


def _plot_erf(
    evoked: mne.Evoked,
    topomap_times: Iterable[float],
    nave: int,
    n_subjects: int,
) -> plt.Figure:
    """Create the figure with topomaps, sensor traces, and GFP shading."""
    data = evoked.data * UNIT_SCALE_FT
    times = evoked.times
    gfp = np.sqrt((data**2).mean(axis=0))

    fig = plt.figure(figsize=(11, 6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.2, 4.0], hspace=0.3)
    topo_gs = outer[0].subgridspec(1, len(topomap_times), wspace=0.05)

    vmax = np.percentile(np.abs(data), 99.0)
    vmax = vmax if vmax else np.max(np.abs(data))
    vmax = max(vmax, 1e-3)

    top_axes = [fig.add_subplot(topo_gs[0, idx]) for idx in range(len(topomap_times))]
    im = None
    for ax, time_point in zip(top_axes, topomap_times):
        idx = np.argmin(np.abs(times - time_point))
        im, _ = mne.viz.plot_topomap(
            data[:, idx],
            evoked.info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            sensors=False,
            outlines="head",
            contours=6,
        )
        ax.set_title(f"{time_point:+0.3f} s", fontsize=9)
    if im is not None:
        cbar = fig.colorbar(im, ax=top_axes, shrink=0.6, pad=0.02)
        cbar.set_label(UNIT_LABEL)

    trace_ax = fig.add_subplot(outer[1])
    trace_ax.set_title("Median evoked response (all word epochs)")
    trace_ax.set_xlabel("Time (s)")
    trace_ax.set_ylabel(f"Magnetic field ({UNIT_LABEL})")
    trace_ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    for time_point in topomap_times:
        trace_ax.axvline(time_point, color="0.85", linewidth=0.7, zorder=0)

    colors = _sensor_colors(evoked)
    for idx, color in enumerate(colors):
        trace_ax.plot(times, data[idx], color=color, linewidth=0.8, alpha=0.9)
    trace_ax.set_xlim(times[0], times[-1])

    gfp_ax = trace_ax.twinx()
    gfp_ax.fill_between(
        times, 0.0, gfp, color="0.85", alpha=0.8, linewidth=0, zorder=0
    )
    gfp_ax.plot(times, gfp, color="0.3", linewidth=1.2, label="GFP")
    gfp_ax.set_ylim(0.0, gfp.max() * 1.2 if gfp.max() else 1.0)
    gfp_ax.set_yticks([])
    gfp_ax.set_ylabel("GFP", color="0.3")
    gfp_ax.spines["right"].set_visible(False)
    gfp_ax.spines["top"].set_visible(False)
    gfp_ax.set_zorder(trace_ax.get_zorder() - 1)

    trace_ax.text(
        0.01,
        0.92,
        f"Nave={nave}",
        transform=trace_ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
    )
    fig.suptitle(f"Magnetometers ({len(evoked.ch_names)} channels, n={n_subjects})")
    return fig


def main() -> None:
    args = _parse_args()
    topomap_times = (
        tuple(args.topomap_times) if args.topomap_times else DEFAULT_TOPOMAP_TIMES
    )

    subject_list = args.subjects if args.subjects else list(decoding.subjects)
    evokeds = []
    total_epochs = 0
    missing = []
    for subject in subject_list:
        out = _subject_word_evoked(subject, args.channel_type)
        if out is None:
            missing.append(subject)
            continue
        evoked, n_epochs = out
        total_epochs += n_epochs
        evokeds.append(evoked)

    if not evokeds:
        raise RuntimeError("No subjects with word epochs were found.")

    median_evoked = _median_evoked(evokeds, nave=total_epochs)
    fig = _plot_erf(
        median_evoked,
        topomap_times=topomap_times,
        nave=total_epochs,
        n_subjects=len(evokeds),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists. Use --overwrite to replace it."
        )
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    if args.evoked_out:
        if args.evoked_out.exists() and not args.overwrite:
            raise FileExistsError(
                f"{args.evoked_out} already exists. Use --overwrite to replace it."
            )
        median_evoked.save(args.evoked_out, overwrite=args.overwrite)

    if missing:
        print("Skipped subjects without usable data:", ", ".join(missing))
    print(f"Saved figure to {args.output}")
    if args.evoked_out:
        print(f"Saved evoked file to {args.evoked_out}")


if __name__ == "__main__":
    main()
