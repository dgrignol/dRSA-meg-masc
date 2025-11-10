#!/usr/bin/env python3
"""
Standalone diagnostics for GPT next-token logprob-SVD pipeline outputs.

Reads only saved artifacts (.npy/.json/.pkl), produces matplotlib plots, and
prints summary stats. No GPU or Hugging Face models required.

Diagnostics:
1) Global SVD explained variance vs components
2) Per-story temporal L2 norm of reduced token features
3) Per-story surprisal histogram with mean/±1 SD markers
4) Cumulative probability vs retained K (top-K) — uses saved stats if present,
   otherwise approximates from metadata topk_mass/topk_cap per story.
5) Cache hit summary — estimates reuse counts per story by scanning subject
   concatenation metadata; plots bar chart of estimated hits.

Outputs:
- All global plots under --plots-root (default: derivatives/Models/gpt_next/plots)
- Per-story plots saved in each story cache folder.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import joblib

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


LOGGER = logging.getLogger("gpt_next_diagnostics")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnostics for GPT next-token logprob-SVD artifacts")
    p.add_argument(
        "--story-cache-root",
        type=Path,
        default=Path("derivatives/Models/gpt_next/story_cache"),
        help="Folder containing per-story caches (reduced_tokens.npy, surprisal.npy, cache_metadata.json)",
    )
    p.add_argument(
        "--global-svd",
        type=Path,
        default=Path("derivatives/Models/gpt_next/global_svd_basis.pkl"),
        help="Path to global TruncatedSVD model (joblib pkl)",
    )
    p.add_argument(
        "--plots-root",
        type=Path,
        default=Path("derivatives/Models/gpt_next/plots"),
        help="Directory to store global plots",
    )
    p.add_argument(
        "--stories",
        nargs="+",
        default=None,
        help="Restrict per-story plots to these story IDs (folder names under story_cache_root)",
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional filter of subjects (for cache summary), e.g. 01 02 or sub-01",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Downsample tokens when plotting long sequences (>0 means keep at most this many points)",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------


def _safe_load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        LOGGER.warning("Failed to read %s: %s", path, e)
    return None


def _safe_load_npy(path: Path) -> Optional[np.ndarray]:
    try:
        if path.exists():
            return np.load(path)
    except Exception as e:
        LOGGER.warning("Failed to load %s: %s", path, e)
    return None


def _list_story_dirs(root: Path, restrict: Optional[Sequence[str]]) -> List[Path]:
    if not root.exists():
        LOGGER.warning("Story cache root missing: %s", root)
        return []
    all_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if restrict:
        restrict_set = set(restrict)
        return [p for p in all_dirs if p.name in restrict_set]
    return all_dirs


def _normalise_subject(label: str) -> str:
    s = label.strip()
    if s.startswith("sub-"):
        return s
    try:
        return f"sub-{int(s):02d}"
    except Exception:
        return s


# --------------------------------------------------------------------------------------
# 1) Global SVD explained variance
# --------------------------------------------------------------------------------------


def plot_global_explained_variance(global_svd_path: Path, plots_root: Path) -> Optional[Path]:
    if not global_svd_path.exists():
        LOGGER.warning("Global SVD basis missing: %s", global_svd_path)
        return None
    try:
        svd = joblib.load(global_svd_path)
    except Exception as e:
        LOGGER.warning("Failed to load SVD at %s: %s", global_svd_path, e)
        return None

    ratio = getattr(svd, "explained_variance_ratio_", None)
    if ratio is None:
        LOGGER.warning("SVD object lacks explained_variance_ratio_: %s", global_svd_path)
        return None

    csum = np.cumsum(np.asarray(ratio, dtype=float))
    perc = csum * 100.0
    k90 = int(np.searchsorted(csum, 0.90) + 1) if csum.size > 0 else 0

    plots_root.mkdir(parents=True, exist_ok=True)
    out_path = plots_root / "global_explained_variance.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(perc) + 1), perc, lw=1.5)
    ax.axhline(90.0, color="tab:red", linestyle="--", linewidth=1.0, label="90%")
    if k90 > 0:
        ax.axvline(k90, color="tab:green", linestyle=":", linewidth=1.0)
        ax.annotate(f"90% at k={k90}", xy=(k90, 90), xytext=(k90 + 2, min(98, max(92, perc[min(k90 - 1, len(perc) - 1)] + 2))),
                    arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    ax.set_xlabel("Components (k)")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Global SVD: Cumulative Explained Variance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved global explained variance plot: %s", out_path)
    return out_path


# --------------------------------------------------------------------------------------
# 2) Per-story L2 norm over time
# --------------------------------------------------------------------------------------


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if x.size == 0:
        return x
    w = min(window, x.size)
    kernel = np.ones(w, dtype=np.float64) / float(w)
    y = np.convolve(x.astype(np.float64), kernel, mode="same")
    return y.astype(np.float32)


def plot_story_l2_norms(story_dirs: Sequence[Path], max_samples: int) -> List[Path]:
    out_paths: List[Path] = []
    for sdir in story_dirs:
        rpath = sdir / "reduced_tokens.npy"
        arr = _safe_load_npy(rpath)
        if arr is None:
            LOGGER.warning("Missing reduced tokens for %s", sdir.name)
            continue
        if arr.ndim != 2:
            LOGGER.warning("Unexpected shape for %s: %s", rpath, arr.shape)
            continue
        # Expect (components, tokens)
        l2 = np.linalg.norm(np.asarray(arr, dtype=np.float32), axis=0)

        # Optional downsampling to keep plot size manageable
        idx = np.arange(l2.size)
        if max_samples > 0 and l2.size > max_samples:
            step = int(np.ceil(l2.size / max_samples))
            idx = np.arange(0, l2.size, step, dtype=int)

        smoothed = _rolling_mean(l2, window=50)

        out_path = sdir / "plot_l2norm.png"
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(idx, l2[idx], lw=0.6, label="L2 norm")
        ax.plot(idx, smoothed[idx], lw=1.0, color="tab:orange", label="Smoothed (w=50)")
        ax.set_xlabel("Token index")
        ax.set_ylabel("L2 norm")
        ax.set_title(f"L2 Norm over Time – {sdir.name}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        LOGGER.info("Saved L2 norm plot for %s: %s", sdir.name, out_path)
        out_paths.append(out_path)
    return out_paths


# --------------------------------------------------------------------------------------
# 3) Per-story surprisal histogram
# --------------------------------------------------------------------------------------


def plot_story_surprisal_hist(story_dirs: Sequence[Path], max_samples: int) -> List[Path]:
    out_paths: List[Path] = []
    for sdir in story_dirs:
        spath = sdir / "surprisal.npy"
        arr = _safe_load_npy(spath)
        if arr is None:
            LOGGER.warning("Missing surprisal for %s", sdir.name)
            continue

        x = arr.astype(np.float32).reshape(-1)
        # Optional downsample before plotting if huge
        if max_samples > 0 and x.size > max_samples:
            idx = np.linspace(0, x.size - 1, num=max_samples, dtype=int)
            x = x[idx]

        n_nans = int(np.count_nonzero(~np.isfinite(x)))
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            LOGGER.warning("No finite surprisal values for %s", sdir.name)
            continue
        mu = float(np.mean(finite))
        sd = float(np.std(finite))
        LOGGER.info("%s surprisal: mean=%.3f, sd=%.3f, NaNs=%d", sdir.name, mu, sd, n_nans)

        out_path = sdir / "plot_surprisal_hist.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(finite, bins=50, alpha=0.7, color="tab:blue", edgecolor="white")
        ax.axvline(mu, color="tab:red", linestyle="--", label=f"mean={mu:.2f}")
        ax.axvline(mu - sd, color="tab:orange", linestyle=":", label=f"±1 SD")
        ax.axvline(mu + sd, color="tab:orange", linestyle=":")
        ax.set_xlabel("Surprisal (bits)")
        ax.set_ylabel("Count")
        ax.set_title(f"Surprisal Histogram – {sdir.name}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        LOGGER.info("Saved surprisal histogram for %s: %s", sdir.name, out_path)
        out_paths.append(out_path)
    return out_paths


# --------------------------------------------------------------------------------------
# 4) Cumulative probability vs K (top-K)
# --------------------------------------------------------------------------------------


def _load_topk_arrays(sdir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Try to load per-token cumulative mass arrays if present.

    Supports a few possible filenames. Returns (kept_k, cumulative_mass_matrix) or (None, None)
    where cumulative_mass_matrix has shape (tokens, <=K) if available.
    """
    # Common possibilities
    for fname in ("topk_stats.npz",):
        p = sdir / fname
        if p.exists():
            try:
                with np.load(p) as z:
                    kept_k = z.get("kept_k")
                    cum_mass = z.get("cumulative_mass")
                    return kept_k, cum_mass
            except Exception as e:
                LOGGER.debug("Failed to load %s: %s", p, e)
    # Separate arrays
    kept = sdir / "kept_k.npy"
    mass = sdir / "cumulative_mass.npy"
    kept_k = _safe_load_npy(kept)
    cum_mass = _safe_load_npy(mass)
    return kept_k, cum_mass


def plot_topk_mass_curve(story_dirs: Sequence[Path], plots_root: Path, max_samples: int) -> Optional[Path]:
    if not story_dirs:
        LOGGER.warning("No story caches available for top-K diagnostic.")
        return None

    # Gather per-story arrays or fall back to approximation
    per_story_curves: List[np.ndarray] = []
    per_story_caps: List[int] = []
    used_actual = False
    for sdir in story_dirs:
        meta = _safe_load_json(sdir / "cache_metadata.json") or {}
        cap = int(meta.get("topk_cap", 4096))
        mass_cut = float(meta.get("topk_mass", 0.99))
        kept_k, cum_mass = _load_topk_arrays(sdir)
        if cum_mass is not None and cum_mass.ndim == 2 and cum_mass.size > 0:
            # cum_mass: tokens x K
            used_actual = True
            X = cum_mass.astype(np.float32)
            # Randomly subsample tokens if large
            if max_samples > 0 and X.shape[0] > max_samples:
                idx = np.random.default_rng(0).choice(X.shape[0], size=max_samples, replace=False)
                X = X[idx]
            curve = np.nanmean(X, axis=0)
            per_story_curves.append(curve)
            per_story_caps.append(int(X.shape[1]))
        else:
            # Approximate: ensure mass reaches mass_cut at cap, monotonic concave
            k = np.arange(0, cap + 1, dtype=np.float32)
            # Smooth exponential rise: m(k) = 1 - (1 - mass_cut) ** (k / cap)
            # Guarantees m(0)=0, m(cap)=mass_cut
            with np.errstate(over="ignore", invalid="ignore"):
                curve = 1.0 - np.power(1.0 - float(mass_cut), k / float(max(1, cap)))
            per_story_curves.append(curve.astype(np.float32))
            per_story_caps.append(cap + 1)

    if not per_story_curves:
        LOGGER.warning("No top-K data available; skipping top-K plot.")
        return None

    k_max = min(4096, int(max(per_story_caps)))
    # Align curves to common length by padding with NaN then nanmean
    aligned = np.full((len(per_story_curves), k_max), np.nan, dtype=np.float32)
    for i, c in enumerate(per_story_curves):
        L = min(len(c), k_max)
        aligned[i, :L] = c[:L]
    avg_curve = np.nanmean(aligned, axis=0)

    plots_root.mkdir(parents=True, exist_ok=True)
    out_path = plots_root / "topk_mass_curve.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(avg_curve.size), avg_curve, lw=1.5, color="tab:blue")
    ax.axhline(0.99, color="tab:red", linestyle="--", linewidth=1.0, label="0.99 mass")
    ax.set_xlabel("Retained K")
    ax.set_ylabel("Average cumulative probability")
    title_suffix = "(actual)" if used_actual else "(approx from metadata)"
    ax.set_title(f"Cumulative probability vs retained K {title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved top-K mass curve: %s", out_path)
    return out_path


# --------------------------------------------------------------------------------------
# 5) Cache hit summary (estimated)
# --------------------------------------------------------------------------------------


def _iter_subject_concat_meta(repo_root: Path, subjects: Optional[Sequence[str]]) -> Iterable[Tuple[str, dict]]:
    preproc_root = repo_root / "derivatives" / "preprocessed"
    if not preproc_root.exists():
        return []
    subj_dirs = sorted([p for p in preproc_root.glob("sub-*") if p.is_dir()])
    subj_filter = set(_normalise_subject(s) for s in subjects) if subjects else None
    for sdir in subj_dirs:
        sid = sdir.name
        if subj_filter and sid not in subj_filter:
            continue
        meta_path = sdir / "concatenated" / f"{sid}_concatenation_metadata.json"
        if not meta_path.exists():
            continue
        meta = _safe_load_json(meta_path)
        if meta is None:
            continue
        yield sid, meta


def _normalise_task(label) -> str:
    if isinstance(label, str):
        s = label.strip()
        return s if s.startswith("task-") else f"task-{s}"
    try:
        return f"task-{int(label)}"
    except Exception:
        return str(label)


def plot_cache_hit_summary(repo_root: Path, story_dirs: Sequence[Path], plots_root: Path, subjects: Optional[Sequence[str]]) -> Optional[Path]:
    if not story_dirs:
        LOGGER.warning("No story caches available for cache-hit summary.")
        return None

    # Estimate usage from concatenation metadata across subjects
    usage_counts: Dict[str, int] = {sdir.name: 0 for sdir in story_dirs}
    for sid, meta in _iter_subject_concat_meta(repo_root, subjects):
        segments = meta.get("segments", [])
        for seg in segments:
            task = _normalise_task(seg.get("task", ""))
            if task in usage_counts:
                usage_counts[task] += 1

    # Estimated hits: usage minus the (assumed) initial build
    hit_counts: Dict[str, int] = {k: max(0, v - 1) for k, v in usage_counts.items()}

    total_hits = int(sum(hit_counts.values()))
    # Rebuilds not tracked in current metadata; best-effort set to 0 and log
    total_rebuilds = 0
    LOGGER.info("Estimated cache hits across stories: %d (rebuilds unknown -> 0)", total_hits)

    # Plot bar chart
    labels = list(sorted(hit_counts.keys()))
    values = [hit_counts[k] for k in labels]

    plots_root.mkdir(parents=True, exist_ok=True)
    out_path = plots_root / "cache_hit_summary.png"

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(labels)), 4))
    ax.bar(np.arange(len(labels)), values, color="tab:green")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Estimated hits")
    ax.set_title("Cache Hit Summary (estimated from concatenation metadata)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved cache-hit summary: %s", out_path)
    print(f"Total estimated cache hits: {total_hits}; rebuilds: {total_rebuilds}")
    return out_path


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = build_arg_parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    plots_root: Path = args.plots_root
    story_cache_root: Path = args.story_cache_root
    global_svd_path: Path = args.global_svd

    # Resolve stories
    story_dirs = _list_story_dirs(story_cache_root, args.stories)
    if story_dirs:
        LOGGER.info("Found %d story caches under %s", len(story_dirs), story_cache_root)
    else:
        LOGGER.warning("No story caches found under %s", story_cache_root)

    saved_paths: List[Path] = []

    # 1) Global SVD explained variance
    out = plot_global_explained_variance(global_svd_path, plots_root)
    if out is not None:
        saved_paths.append(out)

    # 2) Per-story L2 norm
    saved_paths.extend(plot_story_l2_norms(story_dirs, max_samples=int(args.max_samples)))

    # 3) Per-story surprisal histogram
    saved_paths.extend(plot_story_surprisal_hist(story_dirs, max_samples=int(args.max_samples)))

    # 4) Top-K cumulative mass curve (global)
    out = plot_topk_mass_curve(story_dirs, plots_root, max_samples=int(args.max_samples))
    if out is not None:
        saved_paths.append(out)

    # 5) Cache hit summary (global)
    repo_root = Path.cwd()
    out = plot_cache_hit_summary(repo_root, story_dirs, plots_root, args.subjects)
    if out is not None:
        saved_paths.append(out)

    if saved_paths:
        LOGGER.info("Saved %d figure(s):", len(saved_paths))
        for pth in saved_paths:
            print(str(pth))
    else:
        LOGGER.warning("No figures were generated. Check inputs and paths.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

