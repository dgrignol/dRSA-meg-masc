#!/usr/bin/env python3
"""
Build concatenated GloVe word-embedding trajectories aligned with 100 Hz MEG timelines.

The script operates at the subject level to avoid creating unnecessary high-rate
arrays. For each subject it:

1. Reads the concatenation metadata to recover the ordered list of (session, task) runs.
2. Loads the corresponding BIDS events to extract word onsets/durations.
3. Loads only the required GloVe vectors from a user-supplied embedding file.
4. Allocates a memory-mapped array shaped (embedding_dim, timepoints_100Hz) and
   fills it with random embeddings sampled from the available vocabulary for every
   time point with no word.
5. Overwrites the segments covered by each word with the appropriate embedding.

Outputs (per subject):
- ``*_concatenated_glove_100Hz.npy`` (float32 memmap-compatible array)
- ``*_concatenated_glove_metadata.json`` with provenance and summary stats
- Optional ``*_plot.png`` showing the L2 norm of the trajectory over time

GloVe embeddings can be downloaded from the Stanford NLP website:
https://nlp.stanford.edu/projects/glove/  (e.g., glove.6B.300d.txt).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.format import open_memmap
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from functions.generic_helpers import read_repository_root


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunDescriptor:
    subject: str
    session: str
    task: str
    metadata_path: Path
    events_path: Path
    sfreq: float
    n_samples: int


@dataclass
class WordEvent:
    word: str
    sample: int
    duration: float


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def normalise_subject(label: str) -> str:
    label = label.strip()
    if label.startswith("sub-"):
        return label
    return f"sub-{int(label):02d}"


def normalise_session(label: str) -> str:
    label = label.strip()
    if label.startswith("ses-"):
        return label
    return f"ses-{label}"


def normalise_task(label: str) -> str:
    label = label.strip()
    if label.startswith("task-"):
        return label
    return f"task-{label}"


def iter_preprocessed_runs(
    preproc_root: Path,
    bids_root: Path,
    subjects: Optional[Sequence[str]],
) -> Iterable[RunDescriptor]:
    subject_filter = None
    if subjects:
        subject_filter = {normalise_subject(s) for s in subjects}

    for subj_dir in sorted(preproc_root.glob("sub-*")):
        subject = subj_dir.name
        if subject_filter and subject not in subject_filter:
            continue

        for meta_path in sorted(subj_dir.glob("ses-*/task-*/sub-*_metadata.json")):
            with meta_path.open("r") as fh:
                meta = json.load(fh)

            session = normalise_session(meta["session"])
            task = normalise_task(meta["task"])
            events_path = (
                bids_root
                / subject
                / session
                / "meg"
                / f"{subject}_{session}_{task}_events.tsv"
            )

            if not events_path.exists():
                LOGGER.warning("Events TSV missing for %s; skipping.", meta_path)
                continue

            sfreq = float(meta["sfreq"])
            n_samples = int(meta["n_samples"])
            yield RunDescriptor(
                subject=subject,
                session=session,
                task=task,
                metadata_path=meta_path,
                events_path=events_path,
                sfreq=sfreq,
                n_samples=n_samples,
            )


def parse_trial_info(value: str, events_path: Path) -> Optional[dict]:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        LOGGER.debug("Unable to parse trial_type in %s", events_path)
        return None


def load_word_events(events_path: Path) -> List[WordEvent]:
    events: List[WordEvent] = []
    with events_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            trial_info = parse_trial_info(row["trial_type"], events_path)
            if not trial_info or trial_info.get("kind") != "word":
                continue

            if float(trial_info.get("pronounced", 1.0)) == 0.0:
                continue

            try:
                onset_sample = int(row["sample"])
                duration = float(row["duration"])
            except (TypeError, ValueError):
                continue

            word = str(trial_info.get("word", "")).strip()
            if not word:
                continue
            events.append(WordEvent(word=word, sample=onset_sample, duration=duration))
    return events


def normalise_word_token(word: str) -> str:
    token = word.lower().strip()
    token = token.replace("’", "'")
    return token


def generate_word_candidates(word: str) -> List[str]:
    base = normalise_word_token(word)
    candidates = [base]
    stripped = base.strip(".,!?;:\"()[]{}")
    if stripped != base:
        candidates.append(stripped)
    if base.replace("-", "") != base:
        candidates.append(base.replace("-", ""))
    if base.replace("'", "") != base:
        candidates.append(base.replace("'", ""))
    if base.replace("’", "") != base:
        candidates.append(base.replace("’", ""))
    return list(dict.fromkeys(candidates))  # remove duplicates preserving order


def load_glove_embeddings(
    glove_path: Path,
    vocabulary: Sequence[str],
    dtype: np.dtype = np.float32,
) -> Tuple[Dict[str, np.ndarray], int]:
    vocab_set = set(vocabulary)
    embeddings: Dict[str, np.ndarray] = {}
    embedding_dim: Optional[int] = None

    with glove_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            token, *vector_str = parts
            if token not in vocab_set:
                continue

            if embedding_dim is None:
                embedding_dim = len(vector_str)
            elif embedding_dim != len(vector_str):
                raise ValueError(
                    f"Inconsistent embedding dimension in {glove_path}: "
                    f"expected {embedding_dim}, got {len(vector_str)} for token {token}"
                )

            vector = np.asarray(vector_str, dtype=np.float64)
            embeddings[token] = vector.astype(dtype, copy=False)

    if embedding_dim is None:
        raise ValueError(
            f"No requested vocabulary words were found in the GloVe file {glove_path}."
        )

    LOGGER.info(
        "Loaded %d/%d requested embeddings (dim=%d) from %s",
        len(embeddings),
        len(vocab_set),
        embedding_dim,
        glove_path,
    )
    return embeddings, embedding_dim


def ensure_embeddings_matrix(embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("No embeddings available to populate the trajectory.")
    matrix = np.vstack(list(embeddings.values())).astype(np.float32, copy=False)
    return matrix


def compute_segment_length(descriptor: RunDescriptor, target_rate: float) -> int:
    return int(round(descriptor.n_samples * target_rate / descriptor.sfreq))


def allocate_subject_memmap(
    output_path: Path,
    embedding_dim: int,
    total_samples: int,
) -> np.memmap:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mm = open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=(embedding_dim, total_samples),
        fortran_order=False,
    )
    return mm


def fill_with_random_embeddings(
    target: np.memmap,
    start: int,
    stop: int,
    embedding_pool: np.ndarray,
    rng: np.random.Generator,
    chunk_size: int = 50000,
) -> None:
    length = stop - start
    if length <= 0:
        return

    pool_size = embedding_pool.shape[0]
    pos = 0
    while pos < length:
        block = min(chunk_size, length - pos)
        idx = rng.integers(0, pool_size, size=block)
        target[:, start + pos : start + pos + block] = embedding_pool[idx].T
        pos += block


def assign_word_embedding(
    target: np.memmap,
    start: int,
    stop: int,
    embedding: np.ndarray,
) -> None:
    if stop <= start:
        return
    target[:, start:stop] = embedding[:, None]


def compute_l2_norm_series(
    data: np.memmap,
    chunk_size: int = 50000,
) -> np.ndarray:
    total = data.shape[1]
    norms = np.empty(total, dtype=np.float32)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        block = np.asarray(data[:, start:stop], dtype=np.float32)
        norms[start:stop] = np.linalg.norm(block, axis=0)
    return norms


def plot_embedding_summary(
    data: np.ndarray,
    norms: np.ndarray,
    sfreq: float,
    output_path: Path,
    max_points: int,
    parameter_caption: str,
) -> Path:
    if max_points <= 0:
        raise ValueError("max_points must be positive.")

    n_timepoints = data.shape[1]
    step = max(1, int(np.ceil(n_timepoints / max_points)))
    indices = np.arange(0, n_timepoints, step, dtype=int)
    sampled = np.asarray(data[:, indices], dtype=np.float32)
    t_axis = indices / float(sfreq)
    norm_samples = norms[indices]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Heatmap view (like MATLAB imagesc)
    im = ax_top.imshow(
        sampled,
        aspect="auto",
        origin="lower",
        extent=[t_axis[0], t_axis[-1] if len(t_axis) > 1 else 0.0, 0, sampled.shape[0]],
        cmap="viridis",
    )
    ax_top.set_ylabel("Embedding dim")
    ax_top.set_title("GloVe trajectory (subsampled heatmap)")
    fig.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04, label="Value")

    # Norm view
    ax_bottom.plot(t_axis, norm_samples, linewidth=0.8, color="tab:purple")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("L2 norm")
    ax_bottom.set_title("Embedding L2 norm over time")
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    fig.subplots_adjust(hspace=0.35)
    fig.text(0.5, 0.03, parameter_caption, ha="center", va="center", fontsize=8)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def prepare_subject_segments(
    subject: str,
    descriptors_by_key: Dict[Tuple[str, str], RunDescriptor],
    preproc_root: Path,
) -> List[RunDescriptor]:
    concat_meta_path = (
        preproc_root / subject / "concatenated" / f"{subject}_concatenation_metadata.json"
    )
    if not concat_meta_path.exists():
        LOGGER.warning("Concatenation metadata missing for %s; skipping subject.", subject)
        return []

    concat_meta = json.loads(concat_meta_path.read_text())
    ordered_descriptors: List[RunDescriptor] = []
    for segment in concat_meta.get("segments", []):
        session = segment["session"]
        task = segment["task"]
        key = (session, task)
        descriptor = descriptors_by_key.get(key)
        if descriptor is None:
            LOGGER.warning(
                "Missing descriptor for %s %s %s; segment skipped.",
                subject,
                session,
                task,
            )
            continue
        ordered_descriptors.append(descriptor)

    return ordered_descriptors


def collect_vocabulary(
    events_iterable: Iterable[List[WordEvent]],
) -> List[str]:
    vocab = set()
    for events in events_iterable:
        for event in events:
            for candidate in generate_word_candidates(event.word):
                vocab.add(candidate)
    return sorted(vocab)


def lookup_embedding(
    word: str,
    embeddings: Dict[str, np.ndarray],
    fallback_pool: np.ndarray,
    rng: np.random.Generator,
    missing_stats: Dict[str, int],
) -> np.ndarray:
    for candidate in generate_word_candidates(word):
        emb = embeddings.get(candidate)
        if emb is not None:
            return emb
    missing_stats["words_missing"] += 1
    idx = rng.integers(0, fallback_pool.shape[0])
    return fallback_pool[idx]


def build_subject_glove(
    subject: str,
    ordered_descriptors: List[RunDescriptor],
    word_events: Dict[Tuple[str, str], List[WordEvent]],
    embeddings: Dict[str, np.ndarray],
    embedding_matrix: np.ndarray,
    embedding_dim: int,
    models_root: Path,
    target_rate: float,
    rng: np.random.Generator,
    random_seed: int,
    overwrite: bool,
    plot: bool,
    plot_max_points: int,
) -> None:
    if not ordered_descriptors:
        LOGGER.warning("No valid descriptors for %s; skipping.", subject)
        return

    total_samples = sum(
        compute_segment_length(descriptor, target_rate) for descriptor in ordered_descriptors
    )
    subject_dir = models_root / subject / "concatenated"
    subject_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{subject}_concatenated_glove_{int(target_rate)}Hz"
    array_path = subject_dir / f"{base_name}.npy"
    metadata_path = subject_dir / f"{base_name}_metadata.json"

    if array_path.exists() and not overwrite:
        LOGGER.info("Concatenated GloVe model exists for %s; skipping.", subject)
        return

    target_memmap = allocate_subject_memmap(array_path, embedding_dim, total_samples)

    offset = 0
    all_words = 0
    missing_stats = {"words_missing": 0}
    coverage_samples = 0

    for descriptor in ordered_descriptors:
        seg_len = compute_segment_length(descriptor, target_rate)
        seg_start = offset
        seg_end = offset + seg_len

        fill_with_random_embeddings(
            target_memmap,
            seg_start,
            seg_end,
            embedding_matrix,
            rng,
        )

        events = word_events.get((descriptor.session, descriptor.task), [])
        for event in events:
            all_words += 1
            embedding = lookup_embedding(
                event.word,
                embeddings,
                embedding_matrix,
                rng,
                missing_stats,
            )

            onset_sec = event.sample / descriptor.sfreq
            offset_sec = onset_sec + max(event.duration, 0.0)

            start_idx = seg_start + int(np.floor(onset_sec * target_rate))
            end_idx = seg_start + int(np.ceil(offset_sec * target_rate))

            # Ensure at least one sample is covered
            if end_idx <= start_idx:
                end_idx = start_idx + 1
            # Clip to segment bounds
            start_idx = max(seg_start, start_idx)
            end_idx = min(seg_end, end_idx)

            assign_word_embedding(
                target_memmap,
                start_idx,
                end_idx,
                embedding.astype(np.float32, copy=False),
            )
            coverage_samples += max(0, end_idx - start_idx)

        offset += seg_len

    target_memmap.flush()

    norm_vector = compute_l2_norm_series(target_memmap)
    norm_path = subject_dir / f"{base_name}_norm.npy"
    np.save(norm_path, norm_vector.astype(np.float32, copy=False))

    metadata = {
        "subject": subject,
        "embedding_dim": embedding_dim,
        "target_rate_hz": target_rate,
        "total_timepoints": total_samples,
        "words_total": all_words,
        "words_missing": missing_stats["words_missing"],
        "coverage_samples": int(coverage_samples),
        "coverage_seconds": coverage_samples / target_rate,
        "random_seed": random_seed,
        "value_description": "Each column is a GloVe embedding (float32); silence/random spans draw from the available word vectors.",
        "glove_words_loaded": len(embeddings),
        "glove_vocabulary_size": int(embedding_matrix.shape[0]),
        "output_file": str(array_path),
        "norm_path": str(norm_path),
        "norm_summary": {
            "min": float(norm_vector.min()),
            "max": float(norm_vector.max()),
            "mean": float(norm_vector.mean()),
        },
    }

    caption_items = [
        f"dim={embedding_dim}",
        f"rate={target_rate:.1f}Hz",
        f"timepoints={total_samples}",
        f"words={all_words}",
        f"missing={missing_stats['words_missing']}",
        f"seed={random_seed}",
    ]
    parameter_caption = "Parameters: " + ", ".join(caption_items)

    if plot:
        plot_path = subject_dir / f"{base_name}_plot.png"
        plot_file = plot_embedding_summary(
            target_memmap,
            norm_vector,
            sfreq=target_rate,
            output_path=plot_path,
            max_points=plot_max_points,
            parameter_caption=parameter_caption,
        )
        metadata["plot"] = str(plot_file)

    metadata_path.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Saved concatenated GloVe model for %s", subject)
    del target_memmap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate subject-level GloVe trajectories aligned with 100 Hz MEG data."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional subset of subjects (e.g., 01 02 or sub-01 sub-02). Defaults to all available.",
    )
    parser.add_argument(
        "--glove-path",
        type=Path,
        required=True,
        help="Path to the GloVe embedding text file (e.g., glove.6B.300d.txt).",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=100.0,
        help="Output sampling rate in Hz (default: 100).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed controlling the random fill for silent intervals (default: 0).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a diagnostic plot (L2 norm over time).",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=20000,
        help="Maximum number of samples to render in the plot (default: 20k).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for console logging.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    glove_path = args.glove_path.expanduser()
    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe file not found: {glove_path}. Download e.g. from https://nlp.stanford.edu/data/glove.6B.zip"
        )

    repo_root = read_repository_root()
    preproc_root = repo_root / "derivatives" / "preprocessed"
    models_root = repo_root / "derivatives" / "Models" / "glove"
    models_root.mkdir(parents=True, exist_ok=True)
    bids_root = repo_root / "bids_anonym"

    subject_filter = [normalise_subject(s) for s in args.subjects] if args.subjects else None

    descriptors_by_subject: Dict[str, Dict[Tuple[str, str], RunDescriptor]] = defaultdict(dict)
    word_events: Dict[Tuple[str, str, str], List[WordEvent]] = {}

    for descriptor in iter_preprocessed_runs(preproc_root, bids_root, subject_filter):
        descriptors_by_subject[descriptor.subject][(descriptor.session, descriptor.task)] = (
            descriptor
        )
        key = (descriptor.subject, descriptor.session, descriptor.task)
        events = load_word_events(descriptor.events_path)
        word_events[key] = events

    subjects = sorted(descriptors_by_subject.keys())
    if not subjects:
        LOGGER.error("No subjects found matching the criteria.")
        return 1

    if not word_events:
        LOGGER.error("No word events found for the selected subjects.")
        return 1

    vocab = collect_vocabulary(word_events.values())
    if not vocab:
        LOGGER.error("Vocabulary extracted from events is empty; cannot build embeddings.")
        return 1
    embeddings, embedding_dim = load_glove_embeddings(glove_path, vocab)
    embedding_matrix = ensure_embeddings_matrix(embeddings)

    rng = np.random.default_rng(args.random_seed)

    for subject in subjects:
        ordered_descriptors = prepare_subject_segments(
            subject,
            descriptors_by_subject[subject],
            preproc_root,
        )
        subject_events = {
            (desc.session, desc.task): word_events.get((subject, desc.session, desc.task), [])
            for desc in ordered_descriptors
        }
        build_subject_glove(
            subject=subject,
            ordered_descriptors=ordered_descriptors,
            word_events=subject_events,
            embeddings=embeddings,
            embedding_matrix=embedding_matrix,
            embedding_dim=embedding_dim,
            models_root=models_root,
            target_rate=args.target_rate,
            rng=rng,
            random_seed=args.random_seed,
            overwrite=args.overwrite,
            plot=args.plot,
            plot_max_points=args.plot_max_points,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
