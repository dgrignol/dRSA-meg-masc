#!/usr/bin/env python3
"""
Build concatenated GPT next-token prediction trajectories aligned with 100 Hz MEG timelines.

For each subject, the script:

1) Iterates the concatenation order (sessions/tasks) and loads BIDS events to recover the
   ordered, pronounced words with onsets and durations.
2) Tokenises words using a Hugging Face GPT tokenizer to obtain a left-to-right token stream
   per run, keeping a mapping from words to their constituent tokens.
3) Runs a GPT causal LM in chunks to obtain next-token distributions; converts to probabilities
   and computes either:
     - Expected-embedding projection (default): p(next-token) @ E (E = token embedding matrix),
       yielding a hidden-size feature per token; OR
     - Optional dimensionality reduction directly on probabilities (NOT IMPLEMENTED for memory
       constraints; use expected-embedding and then PCA/SVD instead).
4) Applies optional dimensionality reduction (IncrementalPCA) on the expected-embedding vectors
   to a target number of components (default 64), then maps token-level features to the MEG time
   grid by dividing each word’s duration evenly across its tokens.
5) Produces, in addition to the reduced feature model, a scalar predictability trajectory
   (probability of the realised next token given context) and an optional surprisal trajectory.
6) Saves metadata and a diagnostic plot summarising key parameters.

Outputs (per subject):
- ``*_concatenated_gpt_next_100Hz.npy`` (float32, shape: n_features x timepoints)
- ``*_concatenated_gpt_predictability_100Hz.npy`` (float32, shape: 1 x timepoints)
- ``*_concatenated_gpt_surprisal_100Hz.npy`` (float32, optional)
- ``*_metadata.json`` (provenance and parameters)
- ``*_plot.png`` (diagnostic)

The script mirrors B4_glove.py’s structure for consistency (paths, metadata, plotting).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.format import open_memmap

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import IncrementalPCA
import joblib

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
# Utilities (adapted from B4_glove.py)
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


def compute_segment_length(n_samples: int, raw_sfreq: float, target_rate: float) -> int:
    return int(round(n_samples * target_rate / raw_sfreq))


def allocate_memmap(
    output_path: Path,
    feature_dim: int,
    total_samples: int,
) -> np.memmap:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mm = open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=(feature_dim, total_samples),
        fortran_order=False,
    )
    return mm


def compute_l2_norm_series(
    data: np.ndarray,
    chunk_size: int = 50000,
) -> np.ndarray:
    total = data.shape[1]
    norms = np.empty(total, dtype=np.float32)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        block = np.asarray(data[:, start:stop], dtype=np.float32)
        norms[start:stop] = np.linalg.norm(block, axis=0)
    return norms


def plot_gpt_summary(
    reduced: np.ndarray,
    predictability: np.ndarray,
    sfreq: float,
    output_path: Path,
    max_points: int,
    parameter_caption: str,
) -> Path:
    if max_points <= 0:
        raise ValueError("max_points must be positive.")

    n_timepoints = reduced.shape[1]
    step = max(1, int(np.ceil(n_timepoints / max_points)))
    indices = np.arange(0, n_timepoints, step, dtype=int)

    times = indices / sfreq
    l2 = np.linalg.norm(reduced[:, indices], axis=0)
    pred = predictability[indices]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(times, l2, lw=0.8)
    axes[0].set_ylabel("Feature L2 norm")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, pred, lw=0.8, color="tab:orange")
    axes[1].set_ylabel("Predictability")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("GPT Next-Token Model (reduced) and Predictability")
    fig.text(0.01, 0.01, parameter_caption, fontsize=9, va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Tokenisation and model inference
# ---------------------------------------------------------------------------


@dataclass
class WordTokens:
    token_start: int  # start index in the global token array
    token_count: int
    onset_sec: float
    offset_sec: float


@dataclass
class RunTokens:
    descriptor: RunDescriptor
    token_ids: List[int]
    words_tokens: List[WordTokens]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_gpt(model_path_or_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path_or_name)
    model.eval()
    model.to(device)
    return tokenizer, model


def build_run_tokens(
    tokenizer: AutoTokenizer,
    events: List[WordEvent],
    raw_sfreq: float,
) -> Tuple[List[int], List[WordTokens]]:
    token_ids: List[int] = []
    words_tokens: List[WordTokens] = []
    for wi, ev in enumerate(events):
        # Convert onset/offset to seconds
        onset_sec = ev.sample / raw_sfreq
        offset_sec = onset_sec + max(ev.duration, 0.0)
        # GPT tokenisation: add_prefix_space except for the first tokenised word
        add_prefix_space = len(token_ids) > 0
        toks = tokenizer.encode(
            ev.word,
            add_special_tokens=False,
            add_prefix_space=add_prefix_space,
        )
        start = len(token_ids)
        token_ids.extend(toks)
        words_tokens.append(WordTokens(token_start=start, token_count=len(toks), onset_sec=onset_sec, offset_sec=offset_sec))
    return token_ids, words_tokens


def iter_subject_runs(
    repo_root: Path,
    subject: str,
    subject_filter: Optional[Sequence[str]],
) -> List[RunDescriptor]:
    preproc_root = repo_root / "derivatives" / "preprocessed"
    bids_root = repo_root / "bids_anonym"
    descriptors = list(iter_preprocessed_runs(preproc_root, bids_root, subject_filter))
    # Preserve only this subject
    return [d for d in descriptors if d.subject == subject]


def prepare_subject_segments(
    subject: str,
    descriptors_by_key: Dict[Tuple[str, str], RunDescriptor],
    preproc_root: Path,
) -> List[RunDescriptor]:
    """Reconstruct ordered segments from concatenation metadata (same logic as B4_glove)."""
    concat_meta_path = preproc_root / subject / "concatenated" / f"{subject}_concatenation_metadata.json"
    if not concat_meta_path.exists():
        LOGGER.warning("Concatenation metadata missing for %s; skipping subject.", subject)
        return []
    concat_meta = json.loads(concat_meta_path.read_text())
    ordered: List[RunDescriptor] = []
    for segment in concat_meta.get("segments", []):
        session = normalise_session(segment["session"]) if isinstance(segment["session"], str) else f"ses-{segment['session']}"
        task = normalise_task(segment["task"]) if isinstance(segment["task"], str) else f"task-{segment['task']}"
        key = (session, task)
        desc = descriptors_by_key.get(key)
        if desc is None:
            LOGGER.warning("Missing descriptor for %s %s %s; segment skipped.", subject, session, task)
            continue
        ordered.append(desc)
    return ordered


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate subject-level GPT next-token models aligned to 100 Hz.")
    p.add_argument("--subjects", nargs="+", required=True, help="Subjects (e.g. 01 02 or sub-01)")
    p.add_argument("--hf-model", default=None, help="HF model path or name (default: derivatives/Models/gpt2 if exists, else 'gpt2')")
    p.add_argument("--context-tokens", type=int, default=512, help="Max left context tokens")
    p.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (1.0 = raw)")
    p.add_argument("--projection", choices=["expected-embedding", "pca-embedding"], default="pca-embedding", help="Projection type: expected-embedding or PCA on expected embeddings")
    p.add_argument("--components", type=int, default=64, help="Number of components after PCA (use 0 to skip PCA)")
    p.add_argument("--pca-batch", type=int, default=4096, help="Batch size for IncrementalPCA fit/transform")
    p.add_argument("--load-projection", type=Path, default=None, help="Load a pre-fitted PCA/IncrementalPCA (joblib .pkl) to use a fixed basis across subjects")
    p.add_argument("--save-projection", type=Path, default=None, help="Save the fitted PCA/IncrementalPCA (joblib .pkl) for reuse across subjects")
    p.add_argument("--positions-per-forward", type=int, default=256, help="Positions processed per forward pass (lower = lower memory)")
    p.add_argument("--target-rate", type=float, default=100.0, help="Output sampling rate in Hz")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
    p.add_argument("--plot", action="store_true", help="Generate diagnostic plot")
    p.add_argument("--plot-max-points", type=int, default=20000)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--reset-context-per-run", action="store_true", help="Reset GPT context at run boundaries (default: True)")
    p.add_argument("--no-reset-context-per-run", dest="reset_context_per_run", action="store_false")
    p.set_defaults(reset_context_per_run=True)
    p.add_argument("--noise-fill", action="store_true", help="Fill uncovered timepoints with low-amplitude Gaussian noise (default: on)")
    p.add_argument("--no-noise-fill", dest="noise_fill", action="store_false")
    p.set_defaults(noise_fill=True)
    p.add_argument("--noise-scale", type=float, default=0.01, help="Relative noise scale (fraction of per-feature std) for gap filling")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def softmax_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.nn.functional.softmax(logits, dim=-1)
    return torch.nn.functional.softmax(logits / temperature, dim=-1)


def run_subject(
    subject: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    repo_root: Path,
    target_rate: float,
    context_tokens: int,
    temperature: float,
    positions_per_forward: int,
    components: int,
    pca_batch: int,
    load_projection: Optional[Path],
    save_projection: Optional[Path],
    reset_context_per_run: bool,
    overwrite: bool,
    plot: bool,
    plot_max_points: int,
    random_seed: int,
    noise_fill: bool,
    noise_scale: float,
) -> None:
    rng = np.random.default_rng(random_seed)

    preproc_root = repo_root / "derivatives" / "preprocessed"
    bids_root = repo_root / "bids_anonym"
    models_root = repo_root / "derivatives" / "Models" / "gpt_next"
    models_root.mkdir(parents=True, exist_ok=True)

    # Gather run descriptors and events
    descriptors_by_key: Dict[Tuple[str, str], RunDescriptor] = {}
    words_by_run: Dict[Tuple[str, str], List[WordEvent]] = {}
    for desc in iter_preprocessed_runs(preproc_root, bids_root, [subject]):
        descriptors_by_key[(desc.session, desc.task)] = desc
        words_by_run[(desc.session, desc.task)] = load_word_events(desc.events_path)

    ordered = prepare_subject_segments(subject, descriptors_by_key, preproc_root)
    if not ordered:
        LOGGER.warning("No valid ordered runs for %s; skipping.", subject)
        return

    # First pass: tokenise and count tokens, compute total samples
    run_tokens: List[RunTokens] = []
    total_tokens = 0
    total_samples = 0
    for desc in ordered:
        events = words_by_run.get((desc.session, desc.task), [])
        token_ids, words_tokens = build_run_tokens(tokenizer, events, desc.sfreq)
        run_tokens.append(RunTokens(descriptor=desc, token_ids=token_ids, words_tokens=words_tokens))
        total_tokens += len(token_ids)
        total_samples += compute_segment_length(desc.n_samples, desc.sfreq, target_rate)

    subject_dir = models_root / subject / "concatenated"
    subject_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{subject}_concatenated_gpt_next_{int(target_rate)}Hz"
    reduced_path = subject_dir / f"{base_name}.npy"
    predictability_path = subject_dir / f"{subject}_concatenated_gpt_predictability_{int(target_rate)}Hz.npy"
    surprisal_path = subject_dir / f"{subject}_concatenated_gpt_surprisal_{int(target_rate)}Hz.npy"
    metadata_path = subject_dir / f"{base_name}_metadata.json"
    plot_path = subject_dir / f"{base_name}_plot.png"

    if reduced_path.exists() and predictability_path.exists() and not overwrite:
        LOGGER.info("Outputs already exist for %s; skipping.", subject)
        return

    # Temporary memmap to store token-level expected embeddings for PCA
    with torch.no_grad():
        hidden_size = int(model.get_output_embeddings().weight.shape[1])
    token_embed_tmp = subject_dir / f"{subject}_token_expected_embeddings.tmp.npy"
    token_embed = open_memmap(str(token_embed_tmp), mode="w+", dtype=np.float32, shape=(total_tokens, hidden_size))
    token_predictability = np.memmap(subject_dir / f"{subject}_token_predictability.tmp.npy", mode="w+", dtype=np.float32, shape=(total_tokens,))
    token_surprisal = np.memmap(subject_dir / f"{subject}_token_surprisal.tmp.npy", mode="w+", dtype=np.float32, shape=(total_tokens,))

    # Second structure to map (run->word->token positions) back to time later
    # We'll build a global list of WordTokens with token indices offset into [0, total_tokens)
    global_words: List[WordTokens] = []

    # First inference pass: compute expected-embedding per token + predictability/surprisal
    tok_offset = 0
    # Track context length usage histogram up to context_tokens
    context_hist = np.zeros(context_tokens, dtype=np.int64)
    embedding_matrix = model.get_output_embeddings().weight.detach().to(device)
    fp16 = (device.type in ("cuda", "mps"))
    for ri, rt in enumerate(run_tokens):
        token_ids = rt.token_ids
        words_toks = rt.words_tokens
        n = len(token_ids)
        if n == 0:
            continue

        # Build next-token targets for predictability
        next_targets = np.full(n, -100, dtype=np.int64)
        if n > 1:
            next_targets[:-1] = np.array(token_ids[1:], dtype=np.int64)

        pos = 0
        while pos < n:
            p_start = pos
            p_end = min(n - 1, p_start + positions_per_forward - 1)  # include last position in batch
            ctx_start = max(0, p_start - context_tokens + 1) if reset_context_per_run else max(0, p_start - context_tokens + 1)
            window = token_ids[ctx_start : p_end + 1]

            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attn)
                logits = outputs.logits[0]  # (L, V)
            # Convert to probs with temperature
            probs = softmax_temperature(logits, temperature=temperature).to(device)
            # Expected embedding per position
            # (L, V) @ (V, H) -> (L, H)
            exp_emb = torch.matmul(probs, embedding_matrix)  # (L, H)
            if fp16:
                exp_emb = exp_emb.to(torch.float32)

            # Indices within this window that correspond to positions p_start..p_end
            sel_start = p_start - ctx_start
            sel_end = p_end - ctx_start  # inclusive
            exp_emb_sel = exp_emb[sel_start : sel_end + 1]

            # Write expected embeddings
            batch_len = int(exp_emb_sel.shape[0])
            token_embed[tok_offset : tok_offset + batch_len, :] = exp_emb_sel.detach().cpu().numpy().astype(np.float32, copy=False)

            # Predictability and surprisal
            # We need the probability assigned to the realised next token
            # Fetch probs for the same indices
            probs_sel = probs[sel_start : sel_end + 1]
            realised_next = next_targets[p_start : p_start + batch_len]
            realised_next_tensor = torch.from_numpy(realised_next).to(device)
            gathered = torch.zeros(batch_len, dtype=torch.float32, device=device)
            valid_mask = realised_next_tensor >= 0
            if torch.any(valid_mask):
                idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                rows = idx
                cols = realised_next_tensor[valid_mask]
                gathered_vals = probs_sel[rows, cols]
                gathered[valid_mask] = gathered_vals.to(torch.float32)
            pred_np = gathered.detach().cpu().numpy()
            token_predictability[tok_offset : tok_offset + batch_len] = pred_np
            # Surprisal: -log2 p
            with np.errstate(divide="ignore"):
                surpr = -np.log2(np.clip(pred_np, 1e-12, 1.0))
            token_surprisal[tok_offset : tok_offset + batch_len] = surpr.astype(np.float32, copy=False)

            # Context usage histogram
            # For j in [sel_start..sel_end], effective context size = min(context_tokens, j+1)
            for j in range(sel_start, sel_end + 1):
                eff = min(context_tokens, j + 1)
                context_hist[eff - 1] += 1

            tok_offset += batch_len
            pos = p_end + 1

        # Extend global word mapping with token index offsets
        for w in words_toks:
            global_words.append(
                WordTokens(
                    token_start=w.token_start + (tok_offset - n),
                    token_count=w.token_count,
                    onset_sec=w.onset_sec,
                    offset_sec=w.offset_sec,
                )
            )

    # PCA fit on expected embeddings if requested
    explained_variance_ratio = None
    ipca: Optional[IncrementalPCA] = None
    if components and components > 0:
        if load_projection is not None and load_projection.exists():
            ipca = joblib.load(load_projection)
            explained_variance_ratio = getattr(ipca, "explained_variance_ratio_", None)
            if explained_variance_ratio is not None:
                explained_variance_ratio = explained_variance_ratio.tolist()
        else:
            ipca = IncrementalPCA(n_components=components)
            for start in range(0, total_tokens, pca_batch):
                end = min(total_tokens, start + pca_batch)
                ipca.partial_fit(np.asarray(token_embed[start:end], dtype=np.float32))
            explained_variance_ratio = ipca.explained_variance_ratio_.tolist()
            if save_projection is not None:
                try:
                    joblib.dump(ipca, save_projection)
                except Exception as exc:
                    LOGGER.warning("Failed to save projection to %s: %s", save_projection, exc)

    # Allocate final time-based arrays
    feature_dim = components if (components and components > 0) else hidden_size
    reduced_mem = allocate_memmap(reduced_path, feature_dim, total_samples)
    pred_mem = allocate_memmap(predictability_path, 1, total_samples)
    surpr_mem = allocate_memmap(surprisal_path, 1, total_samples)
    coverage_mask = np.memmap(subject_dir / f"{subject}_coverage_mask.tmp.npy", mode="w+", dtype=bool, shape=(total_samples,))
    # Initialise to zeros
    reduced_mem[:] = 0.0
    pred_mem[:] = 0.0
    surpr_mem[:] = 0.0

    # Helpers for assignment per word/token
    def assign_interval(arr: np.memmap | np.ndarray, values: np.ndarray, start: int, stop: int):
        if stop <= start:
            return
        if values.ndim == 1:
            arr[:, start:stop] = values[:, None]
        else:
            arr[:, start:stop] = values

    # Time mapping
    offset_samples = 0
    word_idx = 0
    # Build per-run sample lengths so we can compute segment bounds and clip
    run_sample_lengths = [compute_segment_length(d.descriptor.n_samples, d.descriptor.sfreq, target_rate) for d in run_tokens]

    # Iterate runs and words in order again, mapping token-level features to time
    global_token_pos = 0
    for r_i, rt in enumerate(run_tokens):
        seg_len = run_sample_lengths[r_i]
        seg_start = offset_samples
        seg_end = seg_start + seg_len

        # Map words
        events = rt.words_tokens
        for w in events:
            # Compute word start/stop indices on the 100 Hz grid
            start_idx = seg_start + int(np.floor(w.onset_sec * target_rate))
            end_idx = seg_start + int(np.ceil(w.offset_sec * target_rate))
            if end_idx <= start_idx:
                end_idx = start_idx + 1
            # Clip to segment bounds
            start_idx = max(seg_start, start_idx)
            end_idx = min(seg_end, end_idx)
            duration = end_idx - start_idx
            k = max(1, w.token_count)

            # Evenly divide the duration across tokens
            # token j spans [start + floor(j*L/k), start + floor((j+1)*L/k))
            for j in range(k):
                t_global = global_words[word_idx].token_start + j
                t_feature = token_embed[t_global]
                if ipca is not None:
                    t_feature = ipca.transform(np.asarray(t_feature, dtype=np.float32)[None, :])[0]
                # Scalars
                t_pred = float(token_predictability[t_global])
                t_surp = float(token_surprisal[t_global])

                t_s = start_idx + int(np.floor(j * duration / k))
                t_e = start_idx + int(np.floor((j + 1) * duration / k))
                if t_e <= t_s:
                    t_e = t_s + 1
                t_s = max(seg_start, t_s)
                t_e = min(seg_end, t_e)

                # Assign
                assign_interval(reduced_mem, np.asarray(t_feature, dtype=np.float32), t_s, t_e)
                pred_mem[:, t_s:t_e] = t_pred
                surpr_mem[:, t_s:t_e] = t_surp
                coverage_mask[t_s:t_e] = True

            word_idx += 1

        offset_samples += seg_len

    # Optional noise fill for uncovered timepoints
    coverage_mask.flush()
    coverage = np.asarray(coverage_mask, dtype=bool)
    if noise_fill:
        # Compute per-feature std on covered columns only (chunked)
        n_feat = reduced_mem.shape[0]
        total_tp = reduced_mem.shape[1]
        sum_v = np.zeros(n_feat, dtype=np.float64)
        sumsq_v = np.zeros(n_feat, dtype=np.float64)
        count = 0
        chunk = 50000
        for s in range(0, total_tp, chunk):
            e = min(total_tp, s + chunk)
            mask_c = coverage[s:e]
            if not np.any(mask_c):
                continue
            block = np.asarray(reduced_mem[:, s:e], dtype=np.float32)[:, mask_c]
            sum_v += block.sum(axis=1, dtype=np.float64)
            sumsq_v += np.square(block, dtype=np.float64).sum(axis=1)
            count += block.shape[1]
        if count > 0:
            mean_v = sum_v / count
            var_v = np.maximum(sumsq_v / count - mean_v ** 2, 1e-12)
            std_v = np.sqrt(var_v).astype(np.float32)
            scale = std_v * float(noise_scale)
            # Fill gaps in chunks
            for s in range(0, total_tp, chunk):
                e = min(total_tp, s + chunk)
                mask_c = ~coverage[s:e]
                k = int(np.count_nonzero(mask_c))
                if k == 0:
                    continue
                noise = rng.normal(loc=0.0, scale=scale[:, None], size=(n_feat, k)).astype(np.float32)
                reduced_mem[:, s:e][:, mask_c] = noise

    # Flush arrays
    reduced_mem.flush()
    pred_mem.flush()
    surpr_mem.flush()

    # Metadata and plot
    # Compute L2 norm for summary
    norm_vector = compute_l2_norm_series(reduced_mem)
    norm_path = subject_dir / f"{base_name}_norm.npy"
    np.save(norm_path, norm_vector.astype(np.float32, copy=False))

    projection_mode = "pca-embedding" if (components and components > 0 and ipca is not None) else "expected-embedding"
    metadata = {
        "subject": subject,
        "target_rate_hz": target_rate,
        "total_timepoints": int(total_samples),
        "total_tokens": int(total_tokens),
        "projection": projection_mode,
        "hidden_size": int(hidden_size),
        "components": int(components),
        "explained_variance_ratio": explained_variance_ratio,
        "context_tokens": int(context_tokens),
        "temperature": float(temperature),
        "positions_per_forward": int(positions_per_forward),
        "reset_context_per_run": bool(reset_context_per_run),
        "random_seed": int(random_seed),
        "fill_strategy": "gaussian_noise" if noise_fill else "zeros",
        "noise_scale_relative": float(noise_scale) if noise_fill else 0.0,
        "has_coverage_mask": True,
        "context_hist_bins": list(range(1, context_tokens + 1)),
        "context_hist_counts": context_hist.tolist(),
        "value_description": "Each column is a reduced feature vector (float32) derived from GPT next-token expected embeddings; predictability gives P(realised next token|context).",
        "output_files": {
            "reduced": str(reduced_path),
            "predictability": str(predictability_path),
            "surprisal": str(surprisal_path),
            "norm": str(norm_path),
            "coverage_mask": str(subject_dir / f"{subject}_coverage_mask_100Hz.npy"),
        },
        "norm_summary": {
            "min": float(norm_vector.min()),
            "max": float(norm_vector.max()),
            "mean": float(norm_vector.mean()),
        },
    }

    parameter_caption = (
        f"hf=GPT2, proj=expected-embedding, comps={components}, ctx={context_tokens}, "
        f"rate={target_rate:.1f}Hz, tokens={total_tokens}"
    )

    if plot:
        plot_gpt_summary(
            reduced_mem,
            np.asarray(pred_mem).ravel(),
            sfreq=target_rate,
            output_path=plot_path,
            max_points=plot_max_points,
            parameter_caption=parameter_caption,
        )
        metadata["plot"] = str(plot_path)
        # Context histogram plot
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            bins = np.arange(1, context_tokens + 1)
            ax.bar(bins, context_hist, width=1.0)
            ax.set_xlabel("Effective context tokens")
            ax.set_ylabel("Count")
            ax.set_title("GPT Context Usage Histogram")
            ctx_plot = subject_dir / f"{subject}_gpt_context_hist.png"
            fig.tight_layout()
            fig.savefig(ctx_plot, dpi=120)
            plt.close(fig)
            metadata["context_plot"] = str(ctx_plot)
        except Exception as exc:
            LOGGER.warning("Failed to save context histogram plot: %s", exc)

    # Save metadata
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Clean up temp token arrays
    del token_embed
    del token_predictability
    del token_surprisal
    # Persist coverage mask
    try:
        np.save(subject_dir / f"{subject}_coverage_mask_100Hz.npy", np.asarray(coverage_mask, dtype=bool))
    except Exception:
        pass

    try:
        token_embed_tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        (subject_dir / f"{subject}_token_predictability.tmp.npy").unlink(missing_ok=True)  # type: ignore[arg-type]
        (subject_dir / f"{subject}_token_surprisal.tmp.npy").unlink(missing_ok=True)  # type: ignore[arg-type]
        (subject_dir / f"{subject}_coverage_mask.tmp.npy").unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    repo_root = read_repository_root()
    device = pick_device(args.device)
    LOGGER.info("Using device: %s", device)

    # Resolve HF model path/name
    default_local = repo_root / "derivatives" / "Models" / "gpt2"
    if args.hf_model is None:
        if default_local.exists():
            hf_model = str(default_local)
        else:
            hf_model = "gpt2"
    else:
        hf_model = args.hf_model

    # Honor projection choice: when projection is expected-embedding, force components=0
    if args.projection == "expected-embedding":
        args.components = 0

    tokenizer, model = load_gpt(hf_model, device)

    subjects = [normalise_subject(s) for s in args.subjects]
    for idx, subject in enumerate(subjects):
        # Ensure a fixed projection across subjects when --save-projection is provided:
        # - First subject fits and saves
        # - Subsequent subjects load the saved projection and skip saving
        load_proj_this = args.load_projection
        save_proj_this = args.save_projection
        if args.save_projection is not None and idx > 0:
            load_proj_this = args.save_projection
            save_proj_this = None

        run_subject(
            subject=subject,
            tokenizer=tokenizer,
            model=model,
            device=device,
            repo_root=repo_root,
            target_rate=args.target_rate,
            context_tokens=args.context_tokens,
            temperature=args.temperature,
            positions_per_forward=args.positions_per_forward,
            components=args.components,
            pca_batch=args.pca_batch,
            load_projection=load_proj_this,
            save_projection=save_proj_this,
            reset_context_per_run=args.reset_context_per_run,
            overwrite=args.overwrite,
            plot=args.plot,
            plot_max_points=args.plot_max_points,
            random_seed=args.random_seed,
            noise_fill=args.noise_fill,
            noise_scale=args.noise_scale,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
