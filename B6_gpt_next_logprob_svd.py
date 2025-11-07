#!/usr/bin/env python3
"""
Build GPT next-token models using logprob-SVD with a global projection basis and story-level caching.

Outputs per subject (under derivatives/Models/gpt_next/<sub>/concatenated/):
- sub-XX_concatenated_gpt_next_100Hz.npy (components × T)
- sub-XX_concatenated_gpt_surprisal_100Hz.npy (1 × T)
- sub-XX_concatenated_gpt_predictability_100Hz.npy (1 × T)
- metadata JSON and optional plot; coverage mask saved for features; noise-fill applied only to features.

Caches per story (under derivatives/Models/gpt_next/story_cache/<story_id>/):
- reduced_tokens.npy (components × tokens)
- surprisal.npy (tokens,), predictability.npy (tokens,)
- token_map.json (word_token_counts)
- story_words.txt, story_hash.json, cache_metadata.json

Global SVD basis pooled across stories:
- derivatives/Models/gpt_next/global_svd_basis.pkl
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from scipy import sparse as sp

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import TruncatedSVD
import joblib

from functions.generic_helpers import read_repository_root


LOGGER = logging.getLogger(__name__)


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
    condition: str | None = None


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
            condition = str(trial_info.get("condition", "story")).strip().lower() or None
            events.append(WordEvent(word=word, sample=onset_sample, duration=duration, condition=condition))
    return events


def compute_segment_length(n_samples: int, raw_sfreq: float, target_rate: float) -> int:
    return int(round(n_samples * target_rate / raw_sfreq))


def allocate_memmap(path: Path, feature_dim: int, tps: int) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return open_memmap(str(path), mode="w+", dtype=np.float32, shape=(feature_dim, tps), fortran_order=False)


def plot_diagnostics(reduced: np.ndarray, predict: np.ndarray, sfreq: float, out: Path, max_points: int, caption: str) -> Path:
    n_timepoints = reduced.shape[1]
    step = max(1, int(np.ceil(n_timepoints / max_points)))
    idx = np.arange(0, n_timepoints, step, dtype=int)
    times = idx / sfreq
    l2 = np.linalg.norm(reduced[:, idx], axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(times, l2, lw=0.8)
    axes[0].set_ylabel("Feature L2 norm")
    axes[1].plot(times, predict[idx], lw=0.8, color="tab:orange")
    axes[1].set_ylabel("Predictability")
    axes[1].set_xlabel("Time (s)")
    fig.suptitle("GPT Next-Token: reduced + predictability")
    fig.text(0.01, 0.01, caption, fontsize=9, va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_gpt(model_path_or_name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_path_or_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Align with future Transformers default and silence FutureWarning
    # (only affects decode / text cleanup; encoding is unaffected)
    try:
        tok.clean_up_tokenization_spaces = True
    except Exception:
        pass
    mdl = AutoModelForCausalLM.from_pretrained(model_path_or_name)
    mdl.eval()
    mdl.to(device)
    return tok, mdl


def log_softmax_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.nn.functional.log_softmax(logits, dim=-1)
    return torch.nn.functional.log_softmax(logits / temperature, dim=-1)


def sha256_file(path: Path, chunk: int = 2 ** 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compute_fingerprint(model_path_or_name: str, tokenizer: AutoTokenizer) -> dict:
    fp = {"model": model_path_or_name}
    try:
        base = Path(model_path_or_name)
        if base.exists() and base.is_dir():
            weights = base / "pytorch_model.bin"
            if weights.exists():
                fp["weights_sha256"] = sha256_file(weights)
            for name in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "config.json"):
                p = base / name
                if p.exists():
                    fp[f"{name}_sha256"] = sha256_file(p)
    except Exception:
        pass
    fp["eos"] = getattr(tokenizer, "eos_token", None)
    return fp


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPT next-token logprob-SVD with global basis and story caches.")
    p.add_argument("--subjects", nargs="+", required=True, help="Subjects (e.g. 01 02 or sub-01)")
    p.add_argument("--hf-model", default=None, help="HF model path or name (default: derivatives/Models/gpt2 if present, else 'gpt2')")
    p.add_argument("--context-tokens", type=int, default=512, help="Max left context tokens")
    p.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (1.0 = raw)")
    p.add_argument("--positions-per-forward", type=int, default=256, help="Positions processed per forward pass")
    p.add_argument("--components", type=int, default=64, help="Number of SVD components")
    p.add_argument("--topk", type=int, default=4096, help="Cap for top-K indices per row")
    p.add_argument("--topk-mass", type=float, default=0.99, help="Keep smallest K with cumulative prob ≥ mass, capped by --topk")
    p.add_argument("--svd-fit-sample", type=int, default=100000, help="Total pooled rows for SVD fit across stories")
    p.add_argument("--svd-batch", type=int, default=4096, help="Rows per transform batch")
    p.add_argument("--global-svd-basis", type=Path, default=None, help="Path for global TruncatedSVD basis (default under derivatives/Models/gpt_next)")
    p.add_argument("--fit-global-basis", action="store_true", help="Fit global SVD basis when missing")
    p.add_argument("--rebuild-global-basis", action="store_true", help="Force rebuild global SVD basis")
    p.add_argument("--story-cache-root", type=Path, default=None, help="Folder for story caches (default under derivatives/Models/gpt_next)")
    p.add_argument("--reuse-story-cache", action="store_true", help="Reuse existing story caches (default on)")
    p.add_argument("--rebuild-story-cache", action="store_true", help="Force rebuild story caches")
    p.add_argument("--target-rate", type=float, default=100.0, help="Output sampling rate in Hz")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
    p.add_argument("--plot", action="store_true", help="Generate diagnostic plot")
    p.add_argument("--plot-max-points", type=int, default=20000)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def build_run_tokens(tokenizer: AutoTokenizer, events: List[WordEvent], raw_sfreq: float) -> Tuple[List[int], List[Tuple[int, int, float, float]]]:
    token_ids: List[int] = []
    words_tokens: List[Tuple[int, int, float, float]] = []
    for ev in events:
        onset_sec = ev.sample / raw_sfreq
        offset_sec = onset_sec + max(ev.duration, 0.0)
        # Emulate add_prefix_space per word by prefixing a space in the text
        text = (" " + ev.word) if (len(token_ids) > 0) else ev.word
        toks = tokenizer.encode(text, add_special_tokens=False)
        start = len(token_ids)
        token_ids.extend(toks)
        words_tokens.append((start, len(toks), onset_sec, offset_sec))
    return token_ids, words_tokens


def ensure_global_svd(
    basis_path: Path,
    components: int,
    topk_cap: int,
    topk_mass: float,
    context_tokens: int,
    positions_per_forward: int,
    temperature: float,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    tasks: List[str],
    run_tokens_by_task: Dict[str, Tuple[List[int], List[Tuple[int, int, float, float]]]],
    random_seed: int,
    svd_fit_sample: int,
) -> TruncatedSVD:
    if basis_path.exists():
        return joblib.load(basis_path)
    LOGGER.info("Fitting global SVD basis (components=%d, mass=%.2f, cap=%d)", components, topk_mass, topk_cap)
    V = int(model.get_output_embeddings().weight.shape[0])
    rows_target = int(max(1, svd_fit_sample))
    rows_data: List[float] = []
    rows_indices: List[int] = []
    rows_indptr: List[int] = [0]
    total_rows = 0
    for task in tasks:
        token_ids, _ = run_tokens_by_task[task]
        n = len(token_ids)
        pos = 0
        while pos < n and total_rows < rows_target:
            p_start = pos
            p_end = min(n - 1, p_start + positions_per_forward - 1)
            ctx_start = max(0, p_start - context_tokens + 1)
            window = token_ids[ctx_start : p_end + 1]
            input_ids = torch.tensor([window], dtype=torch.long, device=model.device)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0]
            log_probs = log_softmax_temperature(logits, temperature)
            sel_start = p_start - ctx_start
            sel_end = p_end - ctx_start
            for rel in range(sel_start, sel_end + 1):
                row = log_probs[rel]
                vals, idx = torch.topk(row, k=min(topk_cap, row.numel())) if topk_cap > 0 else torch.sort(row, descending=True)
                probs = torch.exp(vals)
                csum = torch.cumsum(probs, dim=0)
                k_mass = int(torch.searchsorted(csum, torch.tensor(topk_mass, device=csum.device)).item() + 1)
                k = min(int(idx.numel()), int(topk_cap), max(1, k_mass))
                rows_indices.extend(idx[:k].detach().cpu().numpy().tolist())
                rows_data.extend(vals[:k].detach().cpu().numpy().astype(np.float32).tolist())
                rows_indptr.append(len(rows_indices))
                total_rows += 1
                if total_rows >= rows_target:
                    break
            pos = p_end + 1
        if total_rows >= rows_target:
            break
    csr = sp.csr_matrix((np.asarray(rows_data, dtype=np.float32), np.asarray(rows_indices, dtype=np.int32), np.asarray(rows_indptr, dtype=np.int32)), shape=(total_rows, V), dtype=np.float32)
    svd = TruncatedSVD(n_components=components, algorithm="randomized", random_state=random_seed)
    svd.fit(csr)
    joblib.dump(svd, basis_path)
    LOGGER.info("Saved global SVD basis to %s (rows=%d)", basis_path, total_rows)
    return svd


def build_story_cache(
    cache_dir: Path,
    task: str,
    events: List[WordEvent],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    svd: TruncatedSVD,
    components: int,
    topk_cap: int,
    topk_mass: float,
    context_tokens: int,
    positions_per_forward: int,
    temperature: float,
    fingerprint: dict,
    random_seed: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    main_words = [e.word for e in events if (e.condition or "story") != "word_list"]
    story_text = "\n".join(main_words)
    story_hash = hashlib.sha256((story_text + json.dumps(fingerprint, sort_keys=True) + f"{topk_mass}|{topk_cap}|{components}|{context_tokens}").encode("utf-8")).hexdigest()
    hash_path = cache_dir / "story_hash.json"
    if (cache_dir / "reduced_tokens.npy").exists() and hash_path.exists():
        try:
            if json.loads(hash_path.read_text()).get("hash") == story_hash:
                return
        except Exception:
            pass

    # Tokenize main story
    add_prefix_space = False
    token_ids: List[int] = []
    word_token_counts: List[int] = []
    for w in main_words:
        text = (" " + w) if add_prefix_space else w
        toks = tokenizer.encode(text, add_special_tokens=False)
        add_prefix_space = True
        token_ids.extend(toks)
        word_token_counts.append(len(toks))
    n = len(token_ids)
    V = int(model.get_output_embeddings().weight.shape[0])

    reduced_tokens = np.memmap(cache_dir / "reduced_tokens.tmp.npy", mode="w+", dtype=np.float32, shape=(components, n))
    predict = np.memmap(cache_dir / "predictability.tmp.npy", mode="w+", dtype=np.float32, shape=(n,))
    surpr = np.memmap(cache_dir / "surprisal.tmp.npy", mode="w+", dtype=np.float32, shape=(n,))

    pos = 0
    while pos < n:
        p_start = pos
        p_end = min(n - 1, p_start + positions_per_forward - 1)
        ctx_start = max(0, p_start - context_tokens + 1)
        window = token_ids[ctx_start : p_end + 1]
        input_ids = torch.tensor([window], dtype=torch.long, device=model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0]
        log_probs = log_softmax_temperature(logits, temperature)
        sel_start = p_start - ctx_start
        sel_end = p_end - ctx_start

        rows_data: List[float] = []
        rows_indices: List[int] = []
        rows_indptr: List[int] = [0]

        for rel in range(sel_start, sel_end + 1):
            row = log_probs[rel]
            abs_pos = ctx_start + rel
            # Predictability: p(token_{i+1} | <= i)
            if abs_pos < n - 1:
                next_tok = token_ids[abs_pos + 1]
                predict[abs_pos] = float(torch.exp(row[next_tok]).item())
            else:
                predict[abs_pos] = np.nan
            # Surprisal: -log2 p(token_i | < i)
            if abs_pos == 0:
                surpr[abs_pos] = np.nan
            else:
                prev_ctx = max(0, abs_pos - 1 - context_tokens + 1)
                prev_window = token_ids[prev_ctx : abs_pos]
                with torch.no_grad():
                    prev_logits = model(input_ids=torch.tensor([prev_window], dtype=torch.long, device=model.device)).logits[0]
                prev_row = log_softmax_temperature(prev_logits, temperature)[-1]
                surpr[abs_pos] = float((-prev_row[token_ids[abs_pos]] / np.log(2.0)).item())
            # TopK-mass truncation for features
            vals, idx = torch.topk(row, k=min(topk_cap, row.numel())) if topk_cap > 0 else torch.sort(row, descending=True)
            probs = torch.exp(vals)
            csum = torch.cumsum(probs, dim=0)
            k_mass = int(torch.searchsorted(csum, torch.tensor(topk_mass, device=csum.device)).item() + 1)
            k = min(int(idx.numel()), int(topk_cap), max(1, k_mass))
            rows_indices.extend(idx[:k].detach().cpu().numpy().tolist())
            rows_data.extend(vals[:k].detach().cpu().numpy().astype(np.float32).tolist())
            rows_indptr.append(len(rows_indices))

        csr = sp.csr_matrix((np.asarray(rows_data, dtype=np.float32), np.asarray(rows_indices, dtype=np.int32), np.asarray(rows_indptr, dtype=np.int32)), shape=(sel_end - sel_start + 1, V), dtype=np.float32)
        reduced_batch = svd.transform(csr).astype(np.float32, copy=False)
        for i_rel in range(sel_start, sel_end + 1):
            abs_pos = ctx_start + i_rel
            reduced_tokens[:, abs_pos] = reduced_batch[i_rel - sel_start]
        pos = p_end + 1

    reduced_tokens.flush(); predict.flush(); surpr.flush()
    # Atomic writes: save to temp then rename to avoid readers seeing partial files
    def _atomic_save_npy(path: Path, array: np.ndarray):
        tmp = path.with_suffix(path.suffix + ".tmpwrite")
        np.save(tmp, array)
        tmp.replace(path)

    _atomic_save_npy(cache_dir / "reduced_tokens.npy", np.asarray(reduced_tokens))
    _atomic_save_npy(cache_dir / "predictability.npy", np.asarray(predict))
    _atomic_save_npy(cache_dir / "surprisal.npy", np.asarray(surpr))
    (cache_dir / "token_map.json.tmp").write_text(json.dumps({"word_token_counts": word_token_counts}, indent=2))
    (cache_dir / "token_map.json.tmp").replace(cache_dir / "token_map.json")
    (cache_dir / "story_words.txt.tmp").write_text("\n".join(main_words))
    (cache_dir / "story_words.txt.tmp").replace(cache_dir / "story_words.txt")
    (cache_dir / "story_hash.json.tmp").write_text(json.dumps({"hash": story_hash}, indent=2))
    (cache_dir / "story_hash.json.tmp").replace(cache_dir / "story_hash.json")
    (cache_dir / "cache_metadata.json").write_text(json.dumps({
        "projection": "logprob-svd",
        "components": components,
        "topk_cap": topk_cap,
        "topk_mass": topk_mass,
        "context_tokens": context_tokens,
        "temperature": temperature,
        "rng_seed": random_seed,
        "n_tokens": n,
        "first_token_surprisal": "NaN",
    }, indent=2))
    try:
        (cache_dir / "reduced_tokens.tmp.npy").unlink(missing_ok=True)  # type: ignore[arg-type]
        (cache_dir / "predictability.tmp.npy").unlink(missing_ok=True)  # type: ignore[arg-type]
        (cache_dir / "surprisal.tmp.npy").unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


def assemble_subject(
    subject: str,
    repo_root: Path,
    subject_runs: List[RunDescriptor],
    events_by_run: Dict[Tuple[str, str], List[WordEvent]],
    story_cache_root: Path,
    target_rate: float,
    feature_dim: int,
    overwrite: bool,
    noise_fill: bool,
    noise_scale: float,
    plot: bool,
    plot_max_points: int,
    random_seed: int,
) -> None:
    models_root = repo_root / "derivatives" / "Models" / "gpt_next"
    subj_dir = models_root / subject / "concatenated"
    subj_dir.mkdir(parents=True, exist_ok=True)
    base = f"{subject}_concatenated_gpt_next_{int(target_rate)}Hz"
    reduced_path = subj_dir / f"{base}.npy"
    pred_path = subj_dir / f"{subject}_concatenated_gpt_predictability_{int(target_rate)}Hz.npy"
    surpr_path = subj_dir / f"{subject}_concatenated_gpt_surprisal_{int(target_rate)}Hz.npy"
    if reduced_path.exists() and pred_path.exists() and surpr_path.exists() and not overwrite:
        LOGGER.info("Subject arrays exist for %s; skipping.", subject)
        return

    total_samples = sum(compute_segment_length(run.n_samples, run.sfreq, target_rate) for run in subject_runs)
    reduced = allocate_memmap(reduced_path, feature_dim, total_samples)
    pred = allocate_memmap(pred_path, 1, total_samples)
    surpr = allocate_memmap(surpr_path, 1, total_samples)
    coverage = np.memmap(subj_dir / f"{subject}_coverage_mask.tmp.npy", mode="w+", dtype=bool, shape=(total_samples,))
    reduced[:] = 0.0; pred[:] = 0.0; surpr[:] = 0.0; coverage[:] = False

    # Build a lookup per task
    cache_lookup = {}
    for run in subject_runs:
        cache_lookup.setdefault(run.task, story_cache_root / run.task)

    offset = 0
    for run in subject_runs:
        seg_len = compute_segment_length(run.n_samples, run.sfreq, target_rate)
        seg_start, seg_end = offset, offset + seg_len
        words = events_by_run.get((run.session, run.task), [])
        cache_dir = cache_lookup.get(run.task)
        red_tok = sur_tok = pred_tok = tok_counts = None
        if cache_dir and (cache_dir / "reduced_tokens.npy").exists():
            red_tok = np.load(cache_dir / "reduced_tokens.npy")
            sur_tok = np.load(cache_dir / "surprisal.npy")
            pred_tok = np.load(cache_dir / "predictability.npy")
            tok_map = json.loads((cache_dir / "token_map.json").read_text())
            tok_counts = tok_map.get("word_token_counts", [])
        cache_word_idx = 0  # index over main story words

        # Build word token counts list for this run with conditions
        # Use cache counts only for main story words
        for ev in words:
            onset_sec = ev.sample / run.sfreq
            offset_sec = onset_sec + max(ev.duration, 0.0)
            start_idx = seg_start + int(np.floor(onset_sec * target_rate))
            end_idx = seg_start + int(np.ceil(offset_sec * target_rate))
            if end_idx <= start_idx:
                end_idx = start_idx + 1
            start_idx = max(seg_start, start_idx)
            end_idx = min(seg_end, end_idx)
            dur = end_idx - start_idx
            is_word_list = (ev.condition or "story") == "word_list"
            if is_word_list or red_tok is None or tok_counts is None or cache_word_idx >= len(tok_counts):
                # No features for word_list segments (left for noise fill / masking)
                continue
            k = max(1, int(tok_counts[cache_word_idx]))
            for j in range(k):
                base = sum(tok_counts[:cache_word_idx])
                t_idx = base + j
                feat = red_tok[:, t_idx]
                pred_val = pred_tok[t_idx]
                surp_val = sur_tok[t_idx]
                t_s = start_idx + int(np.floor(j * dur / k))
                t_e = start_idx + int(np.floor((j + 1) * dur / k))
                if t_e <= t_s:
                    t_e = t_s + 1
                t_s = max(seg_start, t_s)
                t_e = min(seg_end, t_e)
                reduced[:, t_s:t_e] = feat[:, None]
                pred[:, t_s:t_e] = pred_val
                surpr[:, t_s:t_e] = surp_val
                coverage[t_s:t_e] = True
            cache_word_idx += 1
        offset = seg_end

    # Noise fill only features
    coverage.flush()
    cov = np.asarray(coverage, dtype=bool)
    if noise_fill:
        rng = np.random.default_rng(random_seed)
        n_feat, T = reduced.shape
        # per-feature std over covered timepoints
        sum_v = np.zeros(n_feat, dtype=np.float64)
        sumsq_v = np.zeros(n_feat, dtype=np.float64)
        count = 0
        chunk = 50000
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            m = cov[s:e]
            if not np.any(m):
                continue
            blk = np.asarray(reduced[:, s:e], dtype=np.float32)[:, m]
            sum_v += blk.sum(axis=1, dtype=np.float64)
            sumsq_v += np.square(blk, dtype=np.float64).sum(axis=1)
            count += blk.shape[1]
        if count > 0:
            mean_v = sum_v / count
            var_v = np.maximum(sumsq_v / count - mean_v ** 2, 1e-12)
            std_v = np.sqrt(var_v).astype(np.float32)
            scale = std_v * float(noise_scale)
            for s in range(0, T, chunk):
                e = min(T, s + chunk)
                m = ~cov[s:e]
                k = int(np.count_nonzero(m))
                if k == 0:
                    continue
                noise = rng.normal(loc=0.0, scale=scale[:, None], size=(n_feat, k)).astype(np.float32)
                reduced[:, s:e][:, m] = noise

    reduced.flush(); pred.flush(); surpr.flush()
    np.save(subj_dir / f"{subject}_coverage_mask_100Hz.npy", np.asarray(coverage, dtype=bool))
    try:
        (subj_dir / f"{subject}_coverage_mask.tmp.npy").unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass

    norm = np.linalg.norm(np.asarray(reduced, dtype=np.float32), axis=0)
    np.save(subj_dir / f"{base}_norm.npy", norm.astype(np.float32, copy=False))

    meta = {
        "subject": subject,
        "target_rate_hz": target_rate,
        "components": int(feature_dim),
        "notes": {
            "projection": "logprob-svd",
            "noise_fill_on_features_only": True,
        },
    }
    (subj_dir / f"{base}_metadata.json").write_text(json.dumps(meta, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = build_arg_parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    repo_root = read_repository_root()
    device = pick_device(args.device)
    default_local = repo_root / "derivatives" / "Models" / "gpt2"
    hf_model = str(default_local) if args.hf_model is None and default_local.exists() else (args.hf_model or "gpt2")
    tokenizer, model = load_gpt(hf_model, device)

    # Resolve paths
    models_root = repo_root / "derivatives" / "Models" / "gpt_next"
    models_root.mkdir(parents=True, exist_ok=True)
    story_cache_root = args.story_cache_root or (models_root / "story_cache")
    story_cache_root.mkdir(parents=True, exist_ok=True)
    global_svd_path = args.global_svd_basis or (models_root / "global_svd_basis.pkl")

    subjects = [normalise_subject(s) for s in args.subjects]

    # For the first subject: gather runs and fit global SVD if missing
    subj0 = subjects[0]
    preproc_root = repo_root / "derivatives" / "preprocessed"
    bids_root = repo_root / "bids_anonym"
    descriptors_by_key: Dict[str, Dict[Tuple[str, str], RunDescriptor]] = {}
    events_by_run: Dict[Tuple[str, str], List[WordEvent]] = {}
    runs0 = []
    for desc in iter_preprocessed_runs(preproc_root, bids_root, [subj0]):
        descriptors_by_key.setdefault(desc.subject, {})[(desc.session, desc.task)] = desc
        ev = load_word_events(desc.events_path)
        events_by_run[(desc.session, desc.task)] = ev
        if desc.subject == subj0:
            runs0.append(desc)
    # Unique tasks from first subject
    tasks0 = sorted({r.task for r in runs0})
    # Build token streams for each task for SVD fitting
    run_tokens_by_task: Dict[str, Tuple[List[int], List[Tuple[int, int, float, float]]]] = {}
    for r in runs0:
        evs = events_by_run[(r.session, r.task)]
        evs_main = [e for e in evs if (e.condition or "story") != "word_list"]
        token_ids, words_tokens = build_run_tokens(tokenizer, evs_main, r.sfreq)
        run_tokens_by_task[r.task] = (token_ids, words_tokens)

    # Fit or load global SVD
    if args.rebuild_global_basis and global_svd_path.exists():
        try:
            global_svd_path.unlink()
        except Exception:
            pass
    svd = ensure_global_svd(
        basis_path=global_svd_path,
        components=args.components,
        topk_cap=args.topk,
        topk_mass=args.topk_mass,
        context_tokens=args.context_tokens,
        positions_per_forward=args.positions_per_forward,
        temperature=args.temperature,
        tokenizer=tokenizer,
        model=model,
        tasks=tasks0,
        run_tokens_by_task=run_tokens_by_task,
        random_seed=args.random_seed,
        svd_fit_sample=args.svd_fit_sample,
    )

    # Fingerprint for hashes
    fp = compute_fingerprint(hf_model, tokenizer)

    # Build story caches for each unique task in runs0
    for task in tasks0:
        desc = next(r for r in runs0 if r.task == task)
        cache_dir = story_cache_root / task
        if args.rebuild_story_cache and cache_dir.exists():
            # remove core files to force rebuild
            for name in ("reduced_tokens.npy", "surprisal.npy", "predictability.npy"):
                try:
                    (cache_dir / name).unlink()
                except Exception:
                    pass
        build_story_cache(
            cache_dir=cache_dir,
            task=task,
            events=events_by_run[(desc.session, desc.task)],
            tokenizer=tokenizer,
            model=model,
            svd=svd,
            components=args.components,
            topk_cap=args.topk,
            topk_mass=args.topk_mass,
            context_tokens=args.context_tokens,
            positions_per_forward=args.positions_per_forward,
            temperature=args.temperature,
            fingerprint=fp,
            random_seed=args.random_seed,
        )

    # Assemble each subject
    for subj in subjects:
        # Gather ordered segments for subject
        # Read concatenation metadata
        concat_meta_path = preproc_root / subj / "concatenated" / f"{subj}_concatenation_metadata.json"
        if not concat_meta_path.exists():
            LOGGER.warning("Concatenation metadata missing for %s; skipping subject.", subj)
            continue
        concat_meta = json.loads(concat_meta_path.read_text())
        ordered: List[RunDescriptor] = []
        for seg in concat_meta.get("segments", []):
            session = normalise_session(seg["session"]) if isinstance(seg["session"], str) else f"ses-{seg['session']}"
            task = normalise_task(seg["task"]) if isinstance(seg["task"], str) else f"task-{seg['task']}"
            desc = descriptors_by_key.get(subj, {}).get((session, task))
            if desc is None:
                LOGGER.warning("Missing descriptor for %s %s %s; segment skipped.", subj, session, task)
                continue
            ordered.append(desc)
        if not ordered:
            LOGGER.warning("No valid runs for %s; skipping.", subj)
            continue
        # Ensure events for subject
        for desc in ordered:
            if (desc.session, desc.task) not in events_by_run:
                events_by_run[(desc.session, desc.task)] = load_word_events(desc.events_path)

        # Ensure story caches exist for this subject's tasks (idempotent; uses atomic writes)
        for desc in ordered:
            cache_dir = story_cache_root / desc.task
            if not (cache_dir / "reduced_tokens.npy").exists() or args.rebuild_story_cache:
                build_story_cache(
                    cache_dir=cache_dir,
                    task=desc.task,
                    events=events_by_run[(desc.session, desc.task)],
                    tokenizer=tokenizer,
                    model=model,
                    svd=svd,
                    components=args.components,
                    topk_cap=args.topk,
                    topk_mass=args.topk_mass,
                    context_tokens=args.context_tokens,
                    positions_per_forward=args.positions_per_forward,
                    temperature=args.temperature,
                    fingerprint=fp,
                    random_seed=args.random_seed,
                )

        # Compute feature_dim (components)
        feature_dim = args.components
        assemble_subject(
            subject=subj,
            repo_root=repo_root,
            subject_runs=ordered,
            events_by_run=events_by_run,
            story_cache_root=story_cache_root,
            target_rate=args.target_rate,
            feature_dim=feature_dim,
            overwrite=args.overwrite,
            noise_fill=True,
            noise_scale=0.01,
            plot=args.plot,
            plot_max_points=args.plot_max_points,
            random_seed=args.random_seed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
