#!/usr/bin/env python3
"""
Batch runner for the MEG-MASC analysis pipeline.

The wrapper orchestrates the per-subject scripts shipped in this repository
so a single command can execute the full analysis chain:
    A1_preprocess_data.py
    A2_concatenate_subject_data.py
    A3_resample_concatenated_data.py
    B1_model_envelope.py
    B2_wordfreq.py
    B3_voicing.py
    B4_glove.py
    C1_dRSA_run.py
    D1_group_cluster_analysis.py

Example
-------
python pipeline_wrapper.py --subjects 2-23 --glove-path /path/to/glove.6B.300d.txt

Whenever the GloVe embedding path is stable, store it in the environment
variable ``GLOVE_PATH`` or in ``glove_path.txt`` (one path per line, comments
starting with '#') to make the flag optional.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from functions.generic_helpers import read_repository_root


LOGGER = logging.getLogger("pipeline_wrapper")


DEFAULT_MODELS = ["Envelope", "Phoneme Voicing", "Word Frequency", "GloVe", "GloVe Norm"]


@dataclass
class CommandSpec:
    """Runnable command with optional metadata."""

    label: str
    argv: List[str]
    env: Optional[Dict[str, str]] = None
    cwd: Optional[Path] = None


@dataclass
class PipelineContext:
    """Shared state passed to pipeline step builders."""

    repo_root: Path
    python_executable: str
    subject_ids: List[str]  # e.g., ["02", "03"]
    overwrite: bool
    continue_on_error: bool
    glove_path: Optional[Path]
    group_subject_tokens: Optional[List[str]]
    results_dir: Path
    lag_metric: str
    models: List[str]
    d1_output: Optional[Path]
    d1_n_permutations: Optional[int]

    _cached_group_subjects: Optional[List[str]] = None

    @property
    def subject_labels(self) -> List[str]:
        return [f"sub-{sid}" for sid in self.subject_ids]

    def resolve_group_subjects(self) -> List[str]:
        if self._cached_group_subjects is not None:
            return self._cached_group_subjects
        if self.group_subject_tokens:
            subjects = parse_subject_tokens(self.group_subject_tokens)
        else:
            subjects = discover_group_subjects(
                results_dir=self.results_dir,
                lag_metric=self.lag_metric,
            )
        self._cached_group_subjects = subjects
        return subjects


@dataclass
class PipelineStep:
    """Declarative representation of a pipeline stage."""

    name: str
    builder: Callable[[PipelineContext], List[CommandSpec]]
    description: str


def parse_subject_tokens(tokens: Sequence[str]) -> List[str]:
    """Expand subject identifiers (supports ranges like '2-23')."""

    numbers: set[int] = set()
    token_pattern = re.compile(r"^(?:sub-)?(\d+)$", re.IGNORECASE)

    for raw_token in tokens:
        for piece in re.split(r"[,\s]+", raw_token.strip()):
            if not piece:
                continue
            range_match = re.fullmatch(r"(?:sub-)?(\d+)\s*-\s*(?:sub-)?(\d+)", piece, re.IGNORECASE)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if start <= 0 or end <= 0:
                    raise ValueError(f"Subject ranges must be positive integers: {piece}")
                if start <= end:
                    span = range(start, end + 1)
                else:
                    span = range(end, start + 1)
                numbers.update(span)
                continue

            single_match = token_pattern.fullmatch(piece)
            if single_match:
                value = int(single_match.group(1))
                if value <= 0:
                    raise ValueError(f"Subject identifiers must be positive integers: {piece}")
                numbers.add(value)
                continue

            raise ValueError(f"Unrecognised subject token: {piece}")

    return [f"{value:02d}" for value in sorted(numbers)]


def discover_group_subjects(results_dir: Path, lag_metric: str) -> List[str]:
    """Infer subjects for D1 by scanning the results directory."""

    pattern = re.compile(
        rf"sub-?(\d+)_res\d+_{re.escape(lag_metric)}_dRSA_matrices\.npy$", re.IGNORECASE
    )
    subjects: set[int] = set()

    if not results_dir.exists():
        LOGGER.warning("Results directory %s does not exist; skipping group analysis.", results_dir)
        return []

    for path in results_dir.glob(f"sub*_res*_{lag_metric}_dRSA_matrices.npy"):
        match = pattern.search(path.name)
        if match:
            subjects.add(int(match.group(1)))

    resolved = [f"{value:02d}" for value in sorted(subjects)]
    if not resolved:
        LOGGER.warning(
            "No subject results matched the pattern '*_res*_%s_dRSA_matrices.npy' in %s.",
            lag_metric,
            results_dir,
        )
    return resolved


def resolve_glove_path(cli_value: Optional[str], repo_root: Path) -> Optional[Path]:
    """Choose a GloVe embedding path from CLI, environment, or a pointer file."""

    candidates: List[Path] = []
    if cli_value:
        candidates.append(Path(cli_value).expanduser())

    env_value = os.getenv("GLOVE_PATH")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    pointer = repo_root / "glove_path.txt"
    if pointer.exists():
        for line in pointer.read_text().splitlines():
            candidate = line.split("#", 1)[0].strip()
            if candidate:
                candidates.append(Path(candidate).expanduser())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if cli_value:
        raise FileNotFoundError(f"GloVe embedding not found at {cli_value}.")

    return None


def build_step_commands(step: PipelineStep, ctx: PipelineContext) -> List[CommandSpec]:
    """Delegate to the step builder and normalise command metadata."""

    commands = step.builder(ctx)
    for command in commands:
        command.cwd = command.cwd or ctx.repo_root
    return commands


def step_a1(ctx: PipelineContext) -> List[CommandSpec]:
    if not ctx.subject_ids:
        return []
    argv = [ctx.python_executable, "A1_preprocess_data.py", "--subjects", *ctx.subject_ids]
    if ctx.overwrite:
        argv.append("--overwrite")
    return [CommandSpec("A1_preprocess", argv)]


def step_a2(ctx: PipelineContext) -> List[CommandSpec]:
    commands: List[CommandSpec] = []
    for label in ctx.subject_labels:
        argv = [
            ctx.python_executable,
            "A2_concatenate_subject_data.py",
            "--subject",
            label,
        ]
        if ctx.overwrite:
            argv.append("--overwrite")
        commands.append(CommandSpec(f"A2_concatenate[{label}]", argv))
    return commands


def step_a3(ctx: PipelineContext) -> List[CommandSpec]:
    commands: List[CommandSpec] = []
    for label in ctx.subject_labels:
        argv = [
            ctx.python_executable,
            "A3_resample_concatenated_data.py",
            "--subject",
            label,
        ]
        if ctx.overwrite:
            argv.append("--overwrite")
        commands.append(CommandSpec(f"A3_resample[{label}]", argv))
    return commands


def _build_subject_step(script: str, label: str, ctx: PipelineContext) -> Optional[CommandSpec]:
    if not ctx.subject_ids:
        return None
    argv = [ctx.python_executable, script]
    argv.extend(["--subjects", *ctx.subject_ids])
    if ctx.overwrite:
        argv.append("--overwrite")
    return CommandSpec(label, argv)


def step_b1(ctx: PipelineContext) -> List[CommandSpec]:
    command = _build_subject_step("B1_model_envelope.py", "B1_envelope", ctx)
    return [command] if command else []


def step_b2(ctx: PipelineContext) -> List[CommandSpec]:
    command = _build_subject_step("B2_wordfreq.py", "B2_wordfreq", ctx)
    if command:
        command.argv.extend(["--target-rate", "100"])
    return [command] if command else []


def step_b3(ctx: PipelineContext) -> List[CommandSpec]:
    command = _build_subject_step("B3_voicing.py", "B3_voicing", ctx)
    if command:
        command.argv.extend(["--target-rate", "100"])
    return [command] if command else []


def step_b4(ctx: PipelineContext) -> List[CommandSpec]:
    if ctx.glove_path is None:
        raise RuntimeError(
            "GloVe embeddings required for B4_glove.py. "
            "Provide --glove-path, set GLOVE_PATH, or add a glove_path.txt pointer."
        )
    argv = [
        ctx.python_executable,
        "B4_glove.py",
        "--glove-path",
        str(ctx.glove_path),
    ]
    if ctx.subject_ids:
        argv.extend(["--subjects", *ctx.subject_ids])
    argv.extend(["--target-rate", "100"])
    if ctx.overwrite:
        argv.append("--overwrite")
    return [CommandSpec("B4_glove", argv)]


def step_c1(ctx: PipelineContext) -> List[CommandSpec]:
    commands: List[CommandSpec] = []
    for label in ctx.subject_labels:
        argv = [ctx.python_executable, "C1_dRSA_run.py", label]
        commands.append(CommandSpec(f"C1_dRSA[{label}]", argv))
    return commands


def step_d1(ctx: PipelineContext) -> List[CommandSpec]:
    subjects = ctx.resolve_group_subjects()
    if not subjects:
        LOGGER.info("Skipping D1_group_cluster_analysis.py — no group subjects found.")
        return []

    argv = [
        ctx.python_executable,
        "D1_group_cluster_analysis.py",
        "--subjects",
        *subjects,
        "--models",
        *ctx.models,
        "--lag-metric",
        ctx.lag_metric,
        "--results-dir",
        str(ctx.results_dir),
    ]
    if ctx.d1_output is not None:
        argv.extend(["--output", str(ctx.d1_output)])
    if ctx.d1_n_permutations is not None:
        argv.extend(["--n-permutations", str(ctx.d1_n_permutations)])

    return [CommandSpec("D1_group_analysis", argv)]


PIPELINE_STEPS: List[PipelineStep] = [
    PipelineStep("A1", step_a1, "Preprocess raw runs"),
    PipelineStep("A2", step_a2, "Concatenate derivatives per subject"),
    PipelineStep("A3", step_a3, "Resample concatenated derivatives"),
    PipelineStep("B1", step_b1, "Build gammatone envelopes"),
    PipelineStep("B2", step_b2, "Compute word-frequency trajectories"),
    PipelineStep("B3", step_b3, "Compute phoneme voicing trajectories"),
    PipelineStep("B4", step_b4, "Compute GloVe embeddings"),
    PipelineStep("C1", step_c1, "Run subject-level dRSA"),
    PipelineStep("D1", step_d1, "Run group cluster analysis"),
]


def run_command(command: CommandSpec, continue_on_error: bool) -> None:
    """Execute a command, relaying stdout/stderr directly."""

    env = os.environ.copy()
    if command.env:
        env.update(command.env)

    LOGGER.info("Running %s: %s", command.label, " ".join(command.argv))
    try:
        subprocess.run(
            command.argv,
            cwd=command.cwd,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Step '%s' failed with exit code %s.", command.label, exc.returncode)
        if continue_on_error:
            return
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the MEG-MASC analysis pipeline across multiple subjects."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        required=True,
        help="Subject identifiers or ranges (e.g., 2-23 25).",
    )
    parser.add_argument(
        "--glove-path",
        help="Path to the GloVe embedding text file "
        "(falls back to $GLOVE_PATH or glove_path.txt if omitted).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Forward the overwrite flag to steps that support it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not abort the pipeline when a step fails.",
    )
    parser.add_argument(
        "--group-subjects",
        nargs="+",
        help="Override the subjects used for group analysis (supports ranges like 1-23). "
        "By default, the wrapper scans the results directory for available subjects.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing subject-level dRSA outputs for D1 (default: results).",
    )
    parser.add_argument(
        "--lag-metric",
        default="correlation",
        help="Lag metric suffix used by D1 (default: correlation).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model labels passed to D1 (default: %(default)s).",
    )
    parser.add_argument(
        "--d1-output",
        help="Optional override for the D1 summary figure path.",
    )
    parser.add_argument(
        "--d1-n-permutations",
        type=int,
        help="Optional override for the number of permutations used by D1.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for the wrapper itself.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        subject_ids = parse_subject_tokens(args.subjects)
    except ValueError as exc:
        LOGGER.error("Failed to parse --subjects: %s", exc)
        return 2

    repo_root = read_repository_root()
    LOGGER.debug("Repository root resolved to %s", repo_root)

    glove_path = resolve_glove_path(args.glove_path, repo_root)
    if glove_path:
        LOGGER.info("Using GloVe embeddings at %s", glove_path)
    else:
        LOGGER.info("No GloVe path resolved yet; will raise if B4_glove.py is reached.")

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (repo_root / results_dir).resolve()

    d1_output = Path(args.d1_output).resolve() if args.d1_output else None

    context = PipelineContext(
        repo_root=repo_root,
        python_executable=sys.executable,
        subject_ids=subject_ids,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
        glove_path=glove_path,
        group_subject_tokens=args.group_subjects,
        results_dir=results_dir,
        lag_metric=args.lag_metric,
        models=args.models,
        d1_output=d1_output,
        d1_n_permutations=args.d1_n_permutations,
    )

    try:
        for step in PIPELINE_STEPS:
            commands = build_step_commands(step, context)
            if not commands:
                continue
            LOGGER.info("=== %s: %s ===", step.name, step.description)
            for command in commands:
                run_command(command, context.continue_on_error)
    except subprocess.CalledProcessError:
        LOGGER.error("Pipeline aborted due to a failing command.")
        return 1

    LOGGER.info("Pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
