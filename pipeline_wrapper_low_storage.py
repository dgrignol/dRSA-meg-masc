#!/usr/bin/env python3
"""
Low-storage wrapper for the MEG-MASC analysis pipeline.

This helper processes subjects sequentially so only one subject's derivatives
are present on disk at any time. For each requested subject it runs:

    A1_preprocess_data.py
    A2_concatenate_subject_data.py
    A3_resample_concatenated_data.py
    B1_model_envelope.py
    B2_wordfreq.py
    B3_voicing.py
    B4_glove.py
    C1_dRSA_run.py

After C1 finishes, all intermediate derivatives for that subject are deleted,
keeping only the dRSA outputs written to ``results/``. Once every subject has
been processed, the wrapper launches the group-level D1 step.

Compared to ``pipeline_wrapper.py`` this mode trades runtime efficiency for a
drastically smaller storage footprint.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from functions.generic_helpers import (
    ensure_analysis_directories,
    generate_timestamped_analysis_name,
    read_repository_root,
)
from pipeline_wrapper import (
    DEFAULT_MODELS,
    discover_group_subjects,
    parse_subject_tokens,
    resolve_glove_path,
)


LOGGER = logging.getLogger("pipeline_wrapper_low_storage")


def run_command(label: str, argv: Sequence[str], cwd: Path, continue_on_error: bool) -> bool:
    """Execute a subprocess command and report success."""

    LOGGER.info("Running %s: %s", label, " ".join(argv))
    try:
        subprocess.run(list(argv), cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Step '%s' failed with exit code %s.", label, exc.returncode)
        if continue_on_error:
            return False
        raise
    return True


def cleanup_subject_derivatives(repo_root: Path, subject_label: str, remove_reports: bool) -> None:
    """
    Delete bulky intermediate files for ``subject_label`` while keeping the
    100 Hz concatenates and model trajectories needed for future dRSA runs.
    """

    derivatives_root = repo_root / "derivatives"
    preproc_subject_dir = derivatives_root / "preprocessed" / subject_label

    def _collect(paths: Iterable[Path]) -> List[Path]:
        return [path for path in paths if path.exists()]

    removal_map: List[tuple[str, List[Path]]] = []

    if preproc_subject_dir.exists():
        concatenated_dir = preproc_subject_dir / "concatenated"
        removal_map.append(
            (
                "concatenated MEG (native rate)",
                _collect(concatenated_dir.glob(f"{subject_label}_concatenated_meg.npy")),
            )
        )
        removal_map.append(
            (
                "run-level preprocessed FIF",
                _collect(preproc_subject_dir.glob("ses-*/task-*/meg/*.fif")),
            )
        )

    model_root = derivatives_root / "Models"
    model_patterns: List[tuple[str, Path, str]] = [
        ("envelope native arrays", model_root / "envelope" / subject_label, "**/*_envelope_native.npy"),
        ("envelope MEG-rate arrays", model_root / "envelope" / subject_label, "**/*_envelope_megfs.npy"),
        ("wordfreq MEG-rate arrays", model_root / "wordfreq" / subject_label, "**/*_wordfreq_megfs.npy"),
        ("voicing MEG-rate arrays", model_root / "voicing" / subject_label, "**/*_voicing_megfs.npy"),
    ]
    for description, base_dir, pattern in model_patterns:
        if base_dir.exists():
            removal_map.append((description, _collect(base_dir.glob(pattern))))

    for description, paths in removal_map:
        for target in sorted(paths):
            if not target.is_file():
                continue
            try:
                target.unlink()
                LOGGER.debug("Removed %s: %s", description, target)
            except OSError as exc:
                LOGGER.warning("Failed to remove %s (%s): %s", description, target, exc)

    if remove_reports:
        report_dirs = [
            derivatives_root / "reports" / "preprocessing" / subject_label,
            derivatives_root / "reports" / "Models" / "envelope" / subject_label,
            derivatives_root / "reports" / "Models" / "wordfreq" / subject_label,
            derivatives_root / "reports" / "Models" / "voicing" / subject_label,
        ]
        for directory in report_dirs:
            if directory.exists():
                LOGGER.debug("Removing report directory %s", directory)
                shutil.rmtree(directory, ignore_errors=True)


def build_subject_commands(
    python_executable: str,
    subject_id: str,
    subject_label: str,
    glove_path: Path,
    overwrite: bool,
    lock_subsample_to_word_onset: bool,
    allow_overlap: bool,
    analysis_name: str,
    results_root: Path,
) -> Iterable[tuple[str, List[str]]]:
    """Yield (label, argv) tuples for all per-subject steps."""

    overwrite_flag = ["--overwrite"] if overwrite else []
    yield (
        f"A1_preprocess[{subject_label}]",
        [python_executable, "A1_preprocess_data.py", "--subjects", subject_id, *overwrite_flag],
    )
    yield (
        f"A2_concatenate[{subject_label}]",
        [python_executable, "A2_concatenate_subject_data.py", "--subject", subject_label, *overwrite_flag],
    )
    yield (
        f"A3_resample[{subject_label}]",
        [python_executable, "A3_resample_concatenated_data.py", "--subject", subject_label, *overwrite_flag],
    )
    yield (
        f"B1_envelope[{subject_label}]",
        [python_executable, "B1_model_envelope.py", "--subjects", subject_id, *overwrite_flag],
    )
    yield (
        f"B2_wordfreq[{subject_label}]",
        [
            python_executable,
            "B2_wordfreq.py",
            "--subjects",
            subject_id,
            "--target-rate",
            "100",
            *overwrite_flag,
        ],
    )
    yield (
        f"B3_voicing[{subject_label}]",
        [
            python_executable,
            "B3_voicing.py",
            "--subjects",
            subject_id,
            "--target-rate",
            "100",
            *overwrite_flag,
        ],
    )
    yield (
        f"B4_glove[{subject_label}]",
        [
            python_executable,
            "B4_glove.py",
            "--subjects",
            subject_id,
            "--glove-path",
            str(glove_path),
            "--target-rate",
            "100",
            *overwrite_flag,
        ],
    )
    yield (
        f"C1_dRSA[{subject_label}]",
        [
            python_executable,
            "C1_dRSA_run.py",
            subject_label,
            "--analysis-name",
            analysis_name,
            "--results-root",
            str(results_root),
            *(
                ["--lock-subsample-to-word-onset"]
                if lock_subsample_to_word_onset
                else []
            ),
            *(
                ["--allow-overlap"]
                if allow_overlap
                else []
            ),
        ],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the MEG-MASC pipeline one subject at a time, deleting intermediates after C1."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        required=True,
        help="Subject identifiers or ranges (e.g., 2-23 25).",
    )
    parser.add_argument(
        "--glove-path",
        help="Path to the GloVe embedding file. Falls back to $GLOVE_PATH or glove_path.txt.",
    )
    parser.add_argument(
        "--analysis-name",
        help=(
            "Name of the analysis folder to create or reuse under --results-root."
            " Defaults to a timestamp (recommended to provide one)."
        ),
    )
    parser.add_argument(
        "--results-root",
        "--results-dir",
        dest="results_root",
        default="results",
        help="Parent directory containing analysis runs (default: results).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model labels passed to D1 (default: %(default)s).",
    )
    parser.add_argument(
        "--lag-metric",
        default="correlation",
        help="Lag metric suffix used by D1 (default: correlation).",
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
        "--overwrite",
        action="store_true",
        help="Forward the overwrite flag to steps that support it.",
    )
    parser.add_argument(
        "--lock-subsample-to-word-onset",
        action="store_true",
        help="Lock C1 subsample starts to word onsets.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow overlapping subsamples in C1.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining subjects if a step fails.",
    )
    parser.add_argument(
        "--keep-derivatives",
        action="store_true",
        help="Skip the cleanup phase (useful for debugging).",
    )
    parser.add_argument(
        "--keep-reports",
        action="store_true",
        help="Retain HTML/PDF reports generated in the derivatives directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for the wrapper itself.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    repo_root = read_repository_root()
    python_executable = sys.executable

    try:
        subject_ids = parse_subject_tokens(args.subjects)
    except ValueError as exc:
        LOGGER.error("Failed to parse --subjects: %s", exc)
        return 2

    glove_path = resolve_glove_path(args.glove_path, repo_root)
    if glove_path is None:
        LOGGER.error("GloVe embeddings required for B4_glove.py; provide --glove-path or configure GLOVE_PATH.")
        return 2
    LOGGER.info("Using GloVe embeddings at %s", glove_path)

    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = (repo_root / results_root).resolve()
    else:
        results_root = results_root.resolve()

    requested_analysis_name = args.analysis_name
    candidate_name = requested_analysis_name or generate_timestamped_analysis_name()
    (
        analysis_name,
        analysis_root,
        single_subjects_dir,
        group_level_dir,
    ) = ensure_analysis_directories(results_root, candidate_name)

    if requested_analysis_name and analysis_name != requested_analysis_name:
        LOGGER.info(
            "Sanitised analysis name '%s' to '%s'.",
            requested_analysis_name,
            analysis_name,
        )
    LOGGER.info(
        "Using analysis '%s' (subject outputs: %s, group outputs: %s)",
        analysis_name,
        single_subjects_dir,
        group_level_dir,
    )

    processed_subjects: List[str] = []
    failed_subjects: List[str] = []

    for subject_id in subject_ids:
        subject_label = f"sub-{subject_id}"
        LOGGER.info("=== Processing %s ===", subject_label)
        subject_success = True
        for label, command in build_subject_commands(
            python_executable,
            subject_id,
            subject_label,
            glove_path,
            args.overwrite,
            args.lock_subsample_to_word_onset,
            args.allow_overlap,
            analysis_name,
            results_root,
        ):
            if not run_command(label, command, repo_root, args.continue_on_error):
                subject_success = False
                break

        if subject_success:
            processed_subjects.append(subject_id)
            if not args.keep_derivatives:
                cleanup_subject_derivatives(repo_root, subject_label, remove_reports=not args.keep_reports)
        else:
            failed_subjects.append(subject_id)
            if not args.continue_on_error:
                LOGGER.error("Aborting due to failure while processing %s.", subject_label)
                return 1
            LOGGER.warning("Continuing after failure; derivatives for %s were retained for inspection.", subject_label)

    if failed_subjects:
        LOGGER.warning(
            "The following subjects failed and will be excluded from D1: %s",
            ", ".join(failed_subjects),
        )

    available_subjects = discover_group_subjects(single_subjects_dir, args.lag_metric)
    if not available_subjects:
        LOGGER.error(
            "No subject results found in %s matching '*_res*_%s_dRSA_matrices.npy'; skipping D1.",
            single_subjects_dir,
            args.lag_metric,
        )
        return 1

    missing_from_results = sorted(set(processed_subjects) - set(available_subjects))
    if missing_from_results:
        LOGGER.warning(
            "The following processed subjects did not produce D1 inputs and will be skipped: %s",
            ", ".join(f"sub-{sid}" for sid in missing_from_results),
        )

    LOGGER.info("=== Running D1 on %s ===", ", ".join(f"sub-{sid}" for sid in available_subjects))
    d1_command: List[str] = [
        python_executable,
        "D1_group_cluster_analysis.py",
        "--subjects",
        *available_subjects,
        "--models",
        *args.models,
        "--lag-metric",
        args.lag_metric,
        "--analysis-name",
        analysis_name,
        "--results-root",
        str(results_root),
    ]
    if args.d1_output:
        d1_command.extend(["--output", args.d1_output])
    if args.d1_n_permutations is not None:
        d1_command.extend(["--n-permutations", str(args.d1_n_permutations)])

    if not run_command("D1_group_analysis", d1_command, repo_root, args.continue_on_error):
        LOGGER.error("D1 failed.")
        return 1

    LOGGER.info("Low-storage pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
