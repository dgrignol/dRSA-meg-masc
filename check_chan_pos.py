#!/usr/bin/env python3
"""Visualise MEG channel positions for a BIDS subject/session/task."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.viz import get_3d_backend, set_3d_title

from functions.generic_helpers import read_repository_root, resolve_relative_path_casefold

LOGGER = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot KIT/Yokogawa MEG channel geometry in 3D using MNE-Python's "
            "plot_alignment helper (helmet + sensors)."
        )
    )
    parser.add_argument("subject", help="Subject label (with or without the 'sub-' prefix, e.g. '01').")
    parser.add_argument(
        "--session",
        default="0",
        help="Session label without the 'ses-' prefix (default: 0).",
    )
    parser.add_argument(
        "--task",
        default="0",
        help="Task label without the 'task-' prefix (default: 0).",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Optional run label without the 'run-' prefix.",
    )
    parser.add_argument(
        "--acquisition",
        "--acq",
        dest="acq",
        default=None,
        help="Optional acquisition label (without 'acq-').",
    )
    parser.add_argument(
        "--processing",
        "--proc",
        dest="proc",
        default=None,
        help="Optional processing label (without 'proc-').",
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=None,
        help="Explicit path to the BIDS root. Defaults to <repo>/bids_anonym.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Preferred MNE 3D backend (e.g., pyvistaqt). Defaults to trying pyvistaqt then notebook.",
    )
    parser.add_argument(
        "--allow-multi",
        action="store_true",
        help="If multiple runs match the BIDS filters, use the first instead of failing.",
    )
    return parser


def _normalise_entity(value: Optional[str], prefix: str) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    prefix_token = f"{prefix}-"
    if cleaned.lower().startswith(prefix_token):
        cleaned = cleaned[len(prefix_token) :]
    return cleaned


def _resolve_bids_root(user_choice: Optional[Path]) -> Path:
    if user_choice is not None:
        if user_choice.is_absolute():
            root = user_choice
        else:
            root = resolve_relative_path_casefold(Path.cwd(), user_choice) or user_choice
        return root.resolve()
    repo_root = read_repository_root()
    return (repo_root / "bids_anonym").resolve()


def _describe_bids_run(bids_path: BIDSPath) -> str:
    parts: list[str] = [f"sub-{bids_path.subject}"]
    entity_prefixes = (
        ("session", "ses"),
        ("task", "task"),
        ("run", "run"),
        ("acquisition", "acq"),
        ("processing", "proc"),
    )
    for attr, prefix in entity_prefixes:
        value = getattr(bids_path, attr, None)
        if value:
            parts.append(f"{prefix}-{value}")
    return ", ".join(parts)


def _ensure_3d_backend(preferred: Optional[str]) -> bool:
    if preferred:
        candidates = [preferred]
    else:
        candidates = ["pyvistaqt", "notebook"]

    failures: list[tuple[str, str]] = []
    for backend in candidates:
        if backend is None:
            continue
        try:
            mne.viz.set_3d_backend(backend)
        except ValueError as exc:
            failures.append((backend, str(exc)))
            continue
        LOGGER.info("Using MNE 3D backend: %s", backend)
        return True

    if failures:
        LOGGER.error("Could not initialise any 3D backend.")
        for backend, reason in failures:
            LOGGER.error("  %s: %s", backend, reason)
    LOGGER.error("Install 'pyvistaqt' (pip or conda) for the recommended desktop backend.")
    return False


def _plot_alignment_3d(raw: mne.io.BaseRaw, title: str) -> bool:
    LOGGER.info("Opening KIT/Yokogawa 3D alignment (helmet + sensors).")
    try:
        fig = mne.viz.plot_alignment(
            info=raw.info,
            trans=None,
            coord_frame="head",
            meg=("helmet", "sensors"),
            dig=True,
            show_axes=False,
            verbose="error",
        )
    except Exception as exc:  # pragma: no cover - backend/runtime specific
        backend = get_3d_backend()
        LOGGER.error("Failed to render alignment with backend '%s': %s", backend, exc)
        return False

    try:
        set_3d_title(figure=fig, title=title)
    except Exception as exc:  # pragma: no cover - backend/runtime specific
        LOGGER.debug("Could not set 3D title: %s", exc)
    return True


def _load_raw(bids_path: BIDSPath) -> mne.io.BaseRaw:
    LOGGER.info("Loading %s", bids_path.basename)
    raw = read_raw_bids(bids_path, verbose="error")
    raw.load_data()
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not _ensure_3d_backend(args.backend):
        return 2

    bids_root = _resolve_bids_root(args.bids_root)
    if not bids_root.exists():
        LOGGER.error("BIDS root %s does not exist", bids_root)
        return 1
    subject = _normalise_entity(args.subject, "sub")
    session = _normalise_entity(args.session, "ses")
    task = _normalise_entity(args.task, "task")
    run = _normalise_entity(args.run, "run")
    acq = _normalise_entity(args.acq, "acq")
    proc = _normalise_entity(args.proc, "proc")

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        processing=proc,
        suffix="meg",
        datatype="meg",
        root=bids_root,
    )

    matches: Iterable[BIDSPath] = bids_path.match()
    matches = tuple(matches)
    if not matches:
        LOGGER.error("No MEG recordings found for %s", bids_path)
        return 1

    if len(matches) > 1 and not args.allow_multi:
        LOGGER.error(
            "Multiple recordings match the filters (%d). Use --allow-multi to pick the first.",
            len(matches),
        )
        for candidate in matches:
            LOGGER.error("  %s", candidate.basename)
        return 1

    if len(matches) > 1:
        LOGGER.warning("Multiple recordings found, using %s", matches[0].basename)

    selected_path = matches[0]
    raw = _load_raw(selected_path)

    run_label = _describe_bids_run(selected_path)

    success = _plot_alignment_3d(raw, f"{run_label} (KIT)")
    if not success:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
