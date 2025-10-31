from functools import lru_cache
from datetime import datetime
from pathlib import Path
import re
from typing import Optional, Union, Tuple


def _iter_candidate_paths(pointer: Path):
    """
    Yield meaningful candidate paths listed in data_path.txt.

    Empty lines and comments (starting with '#') are ignored. Inline comments are
    supported: everything after a '#' on the same line is dropped.
    """
    for raw_line in pointer.read_text().splitlines():
        candidate = raw_line.split("#", 1)[0].strip()
        if candidate:
            yield candidate


def read_repository_root() -> Path:
    """
    Resolve the repository root using the candidate paths in data_path.txt.

    The first existing path is returned so the same code can run on multiple
    machines (e.g., local workstation + cluster) without editing source files.
    """

    pointer = Path(__file__).parent.parent / "data_path.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            "data_path.txt is missing. Create the file with the absolute path(s) to the repository root."
        )

    tried: list[Path] = []
    for candidate in _iter_candidate_paths(pointer):
        path = Path(candidate).expanduser().resolve()
        tried.append(path)
        if path.exists() and (path / "data_path.txt").exists():
            return path

    tried_str = ", ".join(str(p) for p in tried) if tried else "<no candidates>"
    raise FileNotFoundError(
        "None of the candidate repository roots listed in data_path.txt exist. "
        f"Checked: {tried_str}"
    )


def rebase_path_to_known_root(path: Union[str, Path]) -> Path:
    """
    When ``path`` points inside the repository but uses a root that is not
    currently mounted (e.g., ``/Volumes`` vs ``/mnt``), substitute the root
    with another candidate from ``data_path.txt`` that yields an existing file.
    """

    candidate_path = Path(path)
    if candidate_path.exists() or not candidate_path.is_absolute():
        return candidate_path

    pointer = Path(__file__).parent.parent / "data_path.txt"
    if not pointer.exists():
        return candidate_path

    candidate_roots = [Path(c).expanduser().resolve() for c in _iter_candidate_paths(pointer)]

    relative_path = None
    for root in candidate_roots:
        try:
            relative_path = candidate_path.relative_to(root)
        except ValueError:
            continue
        else:
            break

    if relative_path is None:
        return candidate_path

    alternate_with_parent: Optional[Path] = None
    candidate_parent_exists = candidate_path.parent.exists()

    for root in candidate_roots:
        remapped = (root / relative_path).resolve()
        if remapped.exists():
            return remapped
        if alternate_with_parent is None and remapped.parent.exists():
            alternate_with_parent = remapped

    if not candidate_parent_exists and alternate_with_parent is not None:
        return alternate_with_parent

    return candidate_path


@lru_cache(maxsize=1024)
def _cached_directory_listing(directory: str) -> tuple[str, ...]:
    """Return child entry names for a directory, falling back to empty on failure."""

    try:
        path = Path(directory)
        return tuple(entry.name for entry in path.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return tuple()


def _casefold_lookup(entries: tuple[str, ...], target: str) -> Optional[str]:
    """Find a case-insensitive match for ``target`` within ``entries``."""

    for name in entries:
        if name == target:
            return name

    target_cf = target.casefold()
    for name in entries:
        if name.casefold() == target_cf:
            return name
    return None


def resolve_relative_path_casefold(base: Path, relative: Union[str, Path]) -> Optional[Path]:
    """
    Resolve ``relative`` under ``base`` while ignoring casing differences.

    Parameters
    ----------
    base : Path
        Directory that serves as the starting point for relative paths.
    relative : str or Path
        Relative (or absolute) path that may differ in casing from on-disk entries.

    Returns
    -------
    Path or None
        Concrete path if every component could be matched; ``None`` otherwise.
    """

    relative_path = Path(relative)
    if not relative_path.parts:
        return base

    if relative_path.is_absolute():
        current = Path(relative_path.anchor or "/")
        parts_iter = relative_path.parts[1:]
    else:
        current = base.resolve()
        parts_iter = relative_path.parts

    for part in parts_iter:
        if part in ("", "."):
            continue
        if part == "..":
            current = current.parent
            continue

        entries = _cached_directory_listing(str(current))
        match = _casefold_lookup(entries, part)
        if match is None:
            return None
        current = current / match

    return current


_ANALYSIS_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_ANALYSIS_TIMESTAMP_PATTERN = re.compile(r"^(\d{8})_(\d{6})(?:[._-].*)?$")


def generate_timestamped_analysis_name(moment: Optional[datetime] = None) -> str:
    """Return a default analysis name using ``YYYYMMDD_HHMMSS``."""

    reference = moment or datetime.now()
    return reference.strftime(_ANALYSIS_TIMESTAMP_FORMAT)


def normalise_analysis_name(raw_name: str) -> str:
    """
    Sanitize ``raw_name`` so it can be safely used as a directory name.

    The function replaces characters outside ``[A-Za-z0-9_-]`` with underscores,
    collapses consecutive underscores, and strips leading/trailing separators.
    ``ValueError`` is raised when the name becomes empty or resolves to a reserved
    path fragment (``.`` or ``..``).
    """

    candidate = str(raw_name).strip()
    if not candidate:
        raise ValueError("Analysis name cannot be empty.")

    candidate = candidate.replace("/", "_").replace("\\", "_")
    candidate = re.sub(r"[^\w\-]+", "_", candidate)
    candidate = re.sub(r"_+", "_", candidate)
    candidate = candidate.strip("_.-")

    if not candidate or candidate in {".", ".."}:
        raise ValueError(
            "Analysis name must contain at least one alphanumeric character after sanitisation."
        )

    return candidate


def ensure_analysis_directories(
    results_root: Path, analysis_name: Optional[str] = None
) -> Tuple[str, Path, Path, Path]:
    """
    Create and return the directory layout for a dRSA analysis run.

    Parameters
    ----------
    results_root:
        Base directory under which the ``analysis_name`` folder resides.
    analysis_name:
        Optional user-specified name. When omitted, a timestamped default is used.

    Returns
    -------
    Tuple[str, Path, Path, Path]
        Normalised analysis name, analysis root, ``single_subjects`` path,
        and ``group_level`` path.
    """

    results_root = Path(results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    if analysis_name:
        resolved_name = normalise_analysis_name(analysis_name)
    else:
        resolved_name = generate_timestamped_analysis_name()

    analysis_root = results_root / resolved_name
    single_subjects_dir = analysis_root / "single_subjects"
    group_level_dir = analysis_root / "group_level"

    single_subjects_dir.mkdir(parents=True, exist_ok=True)
    group_level_dir.mkdir(parents=True, exist_ok=True)

    return resolved_name, analysis_root, single_subjects_dir, group_level_dir


def find_latest_analysis_directory(results_root: Path) -> Optional[Path]:
    """
    Return the most recent analysis directory under ``results_root``.

    Timestamp-style names (``YYYYMMDD_HHMMSS``) are prioritised and compared
    lexicographically. When no timestamped folders are present, the directory
    list falls back to modification time. ``None`` is returned if no candidates
    exist or ``results_root`` is missing.
    """

    results_root = Path(results_root).expanduser().resolve()
    if not results_root.exists():
        return None

    timestamped: list[tuple[datetime, Path]] = []
    fallback: list[tuple[float, Path]] = []

    for entry in results_root.iterdir():
        if not entry.is_dir():
            continue
        match = _ANALYSIS_TIMESTAMP_PATTERN.fullmatch(entry.name)
        if match:
            timestamp_str = f"{match.group(1)}_{match.group(2)}"
            try:
                timestamp = datetime.strptime(timestamp_str, _ANALYSIS_TIMESTAMP_FORMAT)
            except ValueError:
                fallback.append((entry.stat().st_mtime, entry))
            else:
                timestamped.append((timestamp, entry))
            continue

        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        fallback.append((mtime, entry))

    if timestamped:
        timestamped.sort(key=lambda item: item[0], reverse=True)
        return timestamped[0][1]
    if fallback:
        fallback.sort(key=lambda item: item[0], reverse=True)
        return fallback[0][1]
    return None
