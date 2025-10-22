from functools import lru_cache
from pathlib import Path
from typing import Optional, Union


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
