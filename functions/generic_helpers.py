from pathlib import Path


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
