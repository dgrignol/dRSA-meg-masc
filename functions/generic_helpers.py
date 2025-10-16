from pathlib import Path

def read_repository_root() -> Path:
    pointer = Path(__file__).parent.parent / "data_path.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            "data_path.txt is missing. Create the file with the absolute path to the repository root."
        )
    root = Path(pointer.read_text().strip()).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {root}")
    return root
