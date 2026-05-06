from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


def iter_text_files(data_dir: Path) -> Iterable[Path]:
    """Yield .txt files recursively under data_dir."""
    for path in data_dir.rglob("*.txt"):
        if path.is_file():
            yield path


def newest_text_mtime(data_dir: Path) -> Optional[float]:
    """Return the newest mtime across .txt files, or None when empty."""
    newest: Optional[float] = None
    for text_file in iter_text_files(data_dir):
        mtime = text_file.stat().st_mtime
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def ensure_parent_dir(path: Path) -> None:
    """Ensure that path's parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text_robust(path: Path) -> str:
    """
    Read text with fallback encodings.

    Attempts UTF-8, Latin-1, then Windows-1252. If all strict attempts fail,
    performs a final UTF-8 decode with errors ignored.
    """
    encodings: List[str] = ["utf-8", "latin-1", "windows-1252"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
