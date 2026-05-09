from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def canonical_index_path(path: Path) -> str:
    """
    Stable absolute path string for SQLite index keys.

    On Windows, paths are case-insensitive; normalize so the same file is not
    indexed twice under different casings (e.g. OneDrive / tooling quirks).
    """
    resolved = path.resolve()
    s = resolved.as_posix()
    if os.name == "nt":
        return s.casefold()
    return s


def file_stat_fingerprint(path: Path) -> Tuple[int, int]:
    """Return (size_bytes, mtime_ns) for robust change detection (no SQLite REAL rounding)."""
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


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


def total_text_bytes(data_dir: Path) -> int:
    """Sum of file sizes for all .txt files under data_dir (recursive)."""
    total = 0
    for p in iter_text_files(data_dir):
        total += p.stat().st_size
    return total


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
