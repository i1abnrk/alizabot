import os
from typing import Generator, Iterable, List, Tuple

from .tokenizer import tokenize


def iter_text_files(root_dir: str) -> Generator[str, None, None]:
	for base, _, files in os.walk(root_dir):
		for name in files:
			if name.lower().endswith(".txt"):
				yield os.path.join(base, name)


def load_tokens_from_file(path: str, lowercase: bool = True, min_len: int = 1) -> List[str]:
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		text = f.read()
	return tokenize(text, lowercase=lowercase, min_len=min_len)


def iter_token_sequences(
	root_dir: str, lowercase: bool = True, min_len: int = 1
) -> Generator[Tuple[str, List[str]], None, None]:
	for path in iter_text_files(root_dir):
		yield path, load_tokens_from_file(path, lowercase=lowercase, min_len=min_len)

