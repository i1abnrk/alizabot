import re
from typing import Iterable, List

TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str, lowercase: bool = True, min_len: int = 1) -> List[str]:
	tokens = TOKEN_PATTERN.findall(text)
	if lowercase:
		tokens = [t.lower() for t in tokens]
	if min_len > 1:
		tokens = [t for t in tokens if len(t) >= min_len]
	return tokens


def detokenize(tokens: Iterable[str]) -> str:
	return " ".join(tokens)

