"""Built-in verification for v0.1.1 DB/indexer smoke + v0.1.2 console I/O (stdlib only)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from . import config
from .console import run_console_iteration
from .db import chat_connection, chat_token_id, connect_db, init_schema
from .indexer import index_text, index_token_sequence


def run_v011_smoke() -> bool:
	"""Connect, report corpus totals, index a tiny sentence via v0.1.1 indexer path."""
	db_path = Path(config.DB_PATH)
	try:
		conn = connect_db(str(db_path))
		init_schema(conn)
		row_t = conn.execute("SELECT COUNT(*) FROM tokens").fetchone()
		row_c = conn.execute("SELECT COUNT(*) FROM cooccurrence").fetchone()
		if row_t is None or row_c is None:
			print("FAIL: v0.1.1 smoke — could not read counts.")
			return False
		n_tokens = int(row_t[0])
		n_co = int(row_c[0])
		print(f"v0.1.1 DB OK: {n_tokens} tokens, {n_co} cooccurrences")
		index_token_sequence(conn, "tiny smoke test phrase here".split())
		conn.close()
	except Exception as exc:
		print(f"FAIL: v0.1.1 smoke — {exc}")
		return False
	return True


def run_console_io_test() -> bool:
	"""Non-interactive harness: fake input(), capture stdout/stderr, assert Aliza lines and no stderr noise."""
	out = io.StringIO()
	err = io.StringIO()
	lines = ["", "hello aliza", "one two three", "/quit"]
	it = iter(lines)

	def fake_input(prompt: str = "") -> str:
		print(prompt, end="", file=out)
		try:
			return next(it)
		except StopIteration:
			raise EOFError("test input exhausted") from None

	old_err = sys.stderr
	try:
		sys.stderr = err
		for _ in range(len(lines)):
			if not run_console_iteration(input_fn=fake_input, print_fn=lambda *a, **k: print(*a, file=out, **k)):
				break
	except Exception as exc:
		print(f"FAIL: console I/O test — {exc}")
		return False
	finally:
		sys.stderr = old_err

	stdout_text = out.getvalue()
	stderr_text = err.getvalue()
	if stderr_text.strip():
		print(f"FAIL: console I/O test — unexpected stderr: {stderr_text!r}")
		return False
	if "\nYou> " not in stdout_text:
		print("FAIL: console I/O test — missing '\\nYou> ' prompt pattern.")
		return False
	if stdout_text.count("Aliza> ") < 3:
		print(
			"FAIL: console I/O test — expected at least three 'Aliza> ' lines "
			f"(replies + goodbye), got {stdout_text.count('Aliza> ')}."
		)
		return False
	if "Aliza> Goodbye!" not in stdout_text:
		print("FAIL: console I/O test — missing goodbye line.")
		return False
	return True


def run_injection_test() -> bool:
	"""Test SQL injection protection: index malicious input, verify it's stored as literal tokens."""
	try:
		# Test malicious inputs
		malicious_inputs = [
			"'; DROP TABLE tokens; --",
			"Robert'); DROP TABLE students;--"
		]
		for inp in malicious_inputs:
			index_text(inp)
		
		# Verify the malicious strings are stored as literal tokens
		with chat_connection(config.DB_PATH) as conn:
			for inp in malicious_inputs:
				tokens = inp.split()
				for token in tokens:
					token_id = chat_token_id(conn, token)
					if token_id is None:
						print(f"FAIL: injection test — token '{token}' not found in DB.")
						return False
					# Check that the token text matches exactly
					from .db import chat_token_text
					stored_text = chat_token_text(conn, token_id)
					if stored_text != token:
						print(f"FAIL: injection test — stored text '{stored_text}' != '{token}'.")
						return False
		
		print("Injection test: PASSED - malicious input safely tokenized")
		return True
	except Exception as exc:
		print(f"FAIL: injection test — {exc}")
		return False


def run_all_tests() -> int:
	"""Run smoke + I/O checks; print PASS/FAIL summary. Returns process exit code."""
	ok = True
	ok = run_v011_smoke() and ok
	ok = run_console_io_test() and ok
	ok = run_injection_test() and ok
	if ok:
		print("PASS: all verification modes succeeded.")
		return 0
	print("FAIL: one or more verification modes failed.")
	return 1


if __name__ == "__main__":
	exit(run_all_tests())
