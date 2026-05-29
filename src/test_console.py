"""Built-in verification for console I/O (stdlib only)."""

from __future__ import annotations

import io
import sys

from .console import run_console_iteration


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


def run_all_tests() -> int:
	"""Run console I/O checks; print PASS/FAIL summary."""
	ok = run_console_io_test()
	ok = run_console_io_test() and ok
	ok = run_injection_test() and ok
	if ok:
		print("PASS: all verification modes succeeded.")
		return 0
	print("FAIL: one or more verification modes failed.")
	return 1


if __name__ == "__main__":
	exit(run_all_tests())
