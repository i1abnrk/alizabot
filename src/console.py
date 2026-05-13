"""Interactive v0.1.2 console: index each line, then reply (stdlib only)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from . import config
from .config import apply_db_cli_arg, build_arg_parser
from .picker import generate_reply

PROMPT = "\nYou> "
_QUIT = frozenset({"/quit", "/exit", "bye", "goodbye"})


def run_console_iteration(
	input_fn: Callable[[str], str] = input,
	print_fn: Callable[..., None] = print,
) -> bool:
	"""One read / generate_reply cycle. Returns False when the session should end."""
	user_input = input_fn(PROMPT).strip()
	if user_input.lower() in _QUIT:
		print_fn("Aliza> Goodbye!")
		return False
	if not user_input:
		return True
	reply = generate_reply(user_input)
	print_fn(f"Aliza> {reply}")
	return True


def run_console() -> int | None:
	parser = build_arg_parser()
	parser.add_argument(
		"--test",
		action="store_true",
		help="Run built-in verification (v0.1.1 DB smoke + console I/O harness).",
	)
	args = parser.parse_args()
	apply_db_cli_arg(args.db)
	config.DB_PATH = Path(config.DB_PATH).expanduser().resolve()

	if args.test:
		from .test_console import run_all_tests

		return run_all_tests()

	while True:
		try:
			if not run_console_iteration():
				break
		except KeyboardInterrupt:
			print("\nAliza> Caught Ctrl+C — shutting down gracefully.")
			break
		except Exception as e:
			print(f"Aliza> Oops... {e}")
	return None


if __name__ == "__main__":
	rc = run_console()
	if rc is not None:
		sys.exit(rc)
