"""Interactive console using the single WorkPicker + ChancePie engine.

MVP phase: One formula working.
Live user input should eventually be compiled into the main DB,
then replies generated via the Bayesian/WorkPicker path.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from . import config
from .config import apply_db_cli_arg, build_arg_parser
from .inference import WorkPicker

# === DEPRECATED import note ===
# Old: from .picker import generate_reply
# We are no longer using the abandoned live-path generate_reply.
# All generation now goes through WorkPicker in inference.py.
# ==================================

# === NOTE: Live indexing into main tables is still needed ===
# For the absolute MVP we are first getting the real engine generating replies.
# Proper live updates to the main cooccurrence tables will be added next.

PROMPT = "\nYou> "
_QUIT = frozenset({"/quit", "/exit", "bye", "goodbye"})

# Global engine instance (simple for MVP)
_engine: WorkPicker | None = None


def _get_engine() -> WorkPicker:
	global _engine
	if _engine is None:
		_engine = WorkPicker(str(config.DB_PATH))
	return _engine


def run_console_iteration(
	input_fn: Callable[[str], str] = input,
	print_fn: Callable[..., None] = print,
) -> bool:
	"""One read / generate reply cycle using the real WorkPicker engine."""
	user_input = input_fn(PROMPT).strip()
	if user_input.lower() in _QUIT:
		print_fn("Aliza> Goodbye!")
		return False
	if not user_input:
		return True

	# TODO (MVP follow-up): Compile user_input into main DB here
	# For now we just generate using whatever is already in the DB.

	engine = _get_engine()

	reply = engine.generate_reply(user_input)
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
