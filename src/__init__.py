"""AlizaBot - Classical weighted n-gram chatbot engine."""

from .config import DB_PATH, apply_db_cli_arg, build_arg_parser

__version__ = "0.1.3"

__all__ = [
	"DB_PATH",
	"apply_db_cli_arg",
	"build_arg_parser",
]