"""AlizaBot - Classical weighted n-gram chatbot engine."""

from .config import DB_PATH, apply_db_cli_arg, build_arg_parser

# === DEPRECATED (Live Path exports) ===
# These are the old live-path entry points.
# They are being phased out in favor of the single engine in inference.py.
# from .indexer import index_text, index_token_sequence
# from .picker import WordPicker, generate_reply
# ======================================

__version__ = "0.1.3"

__all__ = [
	"DB_PATH",
	# "WordPicker",           # old stub - deprecated
	"apply_db_cli_arg",
	"build_arg_parser",
	# "generate_reply",       # old live path - deprecated
	# "index_text",
	# "index_token_sequence",
]