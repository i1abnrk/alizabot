# AlizaBot

**General purpose classical chatbot engine** — a non-standard, interpretable weighted n-gram implementation.

This is a from-scratch Python rebuild of the original `net.sf.alizagameapi` inference engine, focused on controllable, personality-consistent dialogue using distance-weighted co-occurrence.

## Features

- Word-level tokenization (whitespace boundaries)
- 5-distance prior neighbor co-occurrence indexing
- Incremental indexing (only processes changed/new files)
- SQLite-backed storage with full incremental support
- Designed for eventual neuro-symbolic + evolutionary extensions

## Quick Start (Windows PowerShell)

```powershell
# 1. Clone & setup
git clone https://github.com/i1abnrk/alizabot.git
cd alizabot

# 2. (Optional) Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare your training texts
mkdir my_corpus
# Add your .txt files (e.g. from Project Gutenberg)

# 5. Build / update the index
python -m src.cli --data-dir "./my_corpus" --db-path "artifacts/index.sqlite"
```

## Important Flags

```powershell
# Normal incremental build (recommended)
python -m src.cli --data-dir "./my_corpus" --db-path "artifacts/index.sqlite"

# Force clean rebuild (if you changed tokenization rules or want to start fresh)
python -m src.cli --data-dir "./my_corpus" --db-path "artifacts/index.sqlite" --force-rebuild
```

## Quick Start — live console (v0.1.2)

Cross-platform: use `pathlib`-style paths; default DB is `./artifacts/index.sqlite` relative to the current working directory unless you override it.

**Linux / macOS (bash):**

```bash
export ALIZABOT_DB_PATH=./artifacts/index.sqlite
python -m alizabot.console --db ./artifacts/index.sqlite
```

**Windows (PowerShell):**

```powershell
$env:ALIZABOT_DB_PATH = "./artifacts/index.sqlite"
python -m alizabot.console --db ./artifacts/index.sqlite
```

Each line you type is indexed first, then the reply is generated from the updated database (stdlib SQLite only).