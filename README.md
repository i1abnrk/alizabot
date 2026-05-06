## Custom token co-occurrence indexer (t-1..t-5)

This project builds a relational co-occurrence index from plain text sources. For every token, it records counts of the 5 nearest prior neighbors at distances 1..5 (i.e., positions t-1 through t-5), producing a 5-distance frequency for each token pair.

### What it does
- Tokenizes `.txt` files from a folder (recursive).
- Normalizes tokens (lowercase by default).
- Stores a unique `tokens` vocabulary.
- Builds a `cooccurrence` table with `(token_id, neighbor_id, distance, count)` where `distance ∈ {1..5}` counts neighbors at t-distance.
- Uses SQLite for a simple, portable relational store.

### Quick start (Windows PowerShell)
1) (Optional) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install requirements (none required; file kept for future use)

```powershell
pip install -r requirements.txt
```

3) Place your `.txt` files under a folder, e.g. `data/`

4) Run the indexer

```powershell
python -m src.cli --data-dir ".\data" --db-path ".\artifacts\index.sqlite"
```

- The script will create the SQLite database and populate `tokens` and `cooccurrence`.
- Reruns will upsert counts (you can safely run multiple times over the same data directory).

### Database schema
- `tokens(id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL)`
- `cooccurrence(token_id INTEGER NOT NULL, neighbor_id INTEGER NOT NULL, distance INTEGER NOT NULL CHECK(distance BETWEEN 1 AND 5), count INTEGER NOT NULL, PRIMARY KEY(token_id, neighbor_id, distance))`

Indexes are created on `(token_id, distance)` and `(neighbor_id)`.

### Notes
- Distances recorded are strictly prior neighbors (t-1..t-5) to match the specified t(-1, -5) interval.
- Tokenization is regex-based (alphanumeric plus underscore) and lowercased by default.
- If you need forward neighbors (t+1..t+5) as well, we can extend the schema with a `direction` column or allow negative distances.

