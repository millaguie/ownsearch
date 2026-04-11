# ownsearch

Smart local search with full-text search (SQLite FTS5) and semantic search (embeddings via ollama). Zero external dependencies — Python stdlib only.

## Installation

```bash
pipx install /path/to/ownsearch
# or from the project directory:
pipx install .
```

## Initial setup

```bash
# Configure ollama (if not running on localhost:11434)
ownsearch config set ollama_url http://your-ollama-host:11434

# Configure embedding model (default: bge-m3)
ownsearch config set embed_model bge-m3

# Configure database path (default: ~/.ownsearch.db)
ownsearch config set db_path /custom/path.db

# Add directories to index
ownsearch add-dir ~/Documents/notes
ownsearch add-dir ~/workspace/project

# Show current configuration
ownsearch config show
```

Configuration is stored in `~/.config/ownsearch/config.json`.

## Usage

```bash
# Index (incremental — only new/modified/deleted files)
ownsearch index

# Force full re-index
ownsearch index --full

# Full-text search (fast, literal)
ownsearch search "kubernetes cilium"

# Semantic search (finds related content even with different wording)
ownsearch search --semantic "network security"

# Combined search (FTS + semantic, deduplicated)
ownsearch search --both "migration strategy"

# Filter results by directory
ownsearch search --dir ~/workspace/project "deploy"

# JSON output (for integration with other tools/agents)
ownsearch search --json "query"

# Limit results
ownsearch search --limit 5 "query"

# Show status
ownsearch status
```

## Directory management

```bash
ownsearch add-dir PATH      # Add a directory to the index
ownsearch remove-dir PATH   # Remove a directory and its data from the index
ownsearch list-dirs         # List indexed directories
```

## Smart behavior

- **Auto-pull models**: If ollama is reachable but the embedding model is missing, it pulls it automatically during indexing.
- **Incremental indexing**: By default, only processes files whose mtime/size changed since the last run. Deleted files are cleaned up automatically.
- **Graceful degradation**: If ollama is unavailable, FTS5 search still works (semantic search is skipped).
- **Smart chunking**: Splits by markdown headings. Large files are partitioned into ~4000 char chunks while preserving heading context.
- **Retry with backoff**: Embedding requests retry on failure with exponential backoff to handle transient server issues.

## Supported file types

Default: `.md`, `.txt`, `.org`, `.rst`

Configurable in `~/.config/ownsearch/config.json` (`extensions` field).

## Requirements

- Python >= 3.10 (stdlib only, no external packages)
- ollama (optional, for semantic search)

### Why bge-m3?

The default embedding model is `bge-m3` (~1.2GB). It was chosen after benchmarking against `nomic-embed-text`, `mxbai-embed-large`, and `snowflake-arctic-embed2` on a real multilingual corpus (Spanish/English mixed documents). Results:

- **nomic-embed-text**: Essentially useless for non-English content — returned random results for Spanish queries.
- **mxbai-embed-large**: Good scores but introduced noise on technical queries (e.g., kubernetes results mixed with unrelated content).
- **snowflake-arctic-embed2**: Precise results but lower overall scores.
- **bge-m3**: Best balance — top results were consistently correct for both Spanish and English queries, with clean ranking and no noise.

You can change the model with `ownsearch config set embed_model <model>`. Embeddings are automatically invalidated and regenerated on the next index run when the model changes.

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
