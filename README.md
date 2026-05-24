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

## Using ownsearch from AI coding agents (skills)

`ownsearch` is the *retrieval* half of a RAG: instead of building a separate vector-DB stack, you expose this CLI to your coding agent as a **skill** so it knows to search your indexed docs (instead of grepping blindly) and how to call it. The `--json` output is designed exactly for this.

Claude Code, [opencode](https://opencode.ai/docs/), and [Pi](https://pi.dev/) all support the **Agent Skills standard**: a `SKILL.md` Markdown file with `name` + `description` frontmatter. The same skill works in all three — only the install location and invocation differ.

### The skill file

Create `ownsearch/SKILL.md`:

```markdown
---
name: ownsearch
description: Search the user's locally indexed documentation with hybrid full-text + semantic search. Use this BEFORE grepping or guessing when a question is likely answered in the indexed docs — how something is deployed, configured or operated, infra details, runbooks, past decisions.
---

# ownsearch — local hybrid documentation search

`ownsearch` (already in PATH) searches the user's indexed docs with FTS5 (lexical)
+ semantic embeddings. Reach for it when an answer probably lives in the corpus.

## How to search

Prefer hybrid search with JSON output so you can parse hits programmatically:

    ownsearch search --json --both "your query here"

- `--both`     combine lexical + semantic, deduplicated (best default)
- `--semantic` semantic only (related content with different wording)
- (no flag)    fast literal FTS5 only
- `--dir PATH` scope to one indexed directory
- `--limit N`  cap results
- `--json`     machine-readable hits (file path + chunk); always use from a tool flow

Each JSON hit gives the source file path and the matching chunk. Open the file to
get full context before answering — this is retrieval only; reason over the results
yourself, don't treat a single chunk as the whole answer.

## Keeping the index fresh

If results look stale or a recently edited doc is missing:

    ownsearch index     # incremental
    ownsearch status    # DB size, indexed dirs, chunk/embedding counts, ollama health

## Discover what's indexed

    ownsearch list-dirs
```

### Where to put it, per agent

| Agent | Location (user-level) | Project-level | Invocation |
|-------|-----------------------|---------------|------------|
| **Claude Code** | `~/.claude/skills/ownsearch/SKILL.md` | `.claude/skills/ownsearch/SKILL.md` | auto-discovered; or `/ownsearch` |
| **opencode** | `~/.config/opencode/skills/ownsearch/SKILL.md` | `.opencode/skills/ownsearch/SKILL.md` | auto-discovered |
| **Pi** | `~/.pi/agent/skills/ownsearch/SKILL.md` | — | `/skill:ownsearch`, or auto-discovered |

> Claude Code also accepts a flat `~/.claude/skills/ownsearch.md` (no subdirectory). The `ownsearch/SKILL.md` directory form is the portable one that works across all three agents.

To avoid permission prompts on every call, allowlist the read-only commands in your
agent's settings — e.g. for Claude Code add `Bash(ownsearch search:*)` and
`Bash(ownsearch status:*)` to `permissions.allow`.

### opencode/Pi alternative: a slash command

If you prefer an explicit command over an auto-discovered skill, both opencode
(`~/.config/opencode/commands/ownsearch.md`) and Claude Code support command-style
Markdown where the filename becomes `/ownsearch`. A skill is usually better here
because the agent invokes it *on its own* when a question matches the `description`.

## Troubleshooting

### `HTTP Error 500` / some chunks never get embeddings

A 500 during `ownsearch index` usually comes from the **ollama embedding server**, not
ownsearch. Two distinct causes:

- **Transient** (server busy, model briefly evicted from VRAM, OOM): ownsearch retries
  with backoff, and any file whose embeddings failed is automatically re-indexed on the
  next `ownsearch index` run (it is *not* marked as up-to-date).
- **Permanent / content-specific**: some embedding models (notably `bge-m3` under
  ollama) emit `NaN` for certain token sequences, and ollama then returns
  `failed to encode response: json: unsupported value: NaN` (HTTP 500). Retrying never
  helps, so ownsearch skips just that chunk (logged as *"Skipping unembeddable chunk"*)
  and leaves it **FTS-searchable but not semantic**. The rest of the file is unaffected.

To find chunks that are missing an embedding (excluding short ones, which are skipped
by design): they stay searchable via plain FTS5, so this is rarely worth chasing. If a
specific important chunk is affected, lightly rewording it (e.g. punctuation) usually
sidesteps the model's NaN.

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
