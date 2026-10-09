# ownsearch

Smart local search with full-text search (SQLite FTS5) and semantic search (embeddings via ollama). Zero external dependencies — Python stdlib only (document formats are an optional extra).

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

# Full-text search that requires every word (no OR fallback)
ownsearch search --strict "kubernetes cilium"

# Parallel embedding requests while indexing (default: 4)
ownsearch index --workers 8

# Semantic search (finds related content even with different wording)
ownsearch search --semantic "network security"

# Combined search (FTS + semantic, deduplicated)
ownsearch search --both "migration strategy"

# Filter results by directory
ownsearch search --dir ~/workspace/project "deploy"

# Up to N characters of each matching chunk instead of a short snippet
ownsearch search --json --max-chars 1500 "query"

# JSON output (for integration with other tools/agents)
ownsearch search --json "query"

# Limit results
ownsearch search --limit 5 "query"

# Show status
ownsearch status
ownsearch status --json   # same, machine-readable (includes last index time)
```

## Directory management

```bash
ownsearch add-dir PATH      # Add a directory to the index (--images: also OCR its images)
ownsearch remove-dir PATH   # Remove a directory and its data from the index
ownsearch list-dirs         # List indexed directories
```

## Smart behavior

- **Auto-pull models**: If ollama is reachable but the embedding model is missing, it pulls it automatically during indexing.
- **Incremental indexing**: By default, only processes files whose mtime/size changed since the last run. Deleted files are cleaned up automatically.
- **Graceful degradation**: If ollama is unavailable, FTS5 search still works (semantic search is skipped).
- **Smart chunking**: Splits by markdown headings. Large files are partitioned into ~4000 char chunks while preserving heading context.
- **Full-text fallback**: Full-text search first requires every word. If that finds nothing, it retries with any word, and BM25 usually ranks first the chunks with more of the words. Use `--strict` to turn this off. Matches in headings count double.
- **Old embeddings**: Up to 0.2.0, embeddings used only the first 2000 characters of each chunk. `index` and `status` warn about it. Run `ownsearch index --full` once to rebuild them.
- **Context across chunks** (optional): `ownsearch config set chunk_overlap 200` adds the last 200 characters of the previous chunk to the text that gets embedded. A section split in two keeps its context in both vectors. The stored text and the full-text index do not change. Run `ownsearch index --full` after changing it.
- **Parallel indexing**: Embedding requests run in parallel (`--workers`, or `ownsearch config set embed_workers N`).
- **Retry with backoff**: Embedding requests retry on failure with exponential backoff to handle transient server issues.

## Supported file types

Default: `.md`, `.txt`, `.org`, `.rst`

Configurable in `~/.config/ownsearch/config.json` (`extensions` field).

### Documents (PDF, Office, EPUB...)

Install the `docs` extra to also index documents:

```bash
pipx install 'ownsearch[docs]'
# or, if ownsearch is already installed with pipx:
pipx inject ownsearch docvortex
```

With it, ownsearch converts these files to Markdown with [DocVortex](https://github.com/myhloli/DocVortex) and indexes them by heading: `.pdf`, `.doc`, `.docx`, `.rtf`, `.ppt`, `.pptx`, `.xls`, `.xlsx`, `.odt`, `.ods`, `.odp`, `.html`, `.htm`, `.epub`, `.csv`, `.tsv`.

- No extra config: the formats are picked up on the next `ownsearch index`.
- Scanned PDFs go through OCR (docvortex detects them). If a PDF still gives no text, ownsearch retries it forcing OCR.
- The extra is heavy (~500 MB, it pulls OpenCV and NumPy). Conversion is slower than reading text, but only changed files are converted again.
- A file that fails to convert is skipped until it changes.

### Images (OCR)

Images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff`) are indexed only in the directories you turn on, because most images in a wiki are screenshots and icons:

```bash
ownsearch add-dir ~/Documents/scans --images   # also works on a directory already added
ownsearch list-dirs                            # shows [images] next to it
```

Images under 8 KB are skipped. Two OCR engines:

- `local` (default): the OCR models of docvortex, on the CPU. Needs the `docs` extra. The first use downloads the models (~230 MB); after that it works without network. About 1-4 s per image.
- `vlm`: a vision model behind an OpenAI-compatible `/v1/chat/completions` endpoint (for example Qwen-VL on vLLM or LiteLLM). Cleaner text and tables kept as Markdown. A vision model can also invent text, so check it on your own images first. With Pillow (it comes with the `docs` extra) images are reduced to 1600 px before sending; without it, only `.png`, `.jpg`, `.webp` and `.gif` under 5 MB are sent.

```bash
ownsearch config set ocr_engine vlm
ownsearch config set ocr_base_url https://your-gateway/v1
ownsearch config set ocr_model qwen3.8-27b
ownsearch config set ocr_api_key_cmd 'pass show my/gateway-key'   # or OWNSEARCH_OCR_API_KEY
```

If the endpoint does not answer or the engine is not available, the image is skipped and retried on the next `ownsearch index`. An image that fails by itself (a corrupt file) is skipped until it changes. So is an image the endpoint rejects with HTTP 400, 413, 415 or 422. If every image fails with HTTP 400, check that the model accepts images, fix it and run `ownsearch index --full`. If the engine is turned off, images leave the index until it is back. Changing the engine does not redo the images already indexed: run `ownsearch index --full` for that.

### Fast semantic search (numpy)

Install the `fast` extra for large indexes:

```bash
pipx install 'ownsearch[fast]'
# or, if ownsearch is already installed with pipx:
pipx inject ownsearch numpy
```

With numpy, semantic search keeps the vectors in a cache next to the database (`<db_path>.vectors.npy`) and compares them all in one matrix product. Without numpy, ownsearch compares them one by one in Python, which is slow beyond some tens of thousands of chunks.

## Requirements

- Python >= 3.10 (stdlib only, no external packages)
- docvortex (optional, `ownsearch[docs]`, for PDF/Office/EPUB)
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
- `--dir PATH` scope to one indexed directory (applied before `--limit`)
- `--max-chars N` return up to N characters of each chunk, enough to quote it
- `--limit N`  cap results
- `--json`     machine-readable hits; always use from a tool flow. Each hit has
  `path`, `heading`, `snippet`, `score`, `method`, `chunk_id` and `mtime` (file
  date, ISO 8601). With `--both`, `methods` says which searches found it; hits
  found by both are usually the most relevant.
- Exit code 3: semantic search was requested but the embedding backend is down.
  The hits (if any) are full-text only.

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
