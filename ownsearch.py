#!/usr/bin/env python3
"""ownsearch — Smart full-text and semantic search across your local documents."""

import argparse
import json
import math
import os
import re
import sqlite3
import struct
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

__version__ = "0.1.0"

# Defaults
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "ownsearch"
DEFAULT_DB_PATH = Path.home() / ".ownsearch.db"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024
MAX_CHUNK_CHARS = 4000
BATCH_SIZE = 5

INDEXABLE_EXTS = {".md", ".txt", ".org", ".rst"}
SKIP_DIRS = {".git", ".obsidian", ".claude", "node_modules", "__pycache__", ".venv", "venv"}


class Config:
    """Manages ownsearch configuration."""

    def __init__(self):
        self.config_dir = DEFAULT_CONFIG_DIR
        self.config_file = self.config_dir / "config.json"
        self.data = self._load()

    def _load(self):
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text())
            except (json.JSONDecodeError, OSError):
                return self._defaults()
        return self._defaults()

    def _defaults(self):
        return {
            "db_path": str(DEFAULT_DB_PATH),
            "ollama_url": DEFAULT_OLLAMA_URL,
            "embed_model": DEFAULT_EMBED_MODEL,
            "directories": [],
            "extensions": list(INDEXABLE_EXTS),
            "skip_dirs": list(SKIP_DIRS),
        }

    def save(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))

    @property
    def db_path(self):
        return Path(self.data["db_path"])

    @property
    def ollama_url(self):
        return self.data["ollama_url"]

    @property
    def embed_model(self):
        return self.data["embed_model"]

    @property
    def directories(self):
        return [Path(d) for d in self.data["directories"]]

    @property
    def extensions(self):
        return set(self.data["extensions"])

    @property
    def skip_dirs(self):
        return set(self.data["skip_dirs"])


# --- Ollama helpers ---


def ollama_available(config):
    """Check if ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{config.ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            json.loads(resp.read())
        return True
    except Exception:
        return False


def ollama_has_model(config):
    """Check if the configured embedding model is available."""
    try:
        req = urllib.request.Request(f"{config.ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return any(config.embed_model in m for m in models)
    except Exception:
        return False


def ollama_pull_model(config):
    """Pull the embedding model. Returns True on success."""
    print(f"  Pulling model '{config.embed_model}' from ollama...")
    data = json.dumps({"name": config.embed_model, "stream": False}).encode()
    req = urllib.request.Request(
        f"{config.ollama_url}/api/pull",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
        if "error" in result:
            print(f"  Error pulling model: {result['error']}", file=sys.stderr)
            return False
        print(f"  Model '{config.embed_model}' ready.")
        return True
    except Exception as e:
        print(f"  Failed to pull model: {e}", file=sys.stderr)
        return False


def ensure_embeddings_ready(config):
    """Ensure ollama is available and has the embedding model. Auto-pulls if needed."""
    if not ollama_available(config):
        print(f"  Ollama not reachable at {config.ollama_url}. Semantic search disabled.", file=sys.stderr)
        return False

    if ollama_has_model(config):
        return True

    print(f"  Model '{config.embed_model}' not found in ollama.", file=sys.stderr)
    print(f"  Attempting to pull it automatically...", file=sys.stderr)
    return ollama_pull_model(config)


# Sentinel: the server rejected this specific input deterministically (e.g.
# bge-m3 emits NaN for certain token sequences and ollama can't serialize it).
# Retrying never helps, so we fail fast and leave the chunk FTS-only. It is an
# (empty) list subclass so it stays falsy and type-compatible with real results;
# callers tell it apart from a normal empty/None result via `is PERMANENT_FAIL`.
class _PermanentFail(list):
    pass


PERMANENT_FAIL = _PermanentFail()


def get_embeddings_batch(config, texts):
    """Get embeddings for a batch of texts from ollama. Truncates and retries on failure.

    Returns a list aligned with ``texts``; each item is the embedding vector, or
    ``None`` for a transient failure (worth retrying later), or ``PERMANENT_FAIL``
    when the model can't embed that specific text.
    """
    # Truncate texts to avoid OOM on the server
    truncated = [t[:2000] for t in texts]

    # Try batch first
    embeddings = _embed_request(config, truncated)
    if embeddings is not PERMANENT_FAIL and embeddings and len(embeddings) == len(texts):
        return embeddings

    # Batch failed — fall back to one-by-one. A single poisoned text (NaN) makes
    # ollama 500 the whole batch, so isolating per-text salvages the rest.
    results = []
    for text in truncated:
        vec = _embed_request(config, text)
        if vec is PERMANENT_FAIL:
            results.append(PERMANENT_FAIL)
        elif vec:
            results.append(vec[0] if isinstance(vec[0], list) else vec)
        else:
            results.append(None)
        time.sleep(0.1)  # Avoid overwhelming the server
    return results


def _embed_request(config, input_data, retries=5):
    """Raw embed request with exponential backoff retry.

    Returns the embeddings list on success, ``[]`` on transient failure (after
    exhausting retries), or ``PERMANENT_FAIL`` when the server reports a
    deterministic, content-specific error (NaN serialization) — retrying those
    only wastes the backoff budget, so we bail immediately.
    """
    data = json.dumps({"model": config.embed_model, "input": input_data}).encode()

    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{config.ollama_url}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            return result.get("embeddings", [])
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")
            except Exception:
                pass
            if "NaN" in body or "unsupported value" in body:
                print(
                    f"  Skipping unembeddable chunk (model returned NaN): {body.strip()[:120]}",
                    file=sys.stderr,
                )
                return PERMANENT_FAIL
            err = e
        except Exception as e:
            err = e

        if attempt < retries:
            wait = min(2 ** (attempt + 1), 60)  # 2s, 4s, 8s, 16s, 32s
            print(f"  Retry {attempt + 1}/{retries} in {wait}s: {err}", file=sys.stderr)
            time.sleep(wait)
            continue
        print(f"  Warning: embedding failed after {retries + 1} attempts: {err}", file=sys.stderr)
        return []


# --- Database ---


def init_db(conn):
    conn.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            directory TEXT NOT NULL,
            mtime_ns INTEGER,
            size INTEGER
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            heading TEXT,
            content TEXT NOT NULL,
            FOREIGN KEY (file_path) REFERENCES files(path) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            vector BLOB NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    )
    if not cur.fetchone():
        conn.executescript("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                heading,
                file_path UNINDEXED,
                content=chunks,
                content_rowid=id,
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, heading, file_path)
                VALUES (new.id, new.content, new.heading, new.file_path);
            END;

            CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, heading, file_path)
                VALUES ('delete', old.id, old.content, old.heading, old.file_path);
            END;

            CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, heading, file_path)
                VALUES ('delete', old.id, old.content, old.heading, old.file_path);
                INSERT INTO chunks_fts(rowid, content, heading, file_path)
                VALUES (new.id, new.content, new.heading, new.file_path);
            END;
        """)
    conn.commit()


# --- Chunking ---


def chunk_markdown(text):
    """Split markdown into chunks by headings, with breadcrumb context."""
    chunks = []
    parts = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    current_heading = ""
    current_text = ""
    heading_stack = []

    for part in parts:
        heading_match = re.match(r'^(#{1,3})\s+(.+)$', part)
        if heading_match:
            if current_text.strip():
                for sub in _split_large(current_text.strip(), current_heading):
                    chunks.append(sub)

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            current_heading = " > ".join(h[1] for h in heading_stack)
            current_text = ""
        else:
            current_text += part

    if current_text.strip():
        for sub in _split_large(current_text.strip(), current_heading):
            chunks.append(sub)

    if not chunks and text.strip():
        for sub in _split_large(text.strip(), ""):
            chunks.append(sub)

    return chunks


def chunk_plaintext(text):
    """Split plain text into chunks at paragraph boundaries."""
    return _split_large(text.strip(), "")


def _split_large(text, heading):
    """Split text exceeding MAX_CHUNK_CHARS at paragraph boundaries."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [(heading, text)]

    result = []
    paragraphs = re.split(r'\n\n+', text)
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > MAX_CHUNK_CHARS and current:
            result.append((heading, current.strip()))
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        result.append((heading, current.strip()))

    return result if result else [(heading, text[:MAX_CHUNK_CHARS])]


# --- Vector math ---


def pack_vector(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(blob):
    n = len(blob) // 4
    return struct.unpack(f"{n}f", blob)


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --- File walking ---


def walk_directory(dir_path, extensions, skip_dirs):
    """Walk a directory and yield (absolute_path, relative_path, stat) for indexable files."""
    dir_path = dir_path.resolve()
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            p = Path(root) / f
            if p.suffix in extensions:
                rel = str(p.relative_to(dir_path))
                st = p.stat()
                yield str(p), rel, st.st_mtime_ns, st.st_size


# --- Commands ---


def cmd_add_dir(args, config):
    """Add a directory to the search index."""
    path = Path(args.path).resolve()
    if not path.is_dir():
        print(f"Error: '{path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    dirs = config.data["directories"]
    path_str = str(path)
    if path_str in dirs:
        print(f"Directory already indexed: {path}")
        return

    dirs.append(path_str)
    config.save()
    print(f"Added: {path}")
    print(f"Run 'ownsearch index' to index it.")


def cmd_remove_dir(args, config):
    """Remove a directory from the search index."""
    path = Path(args.path).resolve()
    path_str = str(path)

    dirs = config.data["directories"]
    if path_str not in dirs:
        # Try matching by suffix
        matches = [d for d in dirs if d.endswith(args.path) or d.endswith(args.path.rstrip("/"))]
        if matches:
            path_str = matches[0]
        else:
            print(f"Directory not found in config: {args.path}", file=sys.stderr)
            sys.exit(1)

    dirs.remove(path_str)
    config.save()

    # Remove from DB
    conn = sqlite3.connect(str(config.db_path))
    init_db(conn)
    conn.execute("DELETE FROM chunks WHERE file_path IN (SELECT path FROM files WHERE directory = ?)", (path_str,))
    conn.execute("DELETE FROM files WHERE directory = ?", (path_str,))
    conn.commit()
    conn.close()
    print(f"Removed: {path_str}")


def cmd_list_dirs(config):
    """List indexed directories."""
    if not config.directories:
        print("No directories configured. Use: ownsearch add-dir PATH")
        return
    for d in config.directories:
        exists = "✓" if Path(d).exists() else "✗"
        print(f"  {exists} {d}")


def cmd_config_show(config):
    """Show current configuration."""
    print(json.dumps(config.data, indent=2, ensure_ascii=False))


def cmd_config_set(args, config):
    """Set a configuration value."""
    key = args.key
    value = args.value

    valid_keys = {"db_path", "ollama_url", "embed_model"}
    if key not in valid_keys:
        print(f"Invalid key. Valid keys: {', '.join(sorted(valid_keys))}", file=sys.stderr)
        sys.exit(1)

    old_value = config.data.get(key)
    config.data[key] = value
    config.save()
    print(f"Set {key} = {value}")

    # If embed model changed, invalidate all embeddings
    if key == "embed_model" and old_value and old_value != value:
        if config.db_path.exists():
            conn = sqlite3.connect(str(config.db_path))
            conn.execute("DELETE FROM embeddings")
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)", (value,))
            conn.commit()
            conn.close()
            print(f"Embedding model changed ({old_value} -> {value}). All embeddings cleared.")
            print("Run 'ownsearch index --full' to re-generate embeddings.")


def cmd_index(args, config):
    """Index all configured directories."""
    if not config.directories:
        print("No directories configured. Use: ownsearch add-dir PATH")
        sys.exit(1)

    # Ensure DB directory exists
    config.db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(config.db_path))
    init_db(conn)

    # Check if embed model changed since last index
    stored_model = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    if stored_model and stored_model[0] != config.embed_model:
        print(f"Embedding model changed ({stored_model[0]} -> {config.embed_model}). Clearing old embeddings.")
        conn.execute("DELETE FROM embeddings")
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)", (config.embed_model,))
        conn.commit()

    # Check embeddings availability (auto-pull if needed)
    has_embeddings = ensure_embeddings_ready(config)

    # Gather all current files across all directories
    current_files = {}  # absolute_path -> (dir_str, mtime_ns, size)
    for dir_path in config.directories:
        dp = Path(dir_path)
        if not dp.exists():
            print(f"  Warning: directory not found, skipping: {dir_path}", file=sys.stderr)
            continue
        dir_str = str(dp)
        for abs_path, _, mtime_ns, size in walk_directory(dp, config.extensions, config.skip_dirs):
            current_files[abs_path] = (dir_str, mtime_ns, size)

    # Get stored file states
    stored = {}
    for row in conn.execute("SELECT path, mtime_ns, size FROM files"):
        stored[row[0]] = (row[1], row[2])

    # Determine changes
    to_index = []
    to_remove = []

    if args.full:
        to_index = list(current_files.keys())
        to_remove = list(stored.keys())
    else:
        for path, (dir_str, mtime, size) in current_files.items():
            if path not in stored or stored[path] != (mtime, size):
                to_index.append(path)
        for path in stored:
            if path not in current_files:
                to_remove.append(path)

    if not to_index and not to_remove:
        print("Index is up to date.")
        conn.close()
        return

    print(f"Indexing: {len(to_index)} files to process, {len(to_remove)} to remove")

    # Remove deleted/changed files
    for path in to_remove + to_index:
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (path,))
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
    conn.commit()

    # Index new/changed files
    total_chunks = 0
    embed_queue = []
    failed_ids = []  # chunk_ids whose embedding failed during this run

    for i, path in enumerate(to_index, 1):
        full_path = Path(path)
        try:
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError) as e:
            print(f"  Skip {path}: {e}", file=sys.stderr)
            continue

        dir_str, mtime_ns, size = current_files[path]
        conn.execute(
            "INSERT INTO files (path, directory, mtime_ns, size) VALUES (?, ?, ?, ?)",
            (path, dir_str, mtime_ns, size),
        )

        # Choose chunking strategy based on file type
        if full_path.suffix == ".md":
            chunks = chunk_markdown(text)
        else:
            chunks = chunk_plaintext(text)

        for idx, (heading, content) in enumerate(chunks):
            cur = conn.execute(
                "INSERT INTO chunks (file_path, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                (path, idx, heading, content),
            )
            total_chunks += 1
            if has_embeddings and len(content.strip()) >= 50:
                # Prefix heading for better embedding context
                embed_text = f"{heading}: {content}" if heading else content
                embed_queue.append((cur.lastrowid, embed_text[:8000]))

            if has_embeddings and len(embed_queue) >= BATCH_SIZE:
                failed_ids += _process_embed_batch(config, conn, embed_queue)
                embed_queue = []

        if i % 20 == 0:
            print(f"  {i}/{len(to_index)} files...")
            conn.commit()

    if has_embeddings and embed_queue:
        failed_ids += _process_embed_batch(config, conn, embed_queue)

    # Any file with a failed embedding is unmarked (mtime_ns = -1) so the next
    # incremental run re-indexes it. We keep its chunks for now, so the file
    # stays searchable (FTS + whatever embeddings did succeed) in the meantime.
    if failed_ids:
        placeholders = ",".join("?" * len(failed_ids))
        failed_files = {
            row[0]
            for row in conn.execute(
                f"SELECT DISTINCT file_path FROM chunks WHERE id IN ({placeholders})",
                failed_ids,
            )
        }
        for fp in failed_files:
            conn.execute("UPDATE files SET mtime_ns = -1 WHERE path = ?", (fp,))
        print(
            f"  Warning: {len(failed_ids)} chunk(s) across {len(failed_files)} file(s) "
            f"failed to embed; those files will be re-indexed on the next run.",
            file=sys.stderr,
        )

    # Store the model used for these embeddings
    if has_embeddings:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)", (config.embed_model,))
    conn.commit()
    print(f"Done. {len(to_index)} files, {total_chunks} chunks indexed.")
    if has_embeddings:
        embed_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        print(f"  Embeddings: {embed_count}")
    conn.close()


def _process_embed_batch(config, conn, queue):
    """Embed a batch of (chunk_id, text) and store the vectors.

    Returns the list of chunk_ids that failed *transiently* (worth retrying),
    so the caller can avoid marking their source file as fully indexed —
    otherwise an incremental re-index would never retry them (the file's
    mtime/size already match). Permanent failures (the model can't embed that
    specific text) are skipped silently: their chunks stay FTS-only and the
    file is left marked as indexed, so we don't loop on them every run.
    """
    ids = [q[0] for q in queue]
    texts = [q[1] for q in queue]
    vectors = get_embeddings_batch(config, texts)
    if vectors and len(vectors) == len(ids):
        failed = []
        for chunk_id, vec in zip(ids, vectors):
            if vec is PERMANENT_FAIL:
                continue  # already logged; leave FTS-only, do not retry
            if vec is None:
                failed.append(chunk_id)
                continue
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                (chunk_id, pack_vector(vec)),
            )
        return failed
    # Whole batch failed transiently (server unreachable / length mismatch).
    return list(ids)


def cmd_search(args, config):
    """Search the index."""
    if not config.db_path.exists():
        print("No index found. Run: ownsearch index", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(config.db_path))
    query = " ".join(args.query)

    if not query:
        print("No query provided.", file=sys.stderr)
        sys.exit(1)

    if args.semantic:
        results = search_semantic(config, conn, query, args.limit)
    elif args.both:
        fts_results = search_fts(conn, query, args.limit)
        sem_results = search_semantic(config, conn, query, args.limit)
        results = merge_results(fts_results, sem_results, args.limit)
    else:
        results = search_fts(conn, query, args.limit)

    # Apply directory filter if specified
    if args.dir:
        filter_dir = str(Path(args.dir).resolve())
        results = [r for r in results if r["path"].startswith(filter_dir)]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        format_results(results, config)

    conn.close()


def search_fts(conn, query, limit=10):
    """Full-text search using FTS5 BM25."""
    fts_query = query.replace('"', '""')

    try:
        rows = conn.execute(
            """
            SELECT c.file_path, c.heading, snippet(chunks_fts, 0, '>>>', '<<<', '...', 40) as snip,
                   bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        try:
            rows = conn.execute(
                """
                SELECT c.file_path, c.heading, snippet(chunks_fts, 0, '>>>', '<<<', '...', 40) as snip,
                       bm25(chunks_fts) as score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (f'"{fts_query}"', limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

    results = []
    for path, heading, snippet_text, score in rows:
        clean_snippet = snippet_text.replace(">>>", "").replace("<<<", "")
        results.append({
            "path": path,
            "heading": heading or "",
            "snippet": clean_snippet,
            "score": round(-score, 4),
            "method": "fts",
        })
    return results


def search_semantic(config, conn, query, limit=10):
    """Semantic search using embeddings."""
    vectors = get_embeddings_batch(config, [query])
    if not vectors or vectors[0] is PERMANENT_FAIL or not vectors[0]:
        print("Semantic search unavailable (ollama not reachable, model missing, or query not embeddable).", file=sys.stderr)
        return []

    query_vec = vectors[0]

    rows = conn.execute(
        """
        SELECT e.chunk_id, e.vector, c.file_path, c.heading, c.content
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        """
    ).fetchall()

    if not rows:
        print("No embeddings in index. Re-run: ownsearch index", file=sys.stderr)
        return []

    scored = []
    for chunk_id, vec_blob, path, heading, content in rows:
        vec = unpack_vector(vec_blob)
        sim = cosine_sim(query_vec, vec)
        scored.append((sim, path, heading, content))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim, path, heading, content in scored[:limit]:
        snippet = content[:200].replace("\n", " ")
        results.append({
            "path": path,
            "heading": heading or "",
            "snippet": snippet,
            "score": round(sim, 4),
            "method": "semantic",
        })
    return results


def merge_results(fts_results, sem_results, limit=10):
    """Merge results using reciprocal rank fusion."""
    K = 60
    scores = {}
    all_results = {}

    for rank, r in enumerate(fts_results):
        key = (r["path"], r["heading"])
        scores[key] = scores.get(key, 0) + 1.0 / (K + rank + 1)
        if key not in all_results:
            all_results[key] = r

    for rank, r in enumerate(sem_results):
        key = (r["path"], r["heading"])
        scores[key] = scores.get(key, 0) + 1.0 / (K + rank + 1)
        if key not in all_results:
            all_results[key] = r

    merged = []
    for key, r in all_results.items():
        r = r.copy()
        r["score"] = round(scores.get(key, 0), 4)
        r["method"] = "combined"
        merged.append(r)

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:limit]


def format_results(results, config):
    """Print results in human-readable format."""
    if not results:
        print("No results found.")
        return

    for r in results:
        score = r["score"]
        path = r["path"]
        heading = r["heading"]
        snippet = r["snippet"]

        print(f"\033[1;33m[{score:.2f}]\033[0m \033[1m{path}\033[0m")
        if heading:
            print(f"       \033[36m{heading}\033[0m")
        print(f"       {snippet[:200]}")
        print()


def cmd_status(config):
    """Show status of ownsearch."""
    print(f"ownsearch v{__version__}")
    print(f"Config: {config.config_file}")
    print(f"Database: {config.db_path}", end="")
    if config.db_path.exists():
        size_mb = config.db_path.stat().st_size / 1024 / 1024
        print(f" ({size_mb:.1f} MB)")
    else:
        print(" (not created)")

    print(f"\nOllama: {config.ollama_url}", end="")
    if ollama_available(config):
        print(" ✓", end="")
        if ollama_has_model(config):
            print(f" (model '{config.embed_model}' ready)")
        else:
            print(f" (model '{config.embed_model}' NOT found — will auto-pull on index)")
    else:
        print(" ✗ unreachable")

    print(f"\nDirectories ({len(config.directories)}):")
    if config.directories:
        conn = None
        if config.db_path.exists():
            conn = sqlite3.connect(str(config.db_path))
        for d in config.data["directories"]:
            exists = "✓" if Path(d).exists() else "✗"
            count = ""
            if conn:
                n = conn.execute("SELECT COUNT(*) FROM files WHERE directory = ?", (d,)).fetchone()[0]
                count = f" [{n} files]"
            print(f"  {exists} {d}{count}")
        if conn:
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embeds = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            print(f"\n  Total: {total} chunks, {embeds} embeddings")
            conn.close()
    else:
        print("  (none — use 'ownsearch add-dir PATH')")


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        prog="ownsearch",
        description="Smart full-text and semantic search for your local documents",
    )
    parser.add_argument("--version", action="version", version=f"ownsearch {__version__}")
    sub = parser.add_subparsers(dest="command")

    # add-dir
    add = sub.add_parser("add-dir", help="Add a directory to index")
    add.add_argument("path", help="Directory path to add")

    # remove-dir
    rm = sub.add_parser("remove-dir", help="Remove a directory from index")
    rm.add_argument("path", help="Directory path to remove")

    # list-dirs
    sub.add_parser("list-dirs", help="List indexed directories")

    # index
    idx = sub.add_parser("index", help="Index all configured directories")
    idx.add_argument("--full", action="store_true", help="Force full re-index")

    # search
    srch = sub.add_parser("search", help="Search the index")
    srch.add_argument("query", nargs="+", help="Search query")
    srch.add_argument("--semantic", action="store_true", help="Use semantic search")
    srch.add_argument("--both", action="store_true", help="Combined FTS + semantic")
    srch.add_argument("--json", action="store_true", help="Output as JSON")
    srch.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    srch.add_argument("--dir", help="Filter results to a specific directory")

    # config
    cfg = sub.add_parser("config", help="Show or set configuration")
    cfg_sub = cfg.add_subparsers(dest="config_command")
    cfg_sub.add_parser("show", help="Show current config")
    cfg_set = cfg_sub.add_parser("set", help="Set a config value")
    cfg_set.add_argument("key", help="Config key (db_path, ollama_url, embed_model)")
    cfg_set.add_argument("value", help="Config value")

    # status
    sub.add_parser("status", help="Show ownsearch status")

    args = parser.parse_args()
    config = Config()

    if args.command == "add-dir":
        cmd_add_dir(args, config)
    elif args.command == "remove-dir":
        cmd_remove_dir(args, config)
    elif args.command == "list-dirs":
        cmd_list_dirs(config)
    elif args.command == "index":
        cmd_index(args, config)
    elif args.command == "search":
        cmd_search(args, config)
    elif args.command == "config":
        if hasattr(args, "config_command") and args.config_command == "set":
            cmd_config_set(args, config)
        else:
            cmd_config_show(config)
    elif args.command == "status":
        cmd_status(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
