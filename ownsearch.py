#!/usr/bin/env python3
"""ownsearch — Smart full-text and semantic search across your local documents."""

import argparse
import base64
import io
import json
import math
import os
import re
import sqlite3
import subprocess
import struct
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Extra opcional `ownsearch[fast]`: busqueda semantica vectorizada. Se
# importa al primer uso: cuesta unos 80 ms, y una busqueda de texto (lo que
# mas lanzan los agentes, un proceso por consulta) no lo necesita.
_NP_UNSET = object()
np = _NP_UNSET


def _numpy():
    """El modulo numpy, o None si no esta instalado."""
    global np
    if np is _NP_UNSET:
        try:
            import numpy
        except ImportError:
            numpy = None
        np = numpy
    return np


__version__ = "0.2.0"

# Defaults
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "ownsearch"
DEFAULT_DB_PATH = Path.home() / ".ownsearch.db"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024
MAX_CHUNK_CHARS = 4000
# bge-m3 admite 8192 tokens. En texto latino 8000 caracteres caben y el
# fragmento entero (titulo incluido) llega al modelo; en escrituras densas
# (chino, japones...) el servidor puede recortar el final.
EMBED_MAX_CHARS = 8000
BATCH_SIZE = 5
DEFAULT_EMBED_WORKERS = 4
# Codigo de salida de `search` cuando se pidio busqueda semantica y no se
# pudo hacer.
EXIT_SEMANTIC_UNAVAILABLE = 3
SEARCH_EMBED_RETRIES = 1
# Peso de cada columna de chunks_fts en BM25: content, heading.
BM25_WEIGHTS = (1.0, 2.0)

INDEXABLE_EXTS = {".md", ".txt", ".org", ".rst"}
# Formatos que se convierten a Markdown con docvortex (extra opcional
# `ownsearch[docs]`). Se indexan solos cuando docvortex esta instalado.
DOCUMENT_EXTS = {
    ".pdf",
    ".doc",
    ".docx",
    ".rtf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".odt",
    ".ods",
    ".odp",
    ".html",
    ".htm",
    ".epub",
    ".csv",
    ".tsv",
}
# Imagenes: solo en las carpetas activadas con `add-dir --images`, porque en
# una wiki la mayoria son capturas e iconos.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
# Por debajo de esto suelen ser iconos o adornos sin texto util.
IMAGE_MIN_BYTES = 8 * 1024
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
OCR_PROMPT = (
    "Transcribe todo el texto de esta imagen, tal cual, en Markdown. "
    "Respeta tablas y titulos. No resumas ni comentes. "
    "Si no hay texto, responde exactamente: (sin texto)"
)
SKIP_DIRS = {
    ".git",
    ".obsidian",
    ".claude",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
}


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
            # "ollama" (API nativa) o "openai" (/v1/embeddings, que habla
            # cualquier gateway: LiteLLM, vLLM, OpenAI...).
            "embed_backend": "ollama",
            "embed_base_url": "",
            "embed_api_key": "",
            "embed_api_key_cmd": "",
            "embed_workers": DEFAULT_EMBED_WORKERS,
            "chunk_overlap": 0,
            # Carpetas con OCR de imagenes y motor: "local" (docvortex, sin
            # red) o "vlm" (modelo con vision por API compatible con OpenAI).
            "image_dirs": [],
            "ocr_engine": "local",
            "ocr_base_url": "",
            "ocr_model": "",
            "ocr_api_key_cmd": "",
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
    def embed_backend(self):
        return self.data.get("embed_backend", "ollama")

    @property
    def embed_base_url(self):
        """Base del backend OpenAI-compatible, sin barra final."""
        return (self.data.get("embed_base_url") or "").rstrip("/")

    @property
    def embed_api_key(self):
        """Clave del backend OpenAI, calculada una sola vez.

        Cada peticion de embeddings la pide, y con varios hilos ejecutar
        embed_api_key_cmd en cada una lanzaria decenas de `pass` a la vez.
        """
        if not hasattr(self, "_embed_api_key"):
            self._embed_api_key = self._read_embed_api_key()
        return self._embed_api_key

    def _read_embed_api_key(self):
        """Por orden: variable de entorno, comando (para sacarla de `pass` sin
        guardarla en el config) y, como ultimo recurso, el valor literal.
        """
        env = os.environ.get("OWNSEARCH_EMBED_API_KEY")
        if env:
            return env.strip()
        cmd = self.data.get("embed_api_key_cmd")
        if cmd:
            try:
                out = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=15
                )
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout.strip().splitlines()[0]
                print(
                    f"  Warning: embed_api_key_cmd fallo: {out.stderr.strip()[:120]}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"  Warning: embed_api_key_cmd fallo: {e}", file=sys.stderr)
        return (self.data.get("embed_api_key") or "").strip()

    @property
    def embed_workers(self):
        try:
            return max(1, int(self.data.get("embed_workers", DEFAULT_EMBED_WORKERS)))
        except (TypeError, ValueError):
            return DEFAULT_EMBED_WORKERS

    @property
    def image_dirs(self):
        return set(self.data.get("image_dirs") or [])

    @property
    def ocr_engine(self):
        return self.data.get("ocr_engine") or "local"

    @property
    def ocr_api_key(self):
        """Clave del modelo de OCR: variable de entorno o comando, una vez."""
        if not hasattr(self, "_ocr_api_key"):
            key = os.environ.get("OWNSEARCH_OCR_API_KEY", "").strip()
            cmd = self.data.get("ocr_api_key_cmd")
            if not key and cmd:
                try:
                    out = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=15
                    )
                    if out.returncode == 0 and out.stdout.strip():
                        key = out.stdout.strip().splitlines()[0]
                    else:
                        print(
                            f"  Warning: ocr_api_key_cmd fallo: {out.stderr.strip()[:120]}",
                            file=sys.stderr,
                        )
                except Exception as e:  # noqa: BLE001
                    print(f"  Warning: ocr_api_key_cmd fallo: {e}", file=sys.stderr)
            self._ocr_api_key = key
        return self._ocr_api_key

    def images_enabled(self):
        """True si hay motor de OCR para las carpetas con imagenes."""
        if self.ocr_engine == "vlm":
            return bool(self.data.get("ocr_base_url") and self.data.get("ocr_model"))
        return docvortex_available()

    @property
    def chunk_overlap(self):
        try:
            return max(0, int(self.data.get("chunk_overlap", 0)))
        except (TypeError, ValueError):
            return 0

    @property
    def directories(self):
        return [Path(d) for d in self.data["directories"]]

    @property
    def extensions(self):
        exts = set(self.data["extensions"])
        if docvortex_available():
            exts |= DOCUMENT_EXTS
        return exts

    @property
    def skip_dirs(self):
        return set(self.data["skip_dirs"])


# --- Ollama helpers ---


def ollama_available(config):
    """Check if the embedding backend is reachable."""
    if config.embed_backend == "openai":
        try:
            req = urllib.request.Request(f"{config.embed_base_url}/models")
            key = config.embed_api_key
            if key:
                req.add_header("Authorization", f"Bearer {key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                json.loads(resp.read())
            return True
        except Exception:
            return False
    try:
        req = urllib.request.Request(f"{config.ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            json.loads(resp.read())
        return True
    except Exception:
        return False


def ollama_has_model(config):
    """Check if the configured embedding model is available."""
    if config.embed_backend == "openai":
        try:
            req = urllib.request.Request(f"{config.embed_base_url}/models")
            key = config.embed_api_key
            if key:
                req.add_header("Authorization", f"Bearer {key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            return any(config.embed_model == m.get("id") for m in data.get("data", []))
        except Exception:
            return False
    try:
        req = urllib.request.Request(f"{config.ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return any(config.embed_model in m for m in models)
    except Exception:
        return False


def ollama_pull_model(config):
    """Pull the embedding model. Returns True on success.

    Un backend OpenAI-compatible no descarga modelos: los sirve quien los tenga.
    """
    if config.embed_backend == "openai":
        print(
            f"  El backend openai no descarga modelos: publica '{config.embed_model}' "
            f"en {config.embed_base_url} y vuelve a intentarlo.",
            file=sys.stderr,
        )
        return False
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
        print(
            f"  Ollama not reachable at {config.ollama_url}. Semantic search disabled.",
            file=sys.stderr,
        )
        return False

    if ollama_has_model(config):
        return True

    print(f"  Model '{config.embed_model}' not found in ollama.", file=sys.stderr)
    print("  Attempting to pull it automatically...", file=sys.stderr)
    return ollama_pull_model(config)


# Sentinel: the server rejected this specific input deterministically (e.g.
# bge-m3 emits NaN for certain token sequences and ollama can't serialize it).
# Retrying never helps, so we fail fast and leave the chunk FTS-only. It is an
# (empty) list subclass so it stays falsy and type-compatible with real results;
# callers tell it apart from a normal empty/None result via `is PERMANENT_FAIL`.
class _PermanentFail(list):
    pass


PERMANENT_FAIL = _PermanentFail()


def get_embeddings_batch(config, texts, retries=5):
    """Get embeddings for a batch of texts from ollama. Truncates and retries on failure.

    Returns a list aligned with ``texts``; each item is the embedding vector, or
    ``None`` for a transient failure (worth retrying later), or ``PERMANENT_FAIL``
    when the model can't embed that specific text.
    """
    # Truncate texts to avoid OOM on the server
    truncated = [t[:EMBED_MAX_CHARS] for t in texts]

    # Try batch first
    embeddings = _embed_request(config, truncated, retries)
    if (
        embeddings is not PERMANENT_FAIL
        and embeddings
        and len(embeddings) == len(texts)
    ):
        return embeddings
    if embeddings is not PERMANENT_FAIL and len(texts) == 1:
        # Con un solo texto, probarlo "uno a uno" repetiria la misma peticion.
        return [None]

    # Batch failed — fall back to one-by-one. A single poisoned text (NaN) makes
    # ollama 500 the whole batch, so isolating per-text salvages the rest.
    results = []
    for text in truncated:
        vec = _embed_request(config, text, retries)
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
    openai = config.embed_backend == "openai"
    if openai:
        url = f"{config.embed_base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        key = config.embed_api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
    else:
        url = f"{config.ollama_url}/api/embed"
        headers = {"Content-Type": "application/json"}

    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            if openai:
                # La API estandar devuelve [{"index": n, "embedding": [...]}, ...]
                # y no garantiza el orden; el resto del codigo espera la lista
                # alineada con los textos de entrada.
                filas = sorted(result.get("data", []), key=lambda d: d.get("index", 0))
                return [f["embedding"] for f in filas]
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
        print(
            f"  Warning: embedding failed after {retries + 1} attempts: {err}",
            file=sys.stderr,
        )
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
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    current_heading = ""
    current_text = ""
    heading_stack = []

    for part in parts:
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", part)
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
    paragraphs = re.split(r"\n\n+", text)
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


def embeds_truncated_at(conn):
    """Recorte con el que se hicieron los embeddings, si es menor que el actual.

    Hasta 0.2.0 el recorte era de 2000 caracteres y no se guardaba. Devuelve
    None si no hay embeddings o si ya se hicieron con EMBED_MAX_CHARS.
    """
    try:
        if not conn.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone():
            return None
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'embed_max_chars'"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    try:
        stored = int(row[0]) if row else 2000
    except ValueError:
        stored = 2000
    return stored if stored < EMBED_MAX_CHARS else None


def _set_last_index(conn):
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_index', ?)",
        (_iso_now(),),
    )


def _iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def warn_truncated_embeds(conn):
    old = embeds_truncated_at(conn)
    if old is not None:
        print(
            f"  Warning: embeddings were made with text cut at {old} characters "
            f"(now {EMBED_MAX_CHARS}). Run 'ownsearch index --full' to rebuild them.",
            file=sys.stderr,
        )


def bump_vectors_rev(conn):
    """Marca la cache de vectores como caducada.

    Llamalo siempre que cambie la tabla embeddings, aunque sea por borrado
    en cascada de chunks.
    """
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('vectors_rev', ?)",
        (str(time.time_ns()),),
    )


def _vectors_rev(conn):
    """Version de la tabla embeddings para validar la cache.

    Ademas de vectors_rev lleva el numero de filas y el chunk_id maximo:
    una version anterior de ownsearch (sin vectors_rev) que indexe sobre la
    misma base de datos tambien invalida la cache.
    """
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'vectors_rev'"
        ).fetchone()
        count, max_id = conn.execute(
            "SELECT COUNT(*), MAX(chunk_id) FROM embeddings"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    rev = row[0] if row and row[0] is not None else "0"
    return f"{rev}:{count}:{max_id}"


def _cache_matches_db(conn, mat, ids, samples=8):
    """Compara unas filas de la cache con la base de datos.

    vectors_rev y la huella COUNT/MAX no ven todo: una version anterior que
    reindexa entero reutiliza los mismos chunk_id, y dos procesos que
    reconstruyen a la vez pueden mezclar ficheros. Unas pocas lecturas por
    chunk_id lo detectan sin leer toda la tabla.
    """
    n = ids.shape[0]
    if n == 0:
        return True
    picks = {0, n - 1} | {(n * k) // samples for k in range(1, samples)}
    for i in picks:
        row = conn.execute(
            "SELECT vector FROM embeddings WHERE chunk_id = ?", (int(ids[i]),)
        ).fetchone()
        if row is None or len(row[0]) != mat.shape[1] * 4:
            return False
        vec = np.frombuffer(row[0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm and not np.allclose(vec / norm, mat[i], atol=1e-5):
            return False
    return True


def _load_cached_matrix(mat_path, ids_path):
    """Lee la cache; None si falta, esta corrupta o no cuadra."""
    try:
        mat = np.load(mat_path, mmap_mode="r")
        ids = np.load(ids_path, mmap_mode="r")
    except Exception:  # noqa: BLE001 - fichero truncado: EOFError, ValueError...
        return None
    if mat.ndim != 2 or ids.ndim != 1 or ids.dtype != np.int64:
        return None
    if mat.dtype != np.float32 or mat.shape[0] != ids.shape[0]:
        return None
    return mat, ids


def load_vector_matrix(db_path, conn, dim):
    """Matriz (n, dim) de vectores normalizados y sus chunk_ids, con numpy.

    Solo entran los vectores de la dimension `dim`, la de la consulta: los
    restos de otro modelo (mismo nombre, otro backend) no se comparan.

    Se guarda junto a la base de datos en dos .npy y se abre con mmap: leer
    y desempaquetar cien mil BLOBs en cada consulta cuesta mas que la propia
    busqueda. La cache se rehace cuando cambia vectors_rev.
    """
    rev = _vectors_rev(conn)
    base = str(db_path)
    mat_path, ids_path, rev_path = (
        base + ".vectors.npy",
        base + ".ids.npy",
        base + ".vectors.rev",
    )
    try:
        cached_rev = Path(rev_path).read_text()
    except OSError:
        cached_rev = None
    if rev is not None and cached_rev == rev:
        cached = _load_cached_matrix(mat_path, ids_path)
        if (
            cached is not None
            and cached[0].shape[1] == dim
            and _cache_matches_db(conn, *cached)
        ):
            return cached

    rows = conn.execute("SELECT chunk_id, vector FROM embeddings").fetchall()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64)
    size = dim * 4
    rows = [r for r in rows if len(r[1]) == size]
    if not rows:
        return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.int64)
    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
    mat = mat.reshape(len(rows), size // 4).copy()
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms

    if rev is not None:
        # Nombre temporal por proceso: dos busquedas a la vez pueden
        # reconstruir la cache al mismo tiempo.
        tmps = []
        try:
            for path, arr in ((mat_path, mat), (ids_path, ids)):
                tmp = f"{path}.{os.getpid()}.tmp"
                tmps.append(tmp)
                with open(tmp, "wb") as f:
                    np.save(f, arr)
                os.replace(tmp, path)
            Path(rev_path).write_text(rev)
        except OSError as e:
            print(
                f"  Warning: no se pudo guardar la cache de vectores: {e}",
                file=sys.stderr,
            )
            for tmp in tmps:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    return mat, ids


def top_chunks_numpy(db_path, conn, query_vec, limit, allowed=None):
    """[(sim, chunk_id)] de los `limit` vectores mas parecidos.

    `allowed`: conjunto de chunk_ids entre los que buscar (filtro --dir).
    """
    if limit < 1 or _numpy() is None:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    mat, ids = load_vector_matrix(db_path, conn, q.shape[0])
    if mat.shape[0] == 0 or mat.shape[1] != q.shape[0]:
        return []
    norm = np.linalg.norm(q)
    if norm == 0:
        return []
    sims = mat @ (q / norm)
    candidates = len(sims)
    if allowed is not None:
        mask = np.isin(ids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
        sims = np.where(mask, sims, -np.inf)
        candidates = int(mask.sum())
    k = min(limit, candidates)
    if k < 1:
        return []
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(float(sims[i]), int(ids[i])) for i in top]


def top_chunks_python(conn, query_vec, limit, allowed=None):
    """Lo mismo que top_chunks_numpy, sin dependencias. Lento con mucho indice."""
    if limit < 1:
        return []
    scored = []
    size = len(query_vec) * 4
    for chunk_id, vec_blob in conn.execute("SELECT chunk_id, vector FROM embeddings"):
        # Igual que la ruta numpy: los restos de otro modelo no se comparan.
        if allowed is not None and chunk_id not in allowed:
            continue
        if len(vec_blob) == size:
            scored.append((cosine_sim(query_vec, unpack_vector(vec_blob)), chunk_id))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --- Document conversion ---

_docvortex = None


def docvortex_available():
    global _docvortex
    if _docvortex is None:
        try:
            import docvortex
            from loguru import logger

            # docvortex saca trazas DEBUG por stderr en cada fichero.
            logger.disable("docvortex")
            _docvortex = docvortex
        except ImportError:
            _docvortex = False
    return bool(_docvortex)


def ocr_image(path, config):
    """Texto de una imagen con el motor de OCR configurado.

    Un fallo de red (OSError) no registra el fichero: se reintenta en el
    siguiente index.
    """
    if config is not None and config.ocr_engine == "vlm":
        text = _ocr_vlm(path, config)
        if text.strip() == "(sin texto)":
            text = ""
        return text, True
    return _ocr_local(path), False


def _ocr_local(path):
    """OCR con los modelos de docvortex, directamente sobre la imagen.

    docvortex no acepta imagenes como documento, asi que se usan sus piezas
    de OCR de PDF (deteccion de bloques y reconocimiento de lineas). Son
    funciones internas: si cambian en otra version, falla aqui con un error
    claro en vez de indexar mal.
    """
    if not docvortex_available():
        raise RuntimeError("local OCR needs ownsearch[docs]")
    try:
        from docvortex.analyzers.ocr.pdf import _analyze_page
        from docvortex.analyzers.ocr.runtime import get_runtime
        from PIL import Image, ImageOps
    except ImportError as e:
        raise RuntimeError(f"this docvortex version has no usable OCR: {e}") from e
    np_ = _numpy()
    if np_ is None:
        raise RuntimeError("local OCR needs numpy")
    layout, text_model = get_runtime()
    with Image.open(path) as im:
        rgb = np_.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    blocks, _ = _analyze_page(rgb, layout.predict(rgb), text_model, 0)
    parts = []
    for block in blocks:
        content = block.get("content")
        if isinstance(content, list):
            content = "".join(
                c.get("content", "") if isinstance(c, dict) else str(c) for c in content
            )
        if content:
            parts.append(str(content))
    return "\n\n".join(parts)


def _image_data_url(path):
    """Imagen como data URL. Con Pillow se reduce a 1600 px: una foto de
    movil ocupa varios MB y el modelo no necesita tanto."""
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1600, 1600))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
        data, mime = buf.getvalue(), "image/jpeg"
    except ImportError:
        data, mime = path.read_bytes(), IMAGE_MIME.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _ocr_vlm(path, config):
    """OCR con un modelo con vision por /v1/chat/completions."""
    base = config.data.get("ocr_base_url", "").rstrip("/")
    body = {
        "model": config.data.get("ocr_model"),
        "temperature": 0,
        "max_tokens": 4000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": _image_data_url(path)}},
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    key = config.ocr_api_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(), headers=headers
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"].get("content") or ""


def _docvortex_markdown(path, **options):
    result = _docvortex.parse(path, **options)
    md = _docvortex.render_artifact(
        result.middle_json, "markdown", assets=result.assets
    ).content
    if isinstance(md, bytes):
        md = md.decode("utf-8", errors="replace")
    return md


def _has_text(md):
    """True si el Markdown tiene texto real, no solo enlaces a imagenes."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    return len(re.sub(r"\W", "", text)) >= 20


def extract_text(path, config=None):
    """Devuelve (texto, es_markdown). Los documentos pasan por docvortex."""
    if path.suffix.lower() in IMAGE_EXTS:
        return ocr_image(path, config)
    if path.suffix.lower() not in DOCUMENT_EXTS or not docvortex_available():
        return path.read_text(encoding="utf-8", errors="replace"), path.suffix == ".md"
    md = _docvortex_markdown(path)
    if path.suffix.lower() == ".pdf" and not _has_text(md):
        # docvortex decide por PDF si hace falta OCR, y a veces se equivoca:
        # PDFs con fuentes incrustadas raras dan solo imagenes. Se reintenta
        # forzando OCR (solo los PDF admiten parse_mode).
        md = _docvortex_markdown(path, parse_mode="ocr")
    # docvortex pone los titulos en negrita (`## **Titulo**`); sin esto el
    # asterisco acaba en el heading de cada chunk.
    md = re.sub(r"^(#{1,6}\s+)\*\*(.+?)\*\*\s*$", r"\1\2", md, flags=re.MULTILINE)
    # Las imagenes no se extraen y el HTML en linea (<sup>, <strong>...) solo
    # mete ruido en el indice y en los snippets.
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    md = re.sub(r"</?[a-zA-Z][a-zA-Z0-9]*(\s[^<>]*)?/?>", "", md)
    return md, True


# --- File walking ---


def walk_directory(dir_path, extensions, skip_dirs):
    """Walk a directory and yield (absolute_path, relative_path, stat) for indexable files."""
    dir_path = dir_path.resolve()
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in extensions:
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
    images = getattr(args, "images", False)
    image_dirs = config.data.setdefault("image_dirs", [])
    if path_str in dirs and (not images or path_str in image_dirs):
        print(f"Directory already indexed: {path}")
        return

    if path_str not in dirs:
        dirs.append(path_str)
        print(f"Added: {path}")
    if images and path_str not in image_dirs:
        image_dirs.append(path_str)
        print(f"Images (OCR, engine '{config.ocr_engine}') enabled for: {path}")
        if not config.images_enabled():
            print(
                "  Warning: no OCR engine available. Install ownsearch[docs] for "
                "the local engine, or set ocr_engine vlm with ocr_base_url and "
                "ocr_model.",
                file=sys.stderr,
            )
    config.save()
    print("Run 'ownsearch index' to index it.")


def cmd_remove_dir(args, config):
    """Remove a directory from the search index."""
    path = Path(args.path).resolve()
    path_str = str(path)

    dirs = config.data["directories"]
    if path_str not in dirs:
        # Try matching by suffix
        matches = [
            d
            for d in dirs
            if d.endswith(args.path) or d.endswith(args.path.rstrip("/"))
        ]
        if matches:
            path_str = matches[0]
        else:
            print(f"Directory not found in config: {args.path}", file=sys.stderr)
            sys.exit(1)

    dirs.remove(path_str)
    if path_str in config.data.get("image_dirs", []):
        config.data["image_dirs"].remove(path_str)
    config.save()

    # Remove from DB
    conn = sqlite3.connect(str(config.db_path))
    init_db(conn)
    conn.execute(
        "DELETE FROM chunks WHERE file_path IN (SELECT path FROM files WHERE directory = ?)",
        (path_str,),
    )
    conn.execute("DELETE FROM files WHERE directory = ?", (path_str,))
    bump_vectors_rev(conn)
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
        images = " [images]" if str(d) in config.image_dirs else ""
        print(f"  {exists} {d}{images}")


def cmd_config_show(config):
    """Show current configuration."""
    print(json.dumps(config.data, indent=2, ensure_ascii=False))


def cmd_config_set(args, config):
    """Set a configuration value."""
    key = args.key
    value = args.value

    valid_keys = {
        "db_path",
        "ollama_url",
        "embed_model",
        "embed_workers",
        "chunk_overlap",
        "ocr_engine",
        "ocr_base_url",
        "ocr_model",
        "ocr_api_key_cmd",
    }
    if key not in valid_keys:
        print(
            f"Invalid key. Valid keys: {', '.join(sorted(valid_keys))}", file=sys.stderr
        )
        sys.exit(1)

    if key == "ocr_engine" and value not in ("local", "vlm"):
        print("ocr_engine must be 'local' or 'vlm'", file=sys.stderr)
        sys.exit(1)
    if key in ("embed_workers", "chunk_overlap"):
        minimum = 1 if key == "embed_workers" else 0
        try:
            value = int(value)
            if value < minimum:
                raise ValueError
        except ValueError:
            print(f"{key} must be an integer >= {minimum}", file=sys.stderr)
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
            bump_vectors_rev(conn)
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)",
                (value,),
            )
            conn.commit()
            conn.close()
            print(
                f"Embedding model changed ({old_value} -> {value}). All embeddings cleared."
            )
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
    stored_model = conn.execute(
        "SELECT value FROM meta WHERE key = 'embed_model'"
    ).fetchone()
    if stored_model and stored_model[0] != config.embed_model:
        print(
            f"Embedding model changed ({stored_model[0]} -> {config.embed_model}). Clearing old embeddings."
        )
        conn.execute("DELETE FROM embeddings")
        bump_vectors_rev(conn)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)",
            (config.embed_model,),
        )
        conn.commit()

    # Check embeddings availability (auto-pull if needed)
    has_embeddings = ensure_embeddings_ready(config)

    images_ok = bool(config.image_dirs) and config.images_enabled()
    if config.image_dirs and not images_ok:
        print(
            f"  Warning: OCR engine '{config.ocr_engine}' not available; images are skipped.",
            file=sys.stderr,
        )

    # Gather all current files across all directories
    current_files = {}  # absolute_path -> (dir_str, mtime_ns, size)
    for dir_path in config.directories:
        dp = Path(dir_path)
        if not dp.exists():
            print(
                f"  Warning: directory not found, skipping: {dir_path}", file=sys.stderr
            )
            continue
        dir_str = str(dp)
        extensions = config.extensions
        if dir_str in config.image_dirs and images_ok:
            extensions = extensions | IMAGE_EXTS
        for abs_path, _, mtime_ns, size in walk_directory(
            dp, extensions, config.skip_dirs
        ):
            if Path(abs_path).suffix.lower() in IMAGE_EXTS and size < IMAGE_MIN_BYTES:
                continue
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
        _set_last_index(conn)
        conn.commit()
        warn_truncated_embeds(conn)
        conn.close()
        return

    print(f"Indexing: {len(to_index)} files to process, {len(to_remove)} to remove")

    # Remove deleted/changed files
    for path in to_remove + to_index:
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (path,))
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
    conn.commit()
    # Sin embeddings que sobrevivan al borrado (indice nuevo, modelo cambiado
    # o todos los ficheros cambiados), todos los de este run usan el recorte
    # actual.
    fresh_embeds = not conn.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone()

    # Index new/changed files
    total_chunks = 0
    embed_queue = []
    failed_ids = []  # chunk_ids whose embedding failed during this run
    workers = args.workers or config.embed_workers
    embedder = _Embedder(config, conn, workers) if has_embeddings else None

    path = None
    try:
        for i, path in enumerate(to_index, 1):
            full_path = Path(path)
            dir_str, mtime_ns, size = current_files[path]
            try:
                text, is_markdown = extract_text(full_path, config)
            except (OSError, PermissionError) as e:
                print(f"  Skip {path}: {e}", file=sys.stderr)
                continue
            except Exception as e:  # noqa: BLE001
                # Documento corrupto o no soportado. Se registra sin chunks para
                # no reintentar la conversion hasta que cambie el fichero.
                print(f"  Skip {path}: conversion failed: {e}", file=sys.stderr)
                conn.execute(
                    "INSERT INTO files (path, directory, mtime_ns, size) VALUES (?, ?, ?, ?)",
                    (path, dir_str, mtime_ns, size),
                )
                continue

            conn.execute(
                "INSERT INTO files (path, directory, mtime_ns, size) VALUES (?, ?, ?, ?)",
                (path, dir_str, mtime_ns, size),
            )

            if is_markdown:
                chunks = chunk_markdown(text)
            else:
                chunks = chunk_plaintext(text)

            prev_content = ""
            for idx, (heading, content) in enumerate(chunks):
                cur = conn.execute(
                    "INSERT INTO chunks (file_path, chunk_index, heading, content) VALUES (?, ?, ?, ?)",
                    (path, idx, heading, content),
                )
                total_chunks += 1
                if has_embeddings and len(content.strip()) >= 50:
                    embed_text = embed_text_for(
                        heading, content, prev_content, config.chunk_overlap
                    )
                    embed_queue.append((cur.lastrowid, embed_text[:EMBED_MAX_CHARS]))
                prev_content = content

                if has_embeddings and len(embed_queue) >= BATCH_SIZE:
                    failed_ids += embedder.submit(embed_queue)
                    embed_queue = []

            if i % 20 == 0:
                print(f"  {i}/{len(to_index)} files...")
                conn.commit()

        if has_embeddings:
            if embed_queue:
                failed_ids += embedder.submit(embed_queue)
            failed_ids += embedder.finish()
    except BaseException:
        # Ctrl-C o fallo a mitad: los ficheros ya guardados con lotes aun en
        # vuelo quedan marcados para reintentar, o el incremental siguiente
        # los daria por completos sin embeddings.
        if embedder is not None:
            pending = [cid for cid, _ in embed_queue] + embedder.abort()
            _mark_for_retry(conn, failed_ids + pending)
        if path is not None:
            # El fichero en curso puede tener solo parte de sus chunks.
            conn.execute("UPDATE files SET mtime_ns = -1 WHERE path = ?", (path,))
        bump_vectors_rev(conn)
        conn.commit()
        conn.close()
        raise

    # Any file with a failed embedding is unmarked (mtime_ns = -1) so the next
    # incremental run re-indexes it. We keep its chunks for now, so the file
    # stays searchable (FTS + whatever embeddings did succeed) in the meantime.
    _mark_for_retry(conn, failed_ids)

    bump_vectors_rev(conn)

    # Store the model used for these embeddings
    if has_embeddings:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)",
            (config.embed_model,),
        )
        if args.full or fresh_embeds:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_max_chars', ?)",
                (str(EMBED_MAX_CHARS),),
            )
    _set_last_index(conn)
    conn.commit()
    warn_truncated_embeds(conn)
    print(f"Done. {len(to_index)} files, {total_chunks} chunks indexed.")
    if has_embeddings:
        embed_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        print(f"  Embeddings: {embed_count}")
    conn.close()


def embed_text_for(heading, content, prev_content="", overlap=0):
    """Texto que se embebe para un chunk.

    Lleva delante el titulo y, con `chunk_overlap`, el final del chunk
    anterior del mismo fichero: una idea partida en la frontera de dos
    chunks queda en el vector de los dos. Solo afecta al vector; el texto
    guardado y el indice de texto no se repiten.
    """
    if overlap and prev_content:
        content = f"...{prev_content[-overlap:]}\n\n{content}"
    return f"{heading}: {content}" if heading else content


def _mark_for_retry(conn, chunk_ids):
    """Pone mtime_ns = -1 a los ficheros de esos chunks para reindexarlos."""
    if not chunk_ids:
        return
    failed_files = set()
    for i in range(0, len(chunk_ids), 500):
        part = chunk_ids[i : i + 500]
        placeholders = ",".join("?" * len(part))
        failed_files |= {
            row[0]
            for row in conn.execute(
                f"SELECT DISTINCT file_path FROM chunks WHERE id IN ({placeholders})",
                part,
            )
        }
    for fp in failed_files:
        conn.execute("UPDATE files SET mtime_ns = -1 WHERE path = ?", (fp,))
    print(
        f"  Warning: {len(chunk_ids)} chunk(s) across {len(failed_files)} file(s) "
        f"failed to embed; those files will be re-indexed on the next run.",
        file=sys.stderr,
    )


class _Embedder:
    """Pide embeddings en varios hilos y los guarda desde el hilo principal.

    Los hilos solo hacen HTTP; sqlite3 no se comparte entre hilos. Como mucho
    hay 2 lotes por hilo en vuelo, para no llenar la memoria si el servidor
    va mas lento que la lectura de ficheros.
    """

    def __init__(self, config, conn, workers):
        if config.embed_backend == "openai":
            # Resolver la clave aqui, en el hilo principal, antes de los hilos.
            config.embed_api_key
        self.config = config
        self.conn = conn
        self.max_pending = workers * 2
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending = deque()

    def submit(self, queue):
        """Encola un lote. Devuelve los fallos de los lotes ya terminados."""
        ids = [q[0] for q in queue]
        texts = [q[1] for q in queue]
        future = self.pool.submit(get_embeddings_batch, self.config, texts)
        self.pending.append((ids, future))
        failed = []
        while len(self.pending) >= self.max_pending:
            failed += self._drain_one()
        return failed

    def finish(self):
        failed = []
        while self.pending:
            failed += self._drain_one()
        self.pool.shutdown()
        return failed

    def abort(self):
        """Cancela lo pendiente sin esperar. Devuelve los chunk_ids sin guardar."""
        ids = [cid for batch, _ in self.pending for cid in batch]
        self.pending.clear()
        self.pool.shutdown(wait=False, cancel_futures=True)
        return ids

    def _drain_one(self):
        # El lote sale de la cola cuando ya esta guardado: si llega un Ctrl-C
        # mientras se espera, abort() aun lo ve como pendiente.
        ids, future = self.pending[0]
        try:
            vectors = future.result()
        except Exception as e:  # noqa: BLE001
            # Un fallo inesperado en un hilo no debe tirar la indexacion: el
            # lote cuenta como fallo transitorio y se reintenta en otro run.
            print(f"  Warning: embedding batch failed: {e}", file=sys.stderr)
            self.pending.popleft()
            return list(ids)
        # Tambien se guarda antes de sacarlo: si el guardado falla o llega
        # un Ctrl-C, abort() lo marca para reintentar.
        failed = _store_embed_batch(self.conn, ids, vectors)
        self.pending.popleft()
        return failed


def _store_embed_batch(conn, ids, vectors):
    """Store the vectors of a batch of chunk ids.

    Returns the list of chunk_ids that failed *transiently* (worth retrying),
    so the caller can avoid marking their source file as fully indexed —
    otherwise an incremental re-index would never retry them (the file's
    mtime/size already match). Permanent failures (the model can't embed that
    specific text) are skipped silently: their chunks stay FTS-only and the
    file is left marked as indexed, so we don't loop on them every run.
    """
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

    dir_prefix = None
    if args.dir:
        # Con separador final: --dir ~/notes no debe incluir ~/notes-viejas.
        dir_prefix = str(Path(args.dir).resolve()).rstrip(os.sep) + os.sep

    degraded = False
    if args.semantic or args.both:
        sem_results = search_semantic(
            config, conn, query, args.limit, dir_prefix, args.max_chars
        )
        degraded = sem_results is None
        sem_results = sem_results or []
    if args.semantic:
        results = sem_results
    else:
        results = search_fts(
            conn, query, args.limit, args.strict, dir_prefix, args.max_chars
        )
        if args.both:
            results = merge_results(results, sem_results, args.limit)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        format_results(results, config)

    conn.close()
    if degraded:
        # La busqueda semantica pedida no se hizo: los resultados (si los
        # hay) son solo de texto. Codigo propio para que un agente lo sepa.
        sys.exit(EXIT_SEMANTIC_UNAVAILABLE)


FTS_OPERATORS = {"AND", "OR", "NOT", "NEAR"}


def _fts_terms(query):
    """Palabras de la consulta, cada una entre comillas para FTS5.

    Quita los operadores de FTS5 y las palabras con `-` delante: quien
    escribe `-cilium` quiere excluir cilium, no buscarlo en el plan B.
    """
    terms = []
    for token in query.split():
        if token in FTS_OPERATORS or token.startswith("-"):
            continue
        terms += re.findall(r"\w+", token)
    return ['"' + t.replace('"', '""') + '"' for t in terms]


def _has_fts_operators(query):
    return any(token in FTS_OPERATORS for token in query.split())


def _fts_rows(conn, match, limit, dir_prefix=None):
    if limit < 1:
        # En SQLite, LIMIT negativo es "sin limite".
        return []
    # El filtro de directorio va dentro de la consulta, antes del LIMIT: si
    # se aplica despues, los N mejores de otras carpetas lo dejan vacio.
    where_dir, params = "", [match]
    if dir_prefix:
        where_dir = "AND substr(c.file_path, 1, ?) = ?"
        params += [len(dir_prefix), dir_prefix]
    try:
        return conn.execute(
            f"""
            SELECT c.id, c.file_path, c.heading, c.content,
                   snippet(chunks_fts, 0, '>>>', '<<<', '...', 40) as snip,
                   bm25(chunks_fts, {BM25_WEIGHTS[0]}, {BM25_WEIGHTS[1]}) as score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ? {where_dir}
            ORDER BY score
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()
    except sqlite3.OperationalError:
        return None


def search_fts(conn, query, limit=10, strict=False, dir_prefix=None, max_chars=None):
    """Full-text search using FTS5 BM25.

    FTS5 exige todas las palabras. Si eso no da nada y no es `strict`, se
    repite con OR: con consultas largas basta con que falte una palabra
    para perder el documento, y BM25 ya pone primero los que tienen mas.
    """
    terms = _fts_terms(query)
    # Con operadores explicitos (AND, OR, NOT, NEAR) quien busca ya decide
    # la logica: no se cambia por un OR.
    strict = strict or _has_fts_operators(query)
    # La consulta tal cual admite sintaxis FTS5 (frases, prefijo*). Si no es
    # sintaxis valida o no da nada, se buscan las palabras sueltas con AND.
    rows = _fts_rows(conn, query, limit, dir_prefix)
    and_query = " ".join(terms)
    if rows is None:
        # Sintaxis invalida. Con operadores no se adivina: quitar un NOT
        # convertiria una exclusion en una inclusion.
        rows = (
            []
            if _has_fts_operators(query)
            else _fts_rows(conn, and_query, limit, dir_prefix)
        )
    elif not rows and not strict and and_query != query:
        rows = _fts_rows(conn, and_query, limit, dir_prefix)
    if not rows and not strict and len(terms) > 1:
        rows = _fts_rows(conn, " OR ".join(terms), limit, dir_prefix)

    results = []
    for chunk_id, path, heading, content, snippet_text, score in rows or []:
        if max_chars:
            snippet = clip_text(content, max_chars)
        else:
            snippet = snippet_text.replace(">>>", "").replace("<<<", "")
        results.append(
            _result(chunk_id, path, heading, snippet, round(-score, 4), "fts")
        )
    _add_mtimes(conn, results)
    return results


def clip_text(text, max_chars):
    """Corta en el ultimo espacio antes de max_chars, no a mitad de palabra."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return cut.rstrip() + "..."


def _result(chunk_id, path, heading, snippet, score, method):
    return {
        "path": path,
        "heading": heading or "",
        "snippet": snippet,
        "score": score,
        "method": method,
        "chunk_id": chunk_id,
    }


def _add_mtimes(conn, results):
    """Anade a cada resultado la fecha del fichero (ISO 8601, UTC).

    Un agente la usa para juzgar si el dato es reciente sin abrir el fichero.
    Sale del indice; si el fichero esta marcado para reindexar (-1), del disco.
    """
    cache = {}
    for r in results:
        path = r["path"]
        if path not in cache:
            row = conn.execute(
                "SELECT mtime_ns FROM files WHERE path = ?", (path,)
            ).fetchone()
            ns = row[0] if row else None
            if ns is None or ns < 0:
                try:
                    ns = os.stat(path).st_mtime_ns
                except OSError:
                    ns = None
            cache[path] = (
                datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat(
                    timespec="seconds"
                )
                if ns is not None
                else None
            )
        r["mtime"] = cache[path]


def search_semantic(config, conn, query, limit=10, dir_prefix=None, max_chars=None):
    """Semantic search using embeddings.

    Devuelve None si la busqueda no se pudo hacer (backend caido, modelo
    ausente) para distinguirlo de "sin resultados".
    """
    # Un reintento: quien busca (a menudo un agente) espera la respuesta, y
    # los 5 de la indexacion suman un minuto de esperas con el backend caido.
    vectors = get_embeddings_batch(config, [query], retries=SEARCH_EMBED_RETRIES)
    if not vectors or vectors[0] is PERMANENT_FAIL or not vectors[0]:
        print(
            "Semantic search unavailable (embedding backend not reachable, model missing, or query not embeddable).",
            file=sys.stderr,
        )
        return None

    query_vec = vectors[0]

    if not conn.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone():
        print("No embeddings in index. Re-run: ownsearch index", file=sys.stderr)
        return []

    allowed = None
    if dir_prefix:
        allowed = {
            row[0]
            for row in conn.execute(
                "SELECT id FROM chunks WHERE substr(file_path, 1, ?) = ?",
                (len(dir_prefix), dir_prefix),
            )
        }
        if not allowed:
            return []

    if _numpy() is not None:
        top = top_chunks_numpy(config.db_path, conn, query_vec, limit, allowed)
    else:
        top = top_chunks_python(conn, query_vec, limit, allowed)

    results = []
    for sim, chunk_id in top:
        row = conn.execute(
            "SELECT file_path, heading, content FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            continue
        path, heading, content = row
        if max_chars:
            snippet = clip_text(content, max_chars)
        else:
            snippet = clip_text(content.replace("\n", " "), 200)
        results.append(
            _result(chunk_id, path, heading, snippet, round(sim, 4), "semantic")
        )
    _add_mtimes(conn, results)
    return results


def merge_results(fts_results, sem_results, limit=10):
    """Merge results using reciprocal rank fusion.

    La clave es el chunk: los trozos de una seccion larga comparten fichero
    y titulo, y con (path, heading) se fundian en uno. `methods` dice que
    busquedas lo encontraron; si son las dos, suele ser lo mas relevante.
    """
    K = 60
    scores = {}
    all_results = {}
    methods = {}

    for method, results in (("fts", fts_results), ("semantic", sem_results)):
        for rank, r in enumerate(results):
            key = r["chunk_id"]
            scores[key] = scores.get(key, 0) + 1.0 / (K + rank + 1)
            methods.setdefault(key, []).append(method)
            if key not in all_results:
                all_results[key] = r

    merged = []
    for key, r in all_results.items():
        r = r.copy()
        r["score"] = round(scores.get(key, 0), 4)
        r["method"] = "combined"
        r["methods"] = methods[key]
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
        print(f"       {snippet}")
        print()


def status_data(config):
    """Estado en forma de dict, para `status --json`."""
    if config.embed_backend == "openai":
        url = config.embed_base_url
    else:
        url = config.ollama_url
    reachable = ollama_available(config)
    data = {
        "version": __version__,
        "db_path": str(config.db_path),
        "db_bytes": None,
        "embed_backend": config.embed_backend,
        "embed_url": url,
        "embed_reachable": reachable,
        "embed_model": config.embed_model,
        "embed_model_available": bool(reachable and ollama_has_model(config)),
        "directories": [],
        "chunks": 0,
        "embeddings": 0,
        "embed_max_chars": None,
        "last_index": None,
    }
    conn = None
    if config.db_path.exists():
        data["db_bytes"] = config.db_path.stat().st_size
        conn = sqlite3.connect(str(config.db_path))
    try:
        for d in config.data["directories"]:
            files = None
            if conn:
                files = conn.execute(
                    "SELECT COUNT(*) FROM files WHERE directory = ?", (d,)
                ).fetchone()[0]
            data["directories"].append(
                {"path": d, "exists": Path(d).exists(), "files": files}
            )
        if conn:
            data["chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            data["embeddings"] = conn.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0]
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            data["last_index"] = meta.get("last_index")
            if data["embeddings"]:
                # Sin la clave, los embeddings son de 0.2.0 o anterior.
                data["embed_max_chars"] = int(meta.get("embed_max_chars") or 2000)
    finally:
        if conn:
            conn.close()
    return data


def cmd_status(config, as_json=False):
    """Show status of ownsearch."""
    if as_json:
        print(json.dumps(status_data(config), ensure_ascii=False, indent=2))
        return
    print(f"ownsearch v{__version__}")
    print(f"Config: {config.config_file}")
    print(f"Database: {config.db_path}", end="")
    if config.db_path.exists():
        size_mb = config.db_path.stat().st_size / 1024 / 1024
        print(f" ({size_mb:.1f} MB)")
    else:
        print(" (not created)")

    destino = (
        config.embed_base_url if config.embed_backend == "openai" else config.ollama_url
    )
    print(f"\nEmbeddings ({config.embed_backend}): {destino}", end="")
    if ollama_available(config):
        print(" ✓", end="")
        if ollama_has_model(config):
            print(f" (model '{config.embed_model}' ready)")
        else:
            print(
                f" (model '{config.embed_model}' NOT found — will auto-pull on index)"
            )
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
                n = conn.execute(
                    "SELECT COUNT(*) FROM files WHERE directory = ?", (d,)
                ).fetchone()[0]
                count = f" [{n} files]"
            print(f"  {exists} {d}{count}")
        if conn:
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embeds = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            print(f"\n  Total: {total} chunks, {embeds} embeddings")
            warn_truncated_embeds(conn)
            conn.close()
    else:
        print("  (none — use 'ownsearch add-dir PATH')")


# --- Main ---


def _positive_int(value):
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value}")
    return n


def main():
    parser = argparse.ArgumentParser(
        prog="ownsearch",
        description="Smart full-text and semantic search for your local documents",
    )
    parser.add_argument(
        "--version", action="version", version=f"ownsearch {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    # add-dir
    add = sub.add_parser("add-dir", help="Add a directory to index")
    add.add_argument("path", help="Directory path to add")
    add.add_argument(
        "--images",
        action="store_true",
        help="Also index images in it with OCR (engine: ocr_engine)",
    )

    # remove-dir
    rm = sub.add_parser("remove-dir", help="Remove a directory from index")
    rm.add_argument("path", help="Directory path to remove")

    # list-dirs
    sub.add_parser("list-dirs", help="List indexed directories")

    # index
    idx = sub.add_parser("index", help="Index all configured directories")
    idx.add_argument("--full", action="store_true", help="Force full re-index")
    idx.add_argument(
        "--workers",
        type=_positive_int,
        help=f"Parallel embedding requests (default: embed_workers, {DEFAULT_EMBED_WORKERS})",
    )

    # search
    srch = sub.add_parser("search", help="Search the index")
    srch.add_argument("query", nargs="+", help="Search query")
    srch.add_argument("--semantic", action="store_true", help="Use semantic search")
    srch.add_argument("--both", action="store_true", help="Combined FTS + semantic")
    srch.add_argument(
        "--strict",
        action="store_true",
        help="Full-text: require every word (no OR fallback)",
    )
    srch.add_argument("--json", action="store_true", help="Output as JSON")
    srch.add_argument(
        "--limit", type=_positive_int, default=10, help="Max results (default: 10)"
    )
    srch.add_argument("--dir", help="Filter results to a specific directory")
    srch.add_argument(
        "--max-chars",
        type=_positive_int,
        help="Return up to N characters of each chunk instead of a short snippet",
    )

    # config
    cfg = sub.add_parser("config", help="Show or set configuration")
    cfg_sub = cfg.add_subparsers(dest="config_command")
    cfg_sub.add_parser("show", help="Show current config")
    cfg_set = cfg_sub.add_parser("set", help="Set a config value")
    cfg_set.add_argument("key", help="Config key (db_path, ollama_url, embed_model)")
    cfg_set.add_argument("value", help="Config value")

    # status
    st = sub.add_parser("status", help="Show ownsearch status")
    st.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()
    config = Config()

    if args.command == "add-dir":
        cmd_add_dir(args, config)
    elif args.command == "remove-dir":
        cmd_remove_dir(args, config)
    elif args.command == "list-dirs":
        cmd_list_dirs(config)
    elif args.command == "index":
        try:
            cmd_index(args, config)
        except KeyboardInterrupt:
            # cmd_index ya guardo lo hecho y marco lo pendiente. Salir sin
            # esperar a los hilos: una peticion de embeddings colgada puede
            # tardar minutos en soltar.
            print("\nInterrupted.", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(130)
    elif args.command == "search":
        cmd_search(args, config)
    elif args.command == "config":
        if hasattr(args, "config_command") and args.config_command == "set":
            cmd_config_set(args, config)
        else:
            cmd_config_show(config)
    elif args.command == "status":
        cmd_status(config, args.json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
