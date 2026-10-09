import argparse
import hashlib
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import ownsearch


def fake_vec(text, dim=8):
    """Vector fijo por texto, para no depender de un servidor de embeddings."""
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:dim]]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        self.config = ownsearch.Config.__new__(ownsearch.Config)
        self.config.config_dir = self.root / "cfg"
        self.config.config_file = self.config.config_dir / "config.json"
        self.config.data = ownsearch.Config._defaults(self.config)
        self.config.data["db_path"] = str(self.root / "t.db")
        self.config.data["directories"] = [str(self.docs)]

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        (self.docs / name).write_text(text)

    def index(self, embed=True, workers=None, embed_fn=None):
        embed_fn = embed_fn or (lambda config, texts: [fake_vec(t) for t in texts])
        with (
            mock.patch.object(ownsearch, "ensure_embeddings_ready", return_value=embed),
            mock.patch.object(ownsearch, "get_embeddings_batch", side_effect=embed_fn),
            mock.patch("sys.stdout"),
        ):
            ownsearch.cmd_index(
                argparse.Namespace(full=False, workers=workers), self.config
            )

    def conn(self):
        conn = sqlite3.connect(self.config.data["db_path"])
        self.addCleanup(conn.close)
        return conn


class TestFts(Base):
    def setUp(self):
        super().setUp()
        self.write("rf.md", "# Radio\n\nNotes on RF jamming and how to spot it.\n")
        self.write("k8s.md", "# Cluster\n\nKubernetes cilium network policy.\n")
        self.index(embed=False)

    def paths(self, results):
        return [Path(r["path"]).name for r in results]

    def test_all_words_match(self):
        res = ownsearch.search_fts(self.conn(), "kubernetes cilium")
        self.assertEqual(self.paths(res), ["k8s.md"])

    def test_or_fallback_when_and_finds_nothing(self):
        res = ownsearch.search_fts(self.conn(), "RF jamming detection")
        self.assertEqual(self.paths(res), ["rf.md"])

    def test_strict_keeps_and(self):
        res = ownsearch.search_fts(self.conn(), "RF jamming detection", strict=True)
        self.assertEqual(res, [])

    def test_invalid_fts_syntax_does_not_fail(self):
        res = ownsearch.search_fts(self.conn(), 'cilium "network')
        self.assertEqual(self.paths(res), ["k8s.md"])

    def test_heading_weighs_more(self):
        self.write("a.md", "# Notes\n\nSome text that mentions cilium once.\n")
        self.write("b.md", "# Cilium\n\nSome text that mentions nothing else.\n")
        self.index(embed=False)
        res = ownsearch.search_fts(self.conn(), "cilium")
        self.assertEqual(self.paths(res)[0], "b.md")


class TestIndex(Base):
    def test_long_chunk_is_embedded_whole(self):
        seen = []

        def embed(config, texts):
            seen.extend(texts)
            return [fake_vec(t) for t in texts]

        self.write("long.txt", "x" * 3999)
        self.index(embed_fn=embed)
        self.assertEqual(len(seen[0]), 3999)
        self.assertLess(ownsearch.MAX_CHUNK_CHARS, ownsearch.EMBED_MAX_CHARS)

    def test_parallel_index_stores_every_vector(self):
        threads = set()

        def embed(config, texts):
            threads.add(threading.get_ident())
            return [fake_vec(t) for t in texts]

        for i in range(40):
            self.write(f"n{i}.md", f"# Note {i}\n\n" + f"word{i} " * 20)
        self.index(workers=4, embed_fn=embed)
        conn = self.conn()
        rows = conn.execute(
            "SELECT c.heading, c.content, e.vector FROM chunks c "
            "JOIN embeddings e ON e.chunk_id = c.id"
        ).fetchall()
        self.assertEqual(len(rows), 40)
        for heading, content, blob in rows:
            expected = fake_vec(f"{heading}: {content}")
            got = ownsearch.unpack_vector(blob)
            for a, b in zip(expected, got):
                self.assertAlmostEqual(a, b, places=5)

    def test_transient_failure_marks_file_for_retry(self):
        self.write("a.md", "# A\n\n" + "alpha " * 20)
        self.index(embed_fn=lambda config, texts: [None for _ in texts])
        mtime = self.conn().execute("SELECT mtime_ns FROM files").fetchone()[0]
        self.assertEqual(mtime, -1)


class TestTruncationWarning(Base):
    def setUp(self):
        super().setUp()
        self.write("a.md", "# A\n\n" + "alpha " * 20)

    def test_new_index_has_no_warning(self):
        self.index()
        self.assertIsNone(ownsearch.embeds_truncated_at(self.conn()))

    def test_index_from_old_version_warns_until_full(self):
        self.index()
        conn = self.conn()
        # Una base de datos de 0.2.0 no tiene embed_max_chars.
        conn.execute("DELETE FROM meta WHERE key = 'embed_max_chars'")
        conn.commit()
        self.assertEqual(ownsearch.embeds_truncated_at(conn), 2000)

        self.write("b.md", "# B\n\n" + "beta " * 20)
        self.index()
        self.assertEqual(ownsearch.embeds_truncated_at(conn), 2000)

        with (
            mock.patch.object(ownsearch, "ensure_embeddings_ready", return_value=True),
            mock.patch.object(
                ownsearch,
                "get_embeddings_batch",
                side_effect=lambda c, t: [fake_vec(x) for x in t],
            ),
            mock.patch("sys.stdout"),
        ):
            ownsearch.cmd_index(
                argparse.Namespace(full=True, workers=None), self.config
            )
        self.assertIsNone(ownsearch.embeds_truncated_at(conn))


class TestSemantic(Base):
    def setUp(self):
        super().setUp()
        for i in range(30):
            self.write(f"n{i}.md", f"# Note {i}\n\n" + f"topic{i} " * 20)
        self.index()

    def search(self):
        query = fake_vec("query")
        with (
            mock.patch.object(ownsearch, "get_embeddings_batch", return_value=[query]),
            mock.patch("sys.stderr"),
        ):
            return ownsearch.search_semantic(self.config, self.conn(), "q", limit=5)

    def test_python_and_numpy_agree(self):
        with mock.patch.object(ownsearch, "np", None):
            pure = self.search()
        self.assertEqual(len(pure), 5)
        if ownsearch.np is None:
            self.skipTest("numpy not installed")
        fast = self.search()
        self.assertEqual([r["path"] for r in fast], [r["path"] for r in pure])
        for a, b in zip(fast, pure):
            self.assertAlmostEqual(a["score"], b["score"], places=3)

    @unittest.skipIf(ownsearch.np is None, "numpy not installed")
    def test_cache_is_rebuilt_after_index_change(self):
        self.search()
        cache = Path(self.config.data["db_path"] + ".vectors.npy")
        self.assertTrue(cache.exists())
        for i in range(30):
            (self.docs / f"n{i}.md").unlink()
        self.write("only.md", "# Only\n\n" + "single " * 20)
        self.index()
        res = self.search()
        self.assertEqual([Path(r["path"]).name for r in res], ["only.md"])


if __name__ == "__main__":
    unittest.main()
