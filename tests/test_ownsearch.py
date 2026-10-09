import argparse
import io
import json
import os
import hashlib
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import ownsearch

HAS_NUMPY = ownsearch._numpy() is not None


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
        # Mismas longitudes de columna: con pesos 1,1 empatan.
        self.write("a.md", "# Notes\n\ncilium\n")
        self.write("b.md", "# Cilium\n\nnotes\n")
        # Relleno sin la palabra, para que su IDF no sea casi 0.
        for i in range(8):
            self.write(f"f{i}.md", f"# Filler {i}\n\nnothing here\n")
        self.index(embed=False)
        res = ownsearch.search_fts(self.conn(), "cilium")
        self.assertEqual(self.paths(res)[:2], ["b.md", "a.md"])
        self.assertGreater(res[0]["score"], res[1]["score"])

    def test_excluded_word_is_not_searched_in_fallback(self):
        res = ownsearch.search_fts(self.conn(), "radio -cilium")
        self.assertNotIn("k8s.md", self.paths(res))

    def test_explicit_operators_disable_fallback(self):
        res = ownsearch.search_fts(self.conn(), "kubernetes AND jamming")
        self.assertEqual(res, [])

    def test_invalid_syntax_with_not_returns_nothing(self):
        res = ownsearch.search_fts(self.conn(), 'radio NOT jamming "')
        self.assertEqual(res, [])

    def test_negative_limit_returns_nothing(self):
        self.assertEqual(ownsearch.search_fts(self.conn(), "cilium", limit=-1), [])

    def test_dir_filter_applies_before_limit(self):
        other = self.root / "other"
        other.mkdir()
        for i in range(5):
            (other / f"o{i}.md").write_text("# Cilium\n\ncilium cilium cilium\n")
        self.config.data["directories"].append(str(other))
        self.index(embed=False)
        prefix = str(self.docs.resolve()) + os.sep
        res = ownsearch.search_fts(self.conn(), "cilium", limit=2, dir_prefix=prefix)
        self.assertEqual(self.paths(res), ["k8s.md"])

    def test_dir_prefix_does_not_match_sibling_folder(self):
        sibling = self.root / "docs-old"
        sibling.mkdir()
        (sibling / "s.md").write_text("# Cilium\n\ncilium\n")
        self.config.data["directories"].append(str(sibling))
        self.index(embed=False)
        prefix = str(self.docs.resolve()) + os.sep
        res = ownsearch.search_fts(self.conn(), "cilium", dir_prefix=prefix)
        self.assertNotIn("s.md", self.paths(res))

    def test_max_chars_returns_chunk_text(self):
        res = ownsearch.search_fts(self.conn(), "cilium", max_chars=1000)
        self.assertIn("Kubernetes cilium network policy.", res[0]["snippet"])

    def test_results_have_mtime_and_chunk_id(self):
        res = ownsearch.search_fts(self.conn(), "cilium")
        self.assertRegex(res[0]["mtime"], r"^\d{4}-\d\d-\d\dT")
        self.assertIsInstance(res[0]["chunk_id"], int)

    def test_clip_text_cuts_at_word(self):
        self.assertEqual(ownsearch.clip_text("alpha beta gamma", 12), "alpha beta...")
        self.assertEqual(ownsearch.clip_text("short", 12), "short")


class TestMerge(unittest.TestCase):
    def r(self, chunk_id, method):
        return ownsearch._result(chunk_id, "same.md", "Same heading", "x", 1.0, method)

    def test_chunks_of_same_section_stay_separate(self):
        merged = ownsearch.merge_results(
            [self.r(1, "fts"), self.r(2, "fts")], [self.r(2, "semantic")]
        )
        self.assertEqual([m["chunk_id"] for m in merged], [2, 1])
        self.assertEqual(merged[0]["methods"], ["fts", "semantic"])
        self.assertEqual(merged[1]["methods"], ["fts"])


class TestIndex(Base):
    def test_long_chunk_is_embedded_whole(self):
        seen = []

        def embed(config, texts):
            seen.extend(texts)
            return [fake_vec(t) for t in texts]

        # Un solo parrafo no se parte: el fragmento pasa de MAX_CHUNK_CHARS.
        self.write("long.txt", "x" * 9000)
        self.index(embed_fn=embed)
        self.assertEqual(len(seen[0]), ownsearch.EMBED_MAX_CHARS)

    def test_embed_request_gets_text_cut_at_max(self):
        sent = []

        def request(config, texts, retries=5):
            sent.extend(texts)
            return [fake_vec(t) for t in texts]

        with mock.patch.object(ownsearch, "_embed_request", side_effect=request):
            ownsearch.get_embeddings_batch(self.config, ["y" * 9000, "short"])
        self.assertEqual([len(t) for t in sent], [ownsearch.EMBED_MAX_CHARS, 5])

    def test_parallel_index_stores_every_vector(self):
        threads = set()

        def embed(config, texts):
            threads.add(threading.get_ident())
            time.sleep(0.01)
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
        self.assertGreater(len(threads), 1)
        for heading, content, blob in rows:
            expected = fake_vec(f"{heading}: {content}")
            got = ownsearch.unpack_vector(blob)
            for a, b in zip(expected, got):
                self.assertAlmostEqual(a, b, places=5)

    def test_worker_exception_is_a_transient_failure(self):
        def embed(config, texts):
            raise RuntimeError("boom")

        self.write("a.md", "# A\n\n" + "alpha " * 20)
        with mock.patch("sys.stderr"):
            self.index(embed_fn=embed)
        mtime = self.conn().execute("SELECT mtime_ns FROM files").fetchone()[0]
        self.assertEqual(mtime, -1)

    def test_interrupt_marks_unfinished_files_for_retry(self):
        for i in range(12):
            self.write(f"n{i}.md", f"# Note {i}\n\n" + f"word{i} " * 20)
        real_submit = ownsearch._Embedder.submit
        calls = []

        def submit(embedder, queue):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real_submit(embedder, queue)

        with (
            mock.patch.object(ownsearch._Embedder, "submit", submit),
            mock.patch("sys.stderr"),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.index()
        conn = self.conn()
        rows = conn.execute(
            "SELECT f.path, f.mtime_ns, COUNT(e.chunk_id) FROM files f "
            "LEFT JOIN chunks c ON c.file_path = f.path "
            "LEFT JOIN embeddings e ON e.chunk_id = c.id GROUP BY f.path"
        ).fetchall()
        self.assertTrue(rows)
        for path, mtime, embeds in rows:
            self.assertTrue(mtime == -1 or embeds > 0, path)

    def test_interrupt_while_waiting_for_a_batch(self):
        for i in range(12):
            self.write(f"n{i}.md", f"# Note {i}\n\n" + f"word{i} " * 20)

        def embed(config, texts):
            raise KeyboardInterrupt

        with mock.patch("sys.stderr"), self.assertRaises(KeyboardInterrupt):
            self.index(embed_fn=embed)
        conn = self.conn()
        unmarked = conn.execute(
            "SELECT COUNT(*) FROM files WHERE mtime_ns != -1"
        ).fetchone()[0]
        self.assertEqual(unmarked, 0)

    def test_interrupt_while_storing_a_batch(self):
        for i in range(12):
            self.write(f"n{i}.md", f"# Note {i}\n\n" + f"word{i} " * 20)

        with (
            mock.patch.object(
                ownsearch, "_store_embed_batch", side_effect=KeyboardInterrupt
            ),
            mock.patch("sys.stderr"),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.index()
        unmarked = (
            self.conn()
            .execute("SELECT COUNT(*) FROM files WHERE mtime_ns != -1")
            .fetchone()[0]
        )
        self.assertEqual(unmarked, 0)

    def test_chunk_overlap_only_changes_embedded_text(self):
        seen = []

        def embed(config, texts):
            seen.extend(texts)
            return [fake_vec(t) for t in texts]

        self.config.data["chunk_overlap"] = 30
        para_a = "alpha " * 500
        para_b = "omega " * 500
        self.write("two.txt", para_a + "\n\n" + para_b)
        self.index(embed_fn=embed)
        self.assertEqual(len(seen), 2)
        self.assertTrue(seen[0].startswith("alpha"))
        self.assertIn("alpha", seen[1][:40])
        stored = [r[0] for r in self.conn().execute("SELECT content FROM chunks")]
        self.assertTrue(stored[1].startswith("omega"))

    def test_workers_must_be_positive(self):
        with self.assertRaises(Exception):
            ownsearch._positive_int("0")
        self.assertEqual(ownsearch._positive_int("3"), 3)

    def test_transient_failure_marks_file_for_retry(self):
        self.write("a.md", "# A\n\n" + "alpha " * 20)
        self.index(embed_fn=lambda config, texts: [None for _ in texts])
        mtime = self.conn().execute("SELECT mtime_ns FROM files").fetchone()[0]
        self.assertEqual(mtime, -1)


class TestTruncationWarning(Base):
    def setUp(self):
        super().setUp()
        self.write("a.md", "# A\n\n" + "alpha " * 20)

    def test_warning_clears_when_every_file_changes(self):
        self.index()
        conn = self.conn()
        conn.execute("DELETE FROM meta WHERE key = 'embed_max_chars'")
        conn.commit()
        self.write("a.md", "# A\n\n" + "changed " * 20)
        self.index()
        self.assertIsNone(ownsearch.embeds_truncated_at(conn))

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

    def test_python_search(self):
        with mock.patch.object(ownsearch, "np", None):
            self.assertEqual(len(self.search()), 5)

    @unittest.skipIf(not HAS_NUMPY, "numpy not installed")
    def test_python_and_numpy_agree(self):
        with mock.patch.object(ownsearch, "np", None):
            pure = self.search()
        fast = self.search()
        self.assertEqual([r["path"] for r in fast], [r["path"] for r in pure])
        for a, b in zip(fast, pure):
            self.assertAlmostEqual(a["score"], b["score"], places=3)

    def test_negative_limit_returns_nothing(self):
        if HAS_NUMPY:
            self.assertEqual(
                ownsearch.top_chunks_numpy(
                    self.config.db_path, self.conn(), fake_vec("q"), -5
                ),
                [],
            )
        self.assertEqual(
            ownsearch.top_chunks_python(self.conn(), fake_vec("q"), -5), []
        )
        with self.assertRaises(Exception):
            ownsearch._positive_int("-30")

    @unittest.skipIf(not HAS_NUMPY, "numpy not installed")
    def test_cache_detects_rewritten_vectors_with_same_ids(self):
        # Una version anterior que reindexa entero deja los mismos chunk_id.
        self.search()
        conn = self.conn()
        for (cid,) in conn.execute("SELECT chunk_id FROM embeddings").fetchall():
            vec = fake_vec(f"other {cid}")
            conn.execute(
                "UPDATE embeddings SET vector = ? WHERE chunk_id = ?",
                (ownsearch.pack_vector(vec), cid),
            )
        conn.commit()
        with mock.patch.object(ownsearch, "np", None):
            pure = self.search()
        self.assertEqual([r["path"] for r in self.search()], [r["path"] for r in pure])

    @unittest.skipIf(not HAS_NUMPY, "numpy not installed")
    def test_mixed_dimensions_use_the_query_dimension(self):
        # Mismo nombre de modelo en otro backend con otra dimension: la
        # mayoria de vectores viejos no debe tapar los del modelo actual.
        conn = self.conn()
        ids = [r[0] for r in conn.execute("SELECT chunk_id FROM embeddings")]
        for cid in ids[:20]:
            conn.execute(
                "UPDATE embeddings SET vector = ? WHERE chunk_id = ?",
                (ownsearch.pack_vector([0.5] * 4), cid),
            )
        ownsearch.bump_vectors_rev(conn)
        conn.commit()
        with mock.patch.object(ownsearch, "np", None):
            pure = self.search()
        fast = self.search()
        self.assertEqual(len(fast), 5)
        self.assertEqual([r["path"] for r in fast], [r["path"] for r in pure])

    def test_semantic_dir_filter(self):
        other = self.root / "other"
        other.mkdir()
        (other / "x.md").write_text("# X\n\n" + "extra " * 20)
        self.config.data["directories"].append(str(other))
        self.index()
        prefix = str(other.resolve()) + os.sep
        for numpy_off in (True, False):
            if not numpy_off and not HAS_NUMPY:
                continue
            with (
                mock.patch.object(ownsearch, "np", None if numpy_off else ownsearch.np),
                mock.patch.object(
                    ownsearch, "get_embeddings_batch", return_value=[fake_vec("q")]
                ),
            ):
                res = ownsearch.search_semantic(
                    self.config, self.conn(), "q", limit=5, dir_prefix=prefix
                )
            self.assertEqual([Path(r["path"]).name for r in res], ["x.md"])

    def test_backend_down_is_none_not_empty(self):
        with (
            mock.patch.object(ownsearch, "get_embeddings_batch", return_value=[]),
            mock.patch("sys.stderr"),
        ):
            self.assertIsNone(ownsearch.search_semantic(self.config, self.conn(), "q"))

    def test_query_embedding_fails_fast(self):
        calls = []

        def request(config, data, retries=5):
            calls.append(retries)
            return []

        with (
            mock.patch.object(ownsearch, "_embed_request", side_effect=request),
            mock.patch("sys.stderr"),
        ):
            self.assertIsNone(ownsearch.search_semantic(self.config, self.conn(), "q"))
        self.assertEqual(calls, [ownsearch.SEARCH_EMBED_RETRIES])

    def test_search_exits_3_when_semantic_is_unavailable(self):
        args = argparse.Namespace(
            query=["topic1"],
            semantic=False,
            both=True,
            strict=False,
            json=True,
            limit=5,
            dir=None,
            max_chars=None,
        )
        out = io.StringIO()
        with (
            mock.patch.object(ownsearch, "get_embeddings_batch", return_value=[]),
            mock.patch("sys.stderr"),
            mock.patch("sys.stdout", out),
            self.assertRaises(SystemExit) as exit_,
        ):
            ownsearch.cmd_search(args, self.config)
        self.assertEqual(exit_.exception.code, ownsearch.EXIT_SEMANTIC_UNAVAILABLE)
        self.assertTrue(json.loads(out.getvalue()))

    @unittest.skipIf(not HAS_NUMPY, "numpy not installed")
    def test_corrupt_cache_is_rebuilt(self):
        first = self.search()
        Path(self.config.data["db_path"] + ".ids.npy").write_bytes(b"")
        self.assertEqual(self.search(), first)

    @unittest.skipIf(not HAS_NUMPY, "numpy not installed")
    def test_cache_ignores_other_writer(self):
        # Una version anterior de ownsearch cambia embeddings sin vectors_rev.
        first = self.search()
        conn = self.conn()
        top = first[0]["path"]
        conn.execute(
            "DELETE FROM embeddings WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE file_path = ?)",
            (top,),
        )
        conn.commit()
        self.assertNotIn(top, [r["path"] for r in self.search()])

    @unittest.skipIf(not HAS_NUMPY, "numpy not installed")
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
