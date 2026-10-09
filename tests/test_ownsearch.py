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
import urllib.error
from pathlib import Path
from unittest import mock

import ownsearch

HAS_NUMPY = ownsearch._numpy() is not None
try:
    import PIL  # noqa: F401

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


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
        self.assertEqual(ownsearch.clip_text("alpha beta gamma", 13), "alpha beta...")
        self.assertEqual(ownsearch.clip_text("short", 12), "short")
        for n in range(1, 20):
            self.assertLessEqual(len(ownsearch.clip_text("hola cilium network", n)), n)


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


class TestDocuments(Base):
    def fake_docvortex(self, outputs):
        calls = []

        def parse(path, **options):
            calls.append(options)
            return mock.Mock(middle_json=outputs[len(calls) - 1], assets=None)

        def render(middle_json, fmt, assets=None):
            return mock.Mock(content=middle_json)

        fake = mock.Mock(parse=parse, render_artifact=render)
        return fake, calls

    def extract(self, name, outputs):
        path = self.docs / name
        path.write_bytes(b"%PDF-1.4")
        fake, calls = self.fake_docvortex(outputs)
        with (
            mock.patch.object(ownsearch, "_docvortex", fake),
            mock.patch.object(ownsearch, "docvortex_available", return_value=True),
        ):
            text, _ = ownsearch.extract_text(path)
        return text, calls

    def test_pdf_without_text_is_retried_with_ocr(self):
        text, calls = self.extract(
            "scan.pdf", ["![](images/p1.jpg)\n", "# Balance\n\nIngresos 1.200,00"]
        )
        self.assertEqual(calls, [{}, {"parse_mode": "ocr"}])
        self.assertIn("Ingresos", text)

    def test_pdf_with_text_is_not_retried(self):
        text, calls = self.extract("ok.pdf", ["# Title\n\nPlenty of real text here."])
        self.assertEqual(calls, [{}])


class TestImages(Base):
    def setUp(self):
        super().setUp()
        self.config.data["image_dirs"] = [str(self.docs.resolve())]
        self.config.data["directories"] = [str(self.docs.resolve())]

    def image(self, name, size=9000):
        (self.docs / name).write_bytes(b"\x89PNG" + b"\0" * size)

    def index_with_ocr(self, ocr):
        with (
            mock.patch.object(ownsearch, "ocr_image", side_effect=ocr),
            mock.patch.object(ownsearch.Config, "images_enabled", return_value=True),
            mock.patch("sys.stderr"),
        ):
            self.index(embed=False)

    def indexed(self):
        return sorted(
            Path(r[0]).name for r in self.conn().execute("SELECT path FROM files")
        )

    def test_images_only_in_enabled_dirs_and_not_tiny(self):
        other = self.root / "other"
        other.mkdir()
        (other / "o.png").write_bytes(b"\x89PNG" + b"\0" * 9000)
        self.config.data["directories"].append(str(other.resolve()))
        self.image("doc.png")
        self.image("icon.png", size=100)
        self.index_with_ocr(lambda path, config: ("Factura numero 42", False))
        self.assertEqual(self.indexed(), ["doc.png"])
        res = ownsearch.search_fts(self.conn(), "factura")
        self.assertEqual([Path(r["path"]).name for r in res], ["doc.png"])

    def test_network_failure_is_retried_next_run(self):
        self.image("doc.png")

        def fail(path, config):
            raise OSError("gateway down")

        self.index_with_ocr(fail)
        self.assertEqual(self.indexed(), [])
        self.index_with_ocr(lambda path, config: ("Ahora si", False))
        self.assertEqual(self.indexed(), ["doc.png"])

    def test_image_without_text_has_no_chunks(self):
        self.image("doc.png")
        self.index_with_ocr(lambda path, config: ("", True))
        self.assertEqual(self.indexed(), ["doc.png"])
        count = self.conn().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.assertEqual(count, 0)

    def test_vlm_no_text_marker_leaves_no_chunks(self):
        self.config.data["ocr_engine"] = "vlm"
        self.image("doc.png")
        with (
            mock.patch.object(ownsearch, "_ocr_vlm", return_value="(sin texto)"),
            mock.patch.object(ownsearch.Config, "images_enabled", return_value=True),
            mock.patch("sys.stderr"),
        ):
            self.index(embed=False)
        self.assertEqual(self.indexed(), ["doc.png"])
        count = self.conn().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.assertEqual(count, 0)

    def test_engine_problem_is_retried_next_run(self):
        self.image("doc.png")

        def fail(path, config):
            raise ownsearch.OcrUnavailable("docvortex changed")

        self.index_with_ocr(fail)
        self.assertEqual(self.indexed(), [])

    @unittest.skipIf(not HAS_PIL, "Pillow not installed")
    def test_corrupt_image_is_not_retried(self):
        # Pillow avisa con UnidentifiedImageError, que es un OSError.
        self.image("doc.png")
        self.config.data.update(
            ocr_engine="vlm", ocr_base_url="http://gw/v1", ocr_model="m"
        )
        with (
            mock.patch.object(ownsearch.Config, "images_enabled", return_value=True),
            mock.patch("urllib.request.urlopen") as urlopen,
            mock.patch("sys.stderr"),
        ):
            self.index(embed=False)
        urlopen.assert_not_called()
        self.assertEqual(self.indexed(), ["doc.png"])

    def test_without_pillow_only_small_web_images_are_sent(self):
        self.image("doc.png")
        (self.docs / "scan.tif").write_bytes(b"II*\0" + b"\0" * 9000)
        with mock.patch.dict("sys.modules", {"PIL": None}):
            url = ownsearch._image_data_url(self.docs / "doc.png")
            self.assertTrue(url.startswith("data:image/png;base64,"))
            with self.assertRaises(ownsearch.OcrUnavailable):
                ownsearch._image_data_url(self.docs / "scan.tif")

    def test_images_skipped_without_engine(self):
        self.image("doc.png")
        with (
            mock.patch.object(ownsearch.Config, "images_enabled", return_value=False),
            mock.patch("sys.stderr"),
        ):
            self.index(embed=False)
        self.assertEqual(self.indexed(), [])

    def test_vlm_request(self):
        self.config.data.update(
            ocr_engine="vlm",
            ocr_base_url="http://gw/v1/",
            ocr_model="qwen-vl",
        )
        self.config._ocr_api_key = "k"
        sent = {}

        def urlopen(req, timeout):
            sent["url"] = req.full_url
            sent["auth"] = req.get_header("Authorization")
            sent["body"] = json.loads(req.data)
            resp = mock.MagicMock()
            resp.__enter__.return_value.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "(sin texto)"}}]}
            ).encode()
            return resp

        self.image("doc.png")
        with (
            mock.patch.object(ownsearch, "_image_data_url", return_value="data:x"),
            mock.patch("urllib.request.urlopen", side_effect=urlopen),
        ):
            text, is_md = ownsearch.ocr_image(self.docs / "doc.png", self.config)
        self.assertEqual(text, "")
        self.assertEqual(sent["url"], "http://gw/v1/chat/completions")
        self.assertEqual(sent["auth"], "Bearer k")
        self.assertEqual(sent["body"]["model"], "qwen-vl")
        content = sent["body"]["messages"][0]["content"]
        self.assertEqual(content[1]["image_url"]["url"], "data:x")

    def test_vlm_rejected_image_is_not_retried(self):
        self.config.data.update(
            ocr_engine="vlm", ocr_base_url="http://gw/v1", ocr_model="m"
        )
        self.config._ocr_api_key = ""
        self.image("doc.png")
        for code, expected in ((400, ValueError), (503, OSError)):
            err = urllib.error.HTTPError("http://gw", code, "x", {}, None)
            with (
                mock.patch.object(ownsearch, "_image_data_url", return_value="data:x"),
                mock.patch("urllib.request.urlopen", side_effect=err),
                self.assertRaises(expected),
            ):
                ownsearch.ocr_image(self.docs / "doc.png", self.config)

    def test_vlm_bad_response_is_retried(self):
        self.config.data.update(
            ocr_engine="vlm", ocr_base_url="http://gw/v1", ocr_model="m"
        )
        self.config._ocr_api_key = ""
        self.image("doc.png")
        for body in (
            b"<html>oops</html>",
            b'{"error": "x"}',
            b'{"choices": []}',
            b'{"choices": [{"message": {"content": [{"type": "text"}]}}]}',
        ):
            resp = mock.MagicMock()
            resp.__enter__.return_value.read.return_value = body
            with (
                mock.patch.object(ownsearch, "_image_data_url", return_value="data:x"),
                mock.patch("urllib.request.urlopen", return_value=resp),
                self.assertRaises(ownsearch.OcrUnavailable),
            ):
                ownsearch.ocr_image(self.docs / "doc.png", self.config)

    def test_vlm_cut_response_is_retried(self):
        import http.client

        self.config.data.update(
            ocr_engine="vlm", ocr_base_url="http://gw/v1", ocr_model="m"
        )
        self.config._ocr_api_key = ""
        self.image("doc.png")
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.side_effect = http.client.IncompleteRead(b"")
        with (
            mock.patch.object(ownsearch, "_image_data_url", return_value="data:x"),
            mock.patch("urllib.request.urlopen", return_value=resp),
            self.assertRaises(ownsearch.OcrUnavailable),
        ):
            ownsearch.ocr_image(self.docs / "doc.png", self.config)

    def test_add_and_remove_image_dir(self):
        other = self.root / "pics"
        other.mkdir()
        self.config.save = lambda: None
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            ownsearch.cmd_add_dir(
                argparse.Namespace(path=str(other), images=True), self.config
            )
        self.assertIn(str(other.resolve()), self.config.image_dirs)
        with mock.patch("sys.stdout"):
            ownsearch.cmd_remove_dir(argparse.Namespace(path=str(other)), self.config)
        self.assertNotIn(str(other.resolve()), self.config.image_dirs)


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

    def test_unembeddable_query_is_asked_once(self):
        calls = []

        def request(config, data, retries=5):
            calls.append(1)
            return ownsearch.PERMANENT_FAIL

        with (
            mock.patch.object(ownsearch, "_embed_request", side_effect=request),
            mock.patch("sys.stderr"),
        ):
            self.assertEqual(
                ownsearch.search_semantic(self.config, self.conn(), "q"), []
            )
        self.assertEqual(len(calls), 1)

    def test_status_json_with_empty_db_file(self):
        Path(self.config.data["db_path"] + ".empty").write_bytes(b"")
        self.config.data["db_path"] += ".empty"
        with (
            mock.patch.object(ownsearch, "ollama_available", return_value=False),
        ):
            data = ownsearch.status_data(self.config)
        self.assertEqual(data["chunks"], 0)

    def test_status_json_with_non_sqlite_file(self):
        Path(self.config.data["db_path"] + ".junk").write_bytes(b"not a database" * 100)
        self.config.data["db_path"] += ".junk"
        with mock.patch.object(ownsearch, "ollama_available", return_value=False):
            self.assertEqual(ownsearch.status_data(self.config)["chunks"], 0)

    def test_unembeddable_query_is_empty_not_down(self):
        with (
            mock.patch.object(
                ownsearch,
                "get_embeddings_batch",
                return_value=[ownsearch.PERMANENT_FAIL],
            ),
            mock.patch("sys.stderr"),
        ):
            self.assertEqual(
                ownsearch.search_semantic(self.config, self.conn(), "q"), []
            )

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
