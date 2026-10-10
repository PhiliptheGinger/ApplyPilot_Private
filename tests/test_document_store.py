"""FW50 (2026-10-09): per-job document storage + exact-hash dedup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import document_store


@pytest.fixture(autouse=True)
def _isolated_app_dir(monkeypatch, tmp_path):
    """Never touch the real ~/.applypilot/documents/ from tests."""
    from applypilot import config

    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    return tmp_path


class TestStoreDocument:
    def test_first_store_creates_canonical_file(self, tmp_path):
        result = document_store.store_document("https://a.com/job/1", "notice.pdf", b"hello world")

        assert result["is_new_document"] is True
        canonical = Path(result["canonical_path"])
        assert canonical.exists()
        assert canonical.read_bytes() == b"hello world"

    def test_identical_content_from_different_job_dedups(self):
        r1 = document_store.store_document("https://a.com/job/1", "notice.pdf", b"same bytes")
        r2 = document_store.store_document("https://b.com/job/2", "state-notice.pdf", b"same bytes")

        assert r1["is_new_document"] is True
        assert r2["is_new_document"] is False
        assert r1["sha256"] == r2["sha256"]
        assert r1["canonical_path"] == r2["canonical_path"]

    def test_different_content_gets_separate_canonical_files(self):
        r1 = document_store.store_document("https://a.com/job/1", "notice.pdf", b"content A")
        r2 = document_store.store_document("https://a.com/job/1", "other.pdf", b"content B")

        assert r1["sha256"] != r2["sha256"]
        assert r1["canonical_path"] != r2["canonical_path"]
        assert Path(r1["canonical_path"]).exists()
        assert Path(r2["canonical_path"]).exists()

    def test_same_job_same_filename_overwrites_pointer_not_canonical(self):
        """Re-encountering the exact same document on a retry shouldn't
        duplicate the canonical store or the job's own pointer."""
        r1 = document_store.store_document("https://a.com/job/1", "notice.pdf", b"v1 content")
        r2 = document_store.store_document("https://a.com/job/1", "notice.pdf", b"v1 content")

        assert r1["canonical_path"] == r2["canonical_path"]
        assert r1["pointer_path"] == r2["pointer_path"]
        assert r2["is_new_document"] is False

    def test_pointer_file_is_not_a_copy_of_the_document(self):
        """The per-job pointer must be a small JSON reference, not another
        full copy of the bytes -- that's the whole point of dedup."""
        result = document_store.store_document("https://a.com/job/1", "notice.pdf", b"x" * 100_000)

        pointer_path = Path(result["pointer_path"])
        assert pointer_path.exists()
        assert pointer_path.stat().st_size < 1000

    def test_unsafe_filename_characters_sanitized(self):
        result = document_store.store_document("https://a.com/job/1", "../../etc/passwd", b"bytes")

        canonical = Path(result["canonical_path"])
        assert canonical.exists()
        # Must resolve inside the canonical dir, not escape it via traversal.
        assert canonical.resolve().is_relative_to(document_store._canonical_dir().resolve())


class TestListJobDocuments:
    def test_empty_for_unknown_job(self):
        assert document_store.list_job_documents("https://a.com/never-seen") == []

    def test_lists_all_documents_for_a_job(self):
        document_store.store_document("https://a.com/job/1", "resume.pdf", b"resume bytes")
        document_store.store_document("https://a.com/job/1", "notice.pdf", b"notice bytes")

        docs = document_store.list_job_documents("https://a.com/job/1")

        assert len(docs) == 2
        filenames = {d["original_filename"] for d in docs}
        assert filenames == {"resume.pdf", "notice.pdf"}

    def test_different_jobs_dont_see_each_others_documents(self):
        document_store.store_document("https://a.com/job/1", "notice.pdf", b"shared bytes")
        document_store.store_document("https://b.com/job/2", "notice.pdf", b"shared bytes")

        docs_a = document_store.list_job_documents("https://a.com/job/1")
        docs_b = document_store.list_job_documents("https://b.com/job/2")

        assert len(docs_a) == 1
        assert len(docs_b) == 1
        # Both point at the SAME canonical (deduped) file.
        assert docs_a[0]["canonical_path"] == docs_b[0]["canonical_path"]
