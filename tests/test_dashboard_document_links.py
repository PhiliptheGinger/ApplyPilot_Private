"""Regression test for the 2026-09-19 dashboard fix: manual_only jobs (no
application_url, e.g. LinkedIn's own scraper gap) still get real tailored
materials generated, but the dashboard had no direct way to open the
actual document -- `_read_file_safe` only ever previewed `.txt` paths,
while `tailored_resume_path`/`cover_letter_path` store the real .pdf/.docx
document under the current default doc-format. `_build_document_link`
adds a direct file:// link independent of that text-preview path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.view import _build_document_link


def test_returns_empty_for_missing_path():
    assert _build_document_link(None, "Resume") == ""
    assert _build_document_link("", "Resume") == ""


def test_returns_empty_when_file_does_not_exist(tmp_path):
    missing = tmp_path / "nope.pdf"
    assert _build_document_link(str(missing), "Resume") == ""


def test_links_to_a_real_pdf_file(tmp_path):
    doc = tmp_path / "Philip_McLaughlin_SQL_DBA.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")

    html = _build_document_link(str(doc), "Resume")

    assert "file:///" in html
    assert str(doc).replace("\\", "\\") in html or doc.name in html
    assert "PDF" in html
    assert "Resume" in html


def test_links_to_a_real_docx_file(tmp_path):
    doc = tmp_path / "cover.docx"
    doc.write_bytes(b"fake docx bytes")

    html = _build_document_link(str(doc), "Cover Letter")

    assert "DOCX" in html
    assert "Cover Letter" in html
