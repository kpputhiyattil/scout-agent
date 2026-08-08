"""Unit tests for YouTube JS-runtime discovery used by yt-dlp."""
from __future__ import annotations

from pathlib import Path

import pytest

from scout import ingest


def test_find_executable_prefers_path(monkeypatch, tmp_path: Path):
    fake = tmp_path / "deno.exe"
    fake.write_text("")
    monkeypatch.setattr(ingest.shutil, "which", lambda name: str(fake) if name == "deno" else None)
    assert ingest._find_executable("deno") == str(fake)


def test_js_runtimes_uses_deno_when_available(monkeypatch, tmp_path: Path):
    fake = tmp_path / "deno.exe"
    fake.write_text("")
    monkeypatch.setattr(ingest, "_find_executable", lambda name: str(fake) if name == "deno" else None)
    assert ingest._js_runtimes() == {"deno": {"path": str(fake)}}


def test_js_runtimes_falls_back_to_node(monkeypatch, tmp_path: Path):
    fake = tmp_path / "node.exe"
    fake.write_text("")
    monkeypatch.setattr(
        ingest, "_find_executable",
        lambda name: str(fake) if name == "node" else None,
    )
    assert ingest._js_runtimes() == {"node": {"path": str(fake)}}


def test_js_runtimes_raises_when_missing(monkeypatch):
    monkeypatch.setattr(ingest, "_find_executable", lambda name: None)
    with pytest.raises(RuntimeError, match="JavaScript runtime"):
        ingest._js_runtimes()
