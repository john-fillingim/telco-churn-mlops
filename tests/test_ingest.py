"""Tests for GCS ingest caching (the GCS download is mocked)."""

from dataclasses import replace

import src.data.ingest as ingest
from src.config import load_config


def _cfg(tmp_path):
    cfg = load_config("config.yaml")
    return replace(cfg, data=replace(cfg.data, local_path=str(tmp_path / "clean.csv")))


def _fake_download(calls):
    def fake(bucket, blob, dest):
        calls.append((bucket, blob))
        dest.write_text("tenure,Churn\n1,0\n")

    return fake


def test_downloads_when_missing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ingest, "download_from_gcs", _fake_download(calls))
    df = ingest.load_data(_cfg(tmp_path))
    assert len(calls) == 1
    assert len(df) == 1


def test_uses_cache_when_present(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ingest, "download_from_gcs", _fake_download(calls))
    cfg = _cfg(tmp_path)
    ingest.load_data(cfg)
    ingest.load_data(cfg)
    assert len(calls) == 1  # second call used the cache


def test_force_redownloads(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ingest, "download_from_gcs", _fake_download(calls))
    cfg = _cfg(tmp_path)
    ingest.load_data(cfg)
    ingest.load_data(cfg, force=True)
    assert len(calls) == 2
