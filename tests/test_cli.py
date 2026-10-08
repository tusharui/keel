from __future__ import annotations

import pytest

from keel import __version__
from keel.cli import main


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_unknown_command_exits() -> None:
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_demo_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "requests              6/6" in out
    assert "kv prefix cache" in out


def test_bench_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    assert main(["bench", "--requests", "6", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert {"scenario", "makespan_s", "goodput_tokens_per_second"} <= payload[0].keys()


def test_db_upgrade_creates_schema(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy import create_engine, inspect

    monkeypatch.setenv("KEEL_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'migrated.db'}")
    from keel.config import reset_settings_cache

    reset_settings_cache()
    try:
        assert main(["db", "head"]) == 0
    finally:
        reset_settings_cache()

    sync_engine = create_engine(f"sqlite:///{tmp_path / 'migrated.db'}")
    tables = set(inspect(sync_engine).get_table_names())
    assert {"tenants", "inference_requests", "runs", "nodes", "cache_entries"} <= tables
