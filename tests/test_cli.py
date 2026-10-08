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


def test_project_root_prefers_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """In a container the package is installed into site-packages, so walking up
    from cli.py lands somewhere unrelated to the project."""
    from keel.cli import _project_root

    monkeypatch.setenv("KEEL_PROJECT_ROOT", str(tmp_path))
    assert _project_root() == tmp_path


def test_project_root_falls_back_to_the_source_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from keel.cli import _project_root

    monkeypatch.delenv("KEEL_PROJECT_ROOT", raising=False)
    root = _project_root()
    assert (root / "alembic.ini").exists()
    assert (root / "alembic" / "versions").is_dir()


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
