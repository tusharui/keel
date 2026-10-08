from __future__ import annotations

import pytest
from pydantic import ValidationError

from keel.config import (
    CacheConfig,
    EngineConfig,
    ReliabilityConfig,
    SchedulerConfig,
    Settings,
    get_settings,
    reset_settings_cache,
)
from keel.errors import ConfigError


def test_defaults_are_usable() -> None:
    settings = Settings()
    assert settings.scheduler.max_num_seqs <= settings.engine.kv_blocks


def test_database_url_defaults_to_sqlite() -> None:
    """Checked against the field default, not a constructed Settings.

    Settings() resolves the environment, so asserting on the instance fails for
    anyone with KEEL_DATABASE_URL set to Postgres, which is exactly what the
    Postgres job in CI does.
    """
    default = Settings.model_fields["database_url"].default
    assert str(default).startswith("sqlite+")


def test_env_prefix_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEEL_DATABASE_URL", "postgresql+asyncpg://localhost/keel")
    monkeypatch.setenv("KEEL_LOG_JSON", "false")
    settings = Settings()
    assert settings.database_url == "postgresql+asyncpg://localhost/keel"
    assert settings.log_json is False


def test_nested_env_uses_double_underscore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEEL_SCHEDULER__MAX_NUM_SEQS", "8")
    monkeypatch.setenv("KEEL_ENGINE__BLOCK_SIZE", "32")
    settings = Settings()
    assert settings.scheduler.max_num_seqs == 8
    assert settings.engine.block_size == 32


def test_pool_too_small_to_hold_one_sequence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="deadlock"):
        EngineConfig(context_length=4096, block_size=16, kv_blocks=4)


def test_pool_exactly_fitting_one_sequence_is_allowed() -> None:
    engine = EngineConfig(context_length=128, block_size=16, kv_blocks=8)
    assert engine.blocks_per_sequence == 8


def test_blocks_per_sequence_rounds_up() -> None:
    engine = EngineConfig(context_length=129, block_size=16, kv_blocks=9)
    assert engine.blocks_per_sequence == 9


def test_scheduler_budget_that_cannot_yield_is_rejected() -> None:
    with pytest.raises(ValidationError, match="stall"):
        SchedulerConfig(max_num_batched_tokens=1)


def test_more_sequences_than_blocks_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        Settings(
            engine={"context_length": 128, "block_size": 16, "kv_blocks": 8},
            scheduler={"max_num_seqs": 64},
        )


def test_disabling_semantic_neutralises_a_permissive_threshold() -> None:
    cache = CacheConfig(semantic_enabled=False, semantic_threshold=0.5)
    assert cache.semantic_threshold == 1.0


def test_enabled_semantic_cache_keeps_its_threshold() -> None:
    cache = CacheConfig(semantic_enabled=True, semantic_threshold=0.9)
    assert cache.semantic_threshold == 0.9


def test_retry_ceiling_below_floor_is_rejected() -> None:
    with pytest.raises(ValidationError, match="max_delay_s"):
        ReliabilityConfig(base_delay_s=1.0, max_delay_s=0.1)


def test_settings_are_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.log_level = "debug"  # type: ignore[misc]


def test_bad_env_value_surfaces_as_config_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KEEL_SCHEDULER__POLICY", "random")
    reset_settings_cache()
    try:
        with pytest.raises(ConfigError, match="invalid configuration"):
            get_settings()
    finally:
        reset_settings_cache()


def test_settings_singleton_is_cached() -> None:
    reset_settings_cache()
    assert get_settings() is get_settings()
    reset_settings_cache()
