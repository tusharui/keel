from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from keel.errors import ConfigError

AdmissionPolicy = Literal["fcfs", "sjf", "deadline"]
JitterStrategy = Literal["none", "full", "equal"]

_ENV_PREFIX = "KEEL_"


class _Base(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=_ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )


class EngineConfig(_Base):
    context_length: int = Field(default=4096, ge=128)
    block_size: int = Field(default=16, ge=1)
    kv_blocks: int = Field(default=4096, ge=1)
    vocab_size: int = Field(default=32_000, ge=8)
    seed: int = Field(default=0xC0FFEE, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blocks_per_sequence(self) -> int:
        return -(-self.context_length // self.block_size)

    @model_validator(mode="after")
    def _pool_holds_a_full_sequence(self) -> Self:
        # A sequence that cannot be fully resident can never be scheduled, so
        # every request would deadlock rather than fail.
        if self.kv_blocks < self.blocks_per_sequence:
            raise ValueError(
                f"kv_blocks={self.kv_blocks} cannot hold one max-length sequence "
                f"({self.blocks_per_sequence} blocks of size {self.block_size}); "
                "every request would deadlock"
            )
        return self


class SchedulerConfig(_Base):
    max_num_seqs: int = Field(default=64, ge=1)
    max_num_batched_tokens: int = Field(default=8192, ge=1)
    chunked_prefill_tokens: int = Field(default=2048, ge=1)
    policy: AdmissionPolicy = "fcfs"
    enable_preemption: bool = True
    default_max_tokens: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def _decode_budget_is_usable(self) -> Self:
        if self.max_num_batched_tokens < 2:
            raise ValueError(
                "max_num_batched_tokens must leave room for at least one decode "
                "token after a full chunked-prefill pass, otherwise prefill can "
                "never yield and every request stalls"
            )
        if self.chunked_prefill_tokens > self.max_num_batched_tokens:
            raise ValueError(
                f"chunked_prefill_tokens={self.chunked_prefill_tokens} exceeds the "
                f"batch budget {self.max_num_batched_tokens}; the chunk could never "
                "be admitted in one pass"
            )
        return self


class CacheConfig(_Base):
    exact_max_entries: int = Field(default=1024, ge=0)
    prefix_enabled: bool = True
    semantic_enabled: bool = False
    # Measured against the hashed-trigram embedder rather than picked by feel.
    # "capital of France" against "capital of Spain" scores ~0.77, so anything
    # below that serves the wrong country; a case-and-punctuation variant of the
    # same question scores ~0.95. 0.90 sits in that gap.
    semantic_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    semantic_max_entries: int = Field(default=512, ge=0)

    @model_validator(mode="after")
    def _semantic_off_means_threshold_is_inert(self) -> Self:
        if not self.semantic_enabled and self.semantic_threshold < 1.0:
            # Allowed, but a permissive threshold against a disabled tier is the
            # kind of setting that silently ships a correctness bug later.
            object.__setattr__(self, "semantic_threshold", 1.0)
        return self


class ReliabilityConfig(_Base):
    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay_s: float = Field(default=0.05, ge=0.0)
    max_delay_s: float = Field(default=2.0, ge=0.0)
    jitter: JitterStrategy = "full"
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_reset_timeout_s: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def _ceiling_is_above_the_floor(self) -> Self:
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s must be >= base_delay_s")
        return self


class PolicyConfig(_Base):
    rate_limit_per_s: float = Field(default=50.0, gt=0.0)
    rate_limit_burst: float = Field(default=100.0, gt=0.0)
    default_budget_micros: int = Field(default=100_000_000, ge=0)

    @model_validator(mode="after")
    def _burst_covers_a_second_of_traffic(self) -> Self:
        if self.rate_limit_burst < self.rate_limit_per_s:
            raise ValueError(
                f"rate_limit_burst={self.rate_limit_burst} is below "
                f"rate_limit_per_s={self.rate_limit_per_s}; the limiter would refuse "
                "traffic it is nominally allowing"
            )
        return self


class Settings(_Base):
    database_url: str = "sqlite+aiosqlite:///./keel.db"
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    log_json: bool = True

    engine: EngineConfig = Field(default_factory=EngineConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    @property
    def context_length(self) -> int:
        return self.engine.context_length

    @model_validator(mode="after")
    def _scheduler_fits_the_pool(self) -> Self:
        # Sequences beyond the resident block count can never all run at once.
        # Allowing it silently converts into preemption pressure under load
        # rather than an obvious misconfiguration.
        if self.scheduler.max_num_seqs > self.engine.kv_blocks:
            raise ValueError(
                f"max_num_seqs={self.scheduler.max_num_seqs} exceeds "
                f"kv_blocks={self.engine.kv_blocks}"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc


def reset_settings_cache() -> None:
    get_settings.cache_clear()
