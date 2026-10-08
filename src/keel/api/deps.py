from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from keel.cache.tiers import TieredCache
from keel.clock import Clock, ManualClock, SystemClock
from keel.config import Settings, get_settings
from keel.dag.executor import Executor, RunResult
from keel.dag.graph import DAG
from keel.dag.state import RunStateStore, wrap_dag
from keel.db.session import build_engine, build_session_factory
from keel.enums import CacheTier
from keel.errors import (
    BudgetExceeded,
    CircuitOpen,
    ConfigError,
    ContextLengthExceeded,
    KeelError,
    QuotaExceeded,
    ValidationError,
)
from keel.policy.budgets import BudgetTracker
from keel.policy.ratelimit import TenantRateLimiter
from keel.sim_engine import DeviceProfile, SimModel, Tokenizer
from keel.sim_engine.engine import FinishReason, InferenceEngine
from keel.sim_engine.kv_cache import KVCacheManager
from keel.tracing.spans import Tracer

MICROS_PER_TOKEN = 10


@dataclass(slots=True)
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_micros: int = 0
    ttfts: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompletionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    ttft_ms: float
    tpot_ms: float | None
    cache_tier: str
    cache_similarity: float | None
    cost_micros: int


@dataclass
class Service:
    """Everything a request touches, assembled once at startup.

    The simulated backend is driven by a ManualClock, so a request does not
    actually sleep. The engine computes what a real backend would have cost in
    wall time and the response reports that figure. Nothing here pretends the
    latency was measured on hardware.
    """

    settings: Settings
    engine: InferenceEngine
    cache: TieredCache
    limiter: TenantRateLimiter
    budgets: BudgetTracker
    tracer: Tracer
    db_engine: AsyncEngine
    sessions: async_sessionmaker[AsyncSession]
    clock: Clock = field(default_factory=ManualClock)
    dags: dict[str, DAG] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def build(cls, settings: Settings | None = None) -> Service:
        resolved = settings or get_settings()
        tokenizer = Tokenizer(vocab_size=32_000)
        kv = KVCacheManager(
            num_blocks=resolved.engine.kv_blocks,
            block_size=resolved.engine.block_size,
        )
        engine = InferenceEngine(
            tokenizer=tokenizer,
            model=SimModel(vocab_size=32_000, seed=resolved.engine.seed),
            device=DeviceProfile(),
            kv=kv,
            context_length=resolved.context_length,
            clock=ManualClock(),
        )
        db_engine = build_engine(resolved.database_url)
        clock = ManualClock()
        return cls(
            settings=resolved,
            engine=engine,
            cache=TieredCache(config=resolved.cache, clock=clock),
            limiter=TenantRateLimiter(
                rate_per_s=resolved.policy.rate_limit_per_s,
                capacity=resolved.policy.rate_limit_burst,
                clock=clock,
            ),
            budgets=BudgetTracker(
                clock=clock, default_limit_micros=resolved.policy.default_budget_micros
            ),
            # Tracing gets a real clock rather than the service's virtual one.
            # The service deliberately does not sleep for simulated latency, so
            # sharing the manual clock would record every span as 0ms and the
            # trace would carry structure but no timing at all.
            tracer=Tracer(clock=SystemClock()),
            db_engine=db_engine,
            sessions=build_session_factory(db_engine),
            clock=clock,
        )

    def register_dag(self, name: str, dag: DAG) -> None:
        self.dags[name] = dag

    def seed_budget(self, tenant_id: str, limit_micros: int) -> None:
        self.budgets.set_limit(tenant_id, limit_micros)

    @property
    def tenants(self) -> list[str]:
        return sorted(self.budgets.tenants)

    async def aclose(self) -> None:
        await self.db_engine.dispose()

    # --- admission -------------------------------------------------------------------

    def admit(self, tenant_id: str, estimated_micros: int) -> float:
        """Returns milliseconds to wait, or raises.

        Both limits run before any model work, so a throttled caller costs a hash
        lookup rather than a generation. Budget exhaustion raises BudgetExceeded
        rather than QuotaExceeded because the two need different answers from the
        client: one is retry in a moment, the other is a plan change.
        """
        if not self.limiter.try_consume(tenant_id):
            return self.limiter.retry_after(tenant_id) * 1000.0
        self.budgets.reserve(tenant_id, estimated_micros)
        return 0.0

    # --- completions -----------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        stop: tuple[str, ...] = (),
        temperature: float = 0.0,
        tenant_id: str = "default",
        allow_semantic_cache: bool = True,
    ) -> CompletionResult:
        with self.tracer.span("completion", model=model, tenant=tenant_id):
            hit = self.cache.lookup(
                model,
                prompt,
                max_tokens=max_tokens,
                stop=stop,
                temperature=temperature,
                allow_semantic=allow_semantic_cache,
            )
            if hit is not None:
                self.usage.requests += 1
                return CompletionResult(
                    text=hit.value,
                    prompt_tokens=0,
                    completion_tokens=0,
                    finish_reason=FinishReason.STOP.value,
                    ttft_ms=0.0,
                    tpot_ms=None,
                    cache_tier=hit.tier.value,
                    cache_similarity=(hit.similarity if hit.tier is CacheTier.SEMANTIC else None),
                    cost_micros=0,
                )

            with self.tracer.span("generate") as span:
                generation = self.engine.run(
                    self.engine.tokenizer.encode(prompt), max_tokens, stop=stop
                )
                # Wall-clock duration says how long the simulation took to compute.
                # The simulated figures say what it cost a real backend, which is
                # the number an operator actually needs from this endpoint.
                span.attributes.update(
                    sim_ttft_ms=generation.ttft_ms,
                    sim_duration_ms=generation.duration_ms,
                    sim_completion_tokens=generation.completion_tokens,
                )

            text = self.engine.tokenizer.decode(generation.token_ids)
            cost = (generation.prompt_tokens + generation.completion_tokens) * MICROS_PER_TOKEN

            self.usage.prompt_tokens += generation.prompt_tokens
            self.usage.completion_tokens += generation.completion_tokens
            self.usage.cost_micros += cost
            self.usage.requests += 1
            self.usage.ttfts.append(generation.ttft_ms)

            self.cache.store(
                model,
                prompt,
                text,
                max_tokens=max_tokens,
                stop=stop,
                temperature=temperature,
            )

            return CompletionResult(
                text=text,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                finish_reason=generation.finish_reason.value,
                ttft_ms=generation.ttft_ms,
                tpot_ms=generation.tpot_ms,
                cache_tier=CacheTier.MISS.value,
                cache_similarity=None,
                cost_micros=cost,
            )

    # --- dag runs ---------------------------------------------------------------------

    async def run_dag(
        self, name: str, partition: str | None, inputs: dict[str, object]
    ) -> tuple[str, RunResult, list[str]]:
        dag = self.dags[name]
        async with self.sessions() as session:
            store = RunStateStore(session, name, partition)
            run = await store.start(input_hash=partition or name)
            wrapped, reused = wrap_dag(dag, store, inputs=inputs)
            outcome = await Executor(wrapped).run(inputs)
            return run.id, outcome, reused


def error_status(error: Exception) -> int:
    """Map the error taxonomy onto HTTP.

    Kept beside the service rather than scattered across handlers so a new error
    class has exactly one place to be given a status.
    """
    if isinstance(error, (ValidationError, ConfigError, ContextLengthExceeded)):
        return 400
    if isinstance(error, BudgetExceeded):
        return 402
    if isinstance(error, QuotaExceeded):
        return 429
    if isinstance(error, CircuitOpen):
        return 503
    if isinstance(error, KeelError):
        return 500
    return 500
