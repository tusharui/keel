from __future__ import annotations

from dataclasses import dataclass, field

from keel.cache.base import CacheHit, CacheStats, LRUCache, request_key
from keel.cache.semantic import SemanticCache
from keel.clock import Clock, default_clock
from keel.config import CacheConfig


@dataclass(slots=True)
class TieredCache:
    """Exact match first, semantic similarity as a fallback.

    Exact is checked first because it is a hash lookup and cannot be wrong.
    Semantic only runs when exact missed, so widening the threshold never puts a
    possibly-wrong answer ahead of a certainly-right one.
    """

    config: CacheConfig = field(default_factory=CacheConfig)
    clock: Clock = field(default_factory=default_clock)
    _exact: LRUCache = field(init=False)
    _semantic: SemanticCache = field(init=False)

    def __post_init__(self) -> None:
        self._exact = LRUCache(self.config.exact_max_entries, clock=self.clock)
        self._semantic = SemanticCache(
            max_entries=self.config.semantic_max_entries,
            threshold=self.config.semantic_threshold,
            clock=self.clock,
        )

    @property
    def exact(self) -> LRUCache:
        return self._exact

    @property
    def semantic(self) -> SemanticCache:
        return self._semantic

    @property
    def stats(self) -> CacheStats:
        return self._exact.stats

    def key_for(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        stop: tuple[str, ...] = (),
        temperature: float = 0.0,
    ) -> str:
        return request_key(model, prompt, max_tokens=max_tokens, stop=stop, temperature=temperature)

    def lookup(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        stop: tuple[str, ...] = (),
        temperature: float = 0.0,
        allow_semantic: bool = True,
    ) -> CacheHit | None:
        key = self.key_for(model, prompt, max_tokens=max_tokens, stop=stop, temperature=temperature)

        hit = self._exact.lookup(key)
        if hit is not None:
            return hit

        if allow_semantic:
            return self._semantic.lookup(key, prompt)
        self._semantic.stats.misses += 1
        return None

    def store(
        self,
        model: str,
        prompt: str,
        value: str,
        *,
        max_tokens: int,
        stop: tuple[str, ...] = (),
        temperature: float = 0.0,
    ) -> None:
        key = self.key_for(model, prompt, max_tokens=max_tokens, stop=stop, temperature=temperature)
        self._exact.store(key, value)
        self._semantic.store(key, value, prompt)

    def clear(self) -> None:
        self._exact.clear()
        self._semantic.clear()
