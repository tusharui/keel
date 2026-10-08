from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from keel.clock import Clock, default_clock
from keel.enums import CacheTier
from keel.sim_engine.model import stable_digest


@dataclass(slots=True)
class CacheHit:
    value: str
    tier: CacheTier
    key: str
    similarity: float = 1.0


@runtime_checkable
class ResponseCache(Protocol):
    def lookup(self, key: str) -> CacheHit | None: ...
    def store(self, key: str, value: str) -> None: ...
    @property
    def stats(self) -> CacheStats: ...


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass(slots=True)
class LRUCache:
    """Exact-match cache with least-recently-used eviction.

    Exact matching is the only tier with no correctness risk: the key covers
    every parameter that can change the output, so a hit is the same answer the
    model would have produced.
    """

    max_entries: int
    clock: Clock = field(default_factory=default_clock)
    stats: CacheStats = field(default_factory=CacheStats)
    _entries: OrderedDict[str, str] = field(default_factory=OrderedDict, init=False)

    def __post_init__(self) -> None:
        if self.max_entries < 0:
            raise ValueError("max_entries cannot be negative")

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def lookup(self, key: str) -> CacheHit | None:
        value = self._entries.get(key)
        if value is None:
            self.stats.misses += 1
            return None
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return CacheHit(value=value, tier=CacheTier.EXACT, key=key)

    def store(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        self._entries[key] = value
        self._entries.move_to_end(key)
        self.stats.stores += 1
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.stats.evictions += 1

    def clear(self) -> None:
        self._entries.clear()

    def keys(self) -> list[str]:
        return list(self._entries)


def request_key(
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    stop: tuple[str, ...] = (),
    temperature: float = 0.0,
) -> str:
    """Identity of a request.

    Every parameter that can change the output belongs in the key. Leaving one
    out turns a correctness feature into a silent bug, and it is the kind of bug
    that only shows up for the one tenant who set temperature to 0.7.
    """
    return f"{stable_digest(model, prompt, max_tokens, stop, temperature):016x}"
