from __future__ import annotations

from dataclasses import dataclass, field

from keel.cache.base import CacheHit, CacheStats
from keel.clock import Clock, default_clock
from keel.enums import CacheTier
from keel.sim_engine.model import stable_digest

DEFAULT_DIMS = 1024
NGRAM = 3


def hashed_embedding(text: str, dims: int = DEFAULT_DIMS) -> tuple[float, ...]:
    """Hashed character-trigram bag, L2 normalised.

    A stand-in for a real sentence embedding: it captures lexical overlap, which
    is what distinguishes "what is the capital of France" from "what is the
    capital of Spain", without shipping model weights or calling an API. It will
    not see paraphrases that share no trigrams, which is exactly where a real
    embedder earns its keep.

    Cosine similarity reduces to a dot product because both sides are unit
    length, so lookup is a plain sum rather than a division per candidate.
    """
    normalised = f" {text.strip().lower()} "
    vector = [0.0] * dims
    for start in range(len(normalised) - NGRAM + 1):
        gram = normalised[start : start + NGRAM]
        vector[stable_digest(gram) % dims] += 1.0

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(slots=True)
class SemanticEntry:
    vector: tuple[float, ...]
    value: str


@dataclass(slots=True)
class SemanticCache:
    """Serves a cached answer for a prompt that is similar but not identical.

    The risk is not staleness, it is answering the wrong question. A prompt one
    word different can share most of its trigrams while needing a different
    answer, so the threshold is the whole safety mechanism and the caller is
    handed the similarity alongside the value so a hit can be audited rather
    than trusted.

    Lookup is a linear scan, which is fine for the hundreds of entries this is
    sized for and not for tens of thousands. A real deployment swaps the scan for
    an approximate nearest-neighbour index; the threshold semantics do not change.
    """

    max_entries: int
    threshold: float = 0.95
    dims: int = DEFAULT_DIMS
    clock: Clock = field(default_factory=default_clock)
    stats: CacheStats = field(default_factory=CacheStats)
    rejected_below_threshold: int = 0
    _entries: list[SemanticEntry] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.max_entries < 0:
            raise ValueError("max_entries cannot be negative")

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def lookup(self, key: str, prompt: str | None = None) -> CacheHit | None:
        """``prompt`` is required; ``key`` only labels the hit for auditing."""
        if not self.enabled or prompt is None:
            return None

        query = hashed_embedding(prompt, self.dims)
        best: SemanticEntry | None = None
        best_score = 0.0
        for entry in self._entries:
            score = cosine(query, entry.vector)
            if score > best_score:
                best_score = score
                best = entry

        if best is None or best_score < self.threshold:
            self.stats.misses += 1
            self.rejected_below_threshold += 1
            return None

        self.stats.hits += 1
        return CacheHit(
            value=best.value,
            tier=CacheTier.SEMANTIC,
            key=key,
            similarity=round(best_score, 6),
        )

    def store(self, key: str, value: str, prompt: str | None = None) -> None:
        if not self.enabled or prompt is None:
            return
        self._entries.append(SemanticEntry(vector=hashed_embedding(prompt, self.dims), value=value))
        self.stats.stores += 1
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)
            self.stats.evictions += 1

    def clear(self) -> None:
        self._entries.clear()
