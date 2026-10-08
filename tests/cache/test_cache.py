from __future__ import annotations

import pytest

from keel.cache.base import LRUCache, request_key
from keel.cache.semantic import SemanticCache, cosine, hashed_embedding
from keel.cache.tiers import TieredCache
from keel.clock import ManualClock
from keel.enums import CacheTier


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


# --- exact -------------------------------------------------------------------------


def test_miss_then_hit() -> None:
    cache = LRUCache(max_entries=4)
    assert cache.lookup("a") is None
    cache.store("a", "1")
    hit = cache.lookup("a")
    assert hit is not None
    assert hit.value == "1"
    assert hit.tier is CacheTier.EXACT


def test_hit_rate_is_tracked() -> None:
    cache = LRUCache(max_entries=4)
    cache.lookup("missing")
    cache.store("a", "1")
    cache.lookup("a")
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == pytest.approx(0.5)


def test_evicts_least_recently_used() -> None:
    cache = LRUCache(max_entries=2)
    cache.store("a", "1")
    cache.store("b", "2")
    cache.lookup("a")
    cache.store("c", "3")

    assert cache.lookup("b") is None, "b was the least recently used"
    assert cache.lookup("a") is not None
    assert cache.lookup("c") is not None
    assert cache.stats.evictions == 1


def test_store_refreshes_recency() -> None:
    cache = LRUCache(max_entries=2)
    cache.store("a", "1")
    cache.store("b", "2")
    cache.store("a", "1-updated")
    cache.store("c", "3")
    assert cache.lookup("a") is not None


def test_zero_capacity_disables_the_cache() -> None:
    cache = LRUCache(max_entries=0)
    cache.store("a", "1")
    assert cache.size == 0
    assert cache.lookup("a") is None
    assert not cache.enabled


def test_negative_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        LRUCache(max_entries=-1)


def test_overwriting_does_not_grow() -> None:
    cache = LRUCache(max_entries=2)
    for _ in range(10):
        cache.store("a", "1")
    assert cache.size == 1


# --- request keys -------------------------------------------------------------------


def test_same_request_hashes_identically() -> None:
    a = request_key("m", "hello", max_tokens=10)
    assert a == request_key("m", "hello", max_tokens=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "other"},
        {"prompt": "goodbye"},
        {"max_tokens": 20},
        {"stop": ("x",)},
        {"temperature": 0.7},
    ],
)
def test_every_output_affecting_parameter_changes_the_key(kwargs: dict[str, object]) -> None:
    base = {"model": "m", "prompt": "hello", "max_tokens": 10}
    assert request_key(**base) != request_key(**{**base, **kwargs})


# --- embeddings ----------------------------------------------------------------------


def test_embedding_ignores_case_and_surrounding_space() -> None:
    assert hashed_embedding("what is the capital of france") == hashed_embedding(
        "  What is the capital of France  "
    )


def test_embedding_notices_punctuation() -> None:
    """Punctuation changes the trigrams, so a threshold tuned for reformatted
    text will miss a question that only gained a question mark."""
    assert hashed_embedding("what is the capital of france") != hashed_embedding(
        "what is the capital of france?"
    )


def test_embedding_is_normalised() -> None:
    vector = hashed_embedding("hello world")
    norm = sum(v * v for v in vector) ** 0.5
    assert norm == pytest.approx(1.0)


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    vector = hashed_embedding("capital of france")
    assert cosine(vector, vector) == pytest.approx(1.0)


def test_unrelated_texts_score_low() -> None:
    a = hashed_embedding("what is the capital of france")
    b = hashed_embedding("summarise the quarterly revenue report for finance")
    assert cosine(a, b) < 0.35


def test_one_word_apart_still_scores_moderately_high() -> None:
    """This is the failure mode of a lexical embedding, and the reason the
    threshold has to sit well above the plausible-answer band.

    'capital of France' and 'capital of Spain' differ by one word out of five and
    still score around 0.77. A threshold anywhere below that serves the wrong
    country. A real sentence embedder separates these further; a bag of trigrams
    does not, and the cache reports its similarity precisely so the risk is
    visible instead of hidden behind a boolean.
    """
    france = hashed_embedding("what is the capital of france")
    spain = hashed_embedding("what is the capital of spain")
    score = cosine(france, spain)
    assert 0.6 < score < 0.9


def test_semantic_cache_would_answer_wrongly_at_a_loose_threshold() -> None:
    """Documents the sharp edge rather than pretending it is not there."""
    loose = SemanticCache(max_entries=10, threshold=0.70)
    loose.store("k", "Paris", prompt="what is the capital of france")

    hit = loose.lookup("k2", "what is the capital of spain")
    assert hit is not None
    assert hit.value == "Paris", "the wrong answer, which is why 0.70 is unusable"

    strict = SemanticCache(max_entries=10, threshold=0.95)
    strict.store("k", "Paris", prompt="what is the capital of france")
    assert strict.lookup("k2", "what is the capital of spain") is None


def test_near_duplicates_score_high() -> None:
    a = hashed_embedding("what is the capital of france")
    b = hashed_embedding("what is the capital of france ")
    assert cosine(a, b) > 0.99


def test_empty_text_does_not_divide_by_zero() -> None:
    vector = hashed_embedding("")
    assert cosine(vector, vector) == 0.0


# --- semantic ------------------------------------------------------------------------


def test_semantic_cache_serves_a_close_prompt() -> None:
    cache = SemanticCache(max_entries=10, threshold=0.9)
    cache.store("k1", "Paris", prompt="what is the capital of france")
    hit = cache.lookup("k2", "what is the capital of France ")
    assert hit is not None
    assert hit.value == "Paris"
    assert hit.tier is CacheTier.SEMANTIC
    assert hit.similarity > 0.9


def test_semantic_cache_rejects_different_questions() -> None:
    """The whole point of the threshold. Sharing trigrams is not the same as
    asking the same question."""
    cache = SemanticCache(max_entries=10, threshold=0.95)
    cache.store("k1", "Paris", prompt="what is the capital of france")
    assert cache.lookup("k2", "summarise the quarterly revenue report") is None
    assert cache.rejected_below_threshold >= 1


def test_tighter_threshold_rejects_more() -> None:
    prompt_a = "what is the capital of france"
    prompt_b = "what is the capital of france please explain"
    loose = SemanticCache(max_entries=10, threshold=0.75)
    loose.store("k", "Paris", prompt=prompt_a)
    strict = SemanticCache(max_entries=10, threshold=0.95)
    strict.store("k", "Paris", prompt=prompt_a)

    assert loose.lookup("k2", prompt_b) is not None
    assert strict.lookup("k2", prompt_b) is None


def test_semantic_hit_reports_its_similarity() -> None:
    cache = SemanticCache(max_entries=10, threshold=0.9)
    cache.store("k", "Paris", prompt="what is the capital of france")
    hit = cache.lookup("k", "what is the capital of France")
    assert hit is not None
    assert 0.9 <= hit.similarity <= 1.0


def test_semantic_cache_evicts_oldest() -> None:
    cache = SemanticCache(max_entries=2, threshold=0.99)
    cache.store("a", "1", prompt="alpha question")
    cache.store("b", "2", prompt="beta question")
    cache.store("c", "3", prompt="gamma question")
    assert cache.size == 2
    assert cache.stats.evictions == 1
    assert cache.lookup("x", "alpha question") is None


def test_semantic_cache_disabled_at_zero_capacity() -> None:
    cache = SemanticCache(max_entries=0, threshold=0.5)
    cache.store("k", "v", prompt="hello")
    assert cache.lookup("k", "hello") is None
    assert not cache.enabled


def test_semantic_threshold_is_validated() -> None:
    with pytest.raises(ValueError, match="threshold"):
        SemanticCache(max_entries=4, threshold=1.5)
    with pytest.raises(ValueError, match="negative"):
        SemanticCache(max_entries=-1)


def test_lookup_without_a_prompt_is_a_miss() -> None:
    cache = SemanticCache(max_entries=4, threshold=0.5)
    cache.store("k", "v", prompt="hello")
    assert cache.lookup("k") is None


# --- tiered ---------------------------------------------------------------------------


def test_tiered_prefers_exact_over_semantic() -> None:
    cache = TieredCache(clock=ManualClock())
    cache.store("m", "what is the capital of france", "Paris", max_tokens=10)
    hit = cache.lookup("m", "what is the capital of france", max_tokens=10)
    assert hit is not None
    assert hit.tier is CacheTier.EXACT


def test_tiered_falls_back_to_semantic() -> None:
    cache = TieredCache(clock=ManualClock())
    cache.store("m", "what is the capital of france", "Paris", max_tokens=10)
    hit = cache.lookup("m", "what is the capital of France ", max_tokens=10)
    assert hit is not None
    assert hit.tier is CacheTier.SEMANTIC


def test_tiered_can_refuse_the_semantic_tier() -> None:
    """For calls where answering a slightly different question is unacceptable."""
    cache = TieredCache(clock=ManualClock())
    cache.store("m", "what is the capital of france", "Paris", max_tokens=10)
    hit = cache.lookup("m", "what is the capital of France ", max_tokens=10, allow_semantic=False)
    assert hit is None


def test_tiered_misses_when_nothing_is_close() -> None:
    cache = TieredCache(clock=ManualClock())
    cache.store("m", "what is the capital of france", "Paris", max_tokens=10)
    hit = cache.lookup("m", "delete all customer records", max_tokens=10)
    assert hit is None


def test_disabled_semantic_config_neutralises_the_threshold() -> None:
    from keel.config import CacheConfig

    cache = TieredCache(config=CacheConfig(semantic_enabled=False), clock=ManualClock())
    assert cache.semantic.threshold == 1.0
    cache.store("m", "what is the capital of france", "Paris", max_tokens=10)

    assert cache.lookup("m", "tell me the capital city of france", max_tokens=10) is None
    assert cache.lookup("m", "what is the capital of france", max_tokens=10) is not None


def test_tiered_clear_empties_both_tiers() -> None:
    cache = TieredCache(clock=ManualClock())
    cache.store("m", "what is the capital of france", "Paris", max_tokens=10)
    cache.clear()
    assert cache.exact.size == 0
    assert cache.semantic.size == 0
