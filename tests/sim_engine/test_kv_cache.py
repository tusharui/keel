from __future__ import annotations

import pytest

from keel.errors import OutOfBlocks
from keel.sim_engine.kv_cache import BlockPool, KVCacheManager, block_content_hash


@pytest.fixture
def kv() -> KVCacheManager:
    return KVCacheManager(num_blocks=32, block_size=4)


# --- block hashing ----------------------------------------------------------------


def test_identical_content_hashes_identically() -> None:
    assert block_content_hash(None, [1, 2, 3]) == block_content_hash(None, [1, 2, 3])


def test_parent_hash_changes_the_result() -> None:
    assert block_content_hash(7, [1, 2, 3]) != block_content_hash(8, [1, 2, 3])


def test_same_tail_under_different_parents_does_not_collide() -> None:
    """The bug that makes a naive prefix cache hand back the wrong prompt."""
    left = block_content_hash(1, [9, 9, 9, 9])
    right = block_content_hash(2, [9, 9, 9, 9])
    assert left != right


# --- block pool -------------------------------------------------------------------


def test_pool_allocates_every_block() -> None:
    pool = BlockPool(num_blocks=3, block_size=4)
    for _ in range(3):
        pool.allocate()
    assert pool.num_free == 0
    with pytest.raises(OutOfBlocks):
        pool.allocate()


def test_unref_parks_a_block_as_cached_rather_than_freeing_it() -> None:
    pool = BlockPool(num_blocks=2, block_size=4)
    block_id = pool.allocate()
    pool.unref(block_id)
    assert pool.num_free == 1
    assert pool.num_cached == 1
    assert pool.num_used == 0


def test_allocation_reuses_cached_before_evicting() -> None:
    pool = BlockPool(num_blocks=2, block_size=4)
    first = pool.allocate()
    pool.allocate()
    pool.unref(first)

    assert pool.num_free == 0
    assert pool.allocate() == first
    assert pool.num_used == 2
    assert pool.num_cached == 0


def test_lru_cached_block_is_the_one_evicted() -> None:
    evicted: list[int] = []
    pool = BlockPool(num_blocks=2, block_size=4, on_evict=evicted.append)
    first = pool.allocate()
    second = pool.allocate()
    pool.unref(first)
    pool.unref(second)

    pool.allocate()
    assert evicted == [first], "the earlier unref makes the block least recently used"


def test_drop_cached_reclaims_everything() -> None:
    pool = BlockPool(num_blocks=4, block_size=4)
    ids = [pool.allocate() for _ in range(4)]
    for block_id in ids:
        pool.unref(block_id)
    assert pool.drop_cached() == 4
    assert pool.num_free == 4
    assert pool.num_cached == 0


def test_unref_below_zero_is_a_bug_not_a_state() -> None:
    pool = BlockPool(num_blocks=2, block_size=4)
    block_id = pool.allocate()
    pool.unref(block_id)
    with pytest.raises(RuntimeError, match="zero refs"):
        pool.unref(block_id)


# --- sequence allocation ----------------------------------------------------------


def test_allocate_reserves_enough_blocks(kv: KVCacheManager) -> None:
    seq = kv.allocate([1, 2, 3, 4, 5, 6, 7, 8, 9], max_tokens=3)
    assert seq.num_blocks == 3
    assert kv.num_used_blocks == 3


def test_release_returns_blocks_to_the_cache(kv: KVCacheManager) -> None:
    seq = kv.allocate(list(range(8)), max_tokens=4)
    kv.release(seq)
    assert kv.num_used_blocks == 0
    assert kv.num_free_blocks == 32


def test_failed_allocation_does_not_leak(kv: KVCacheManager) -> None:
    tiny = KVCacheManager(num_blocks=2, block_size=4)
    with pytest.raises(OutOfBlocks):
        tiny.allocate(list(range(100)), max_tokens=0)
    assert tiny.num_used_blocks == 0


# --- prefix sharing ---------------------------------------------------------------


def _fill_and_release(kv: KVCacheManager, prompt: list[int], extra: list[int]) -> None:
    seq = kv.allocate(prompt, max_tokens=len(extra))
    kv.extend(seq, extra)
    kv.release(seq)


def test_shared_prefix_is_reused(kv: KVCacheManager) -> None:
    prompt = list(range(8))
    _fill_and_release(kv, prompt, [10, 11, 12, 13])

    allocations_before = kv.stats.allocated
    second = kv.allocate(prompt, max_tokens=4)
    assert second.shared_prefix_blocks == 2
    assert second.num_blocks == 2
    assert kv.stats.allocated == allocations_before, "reuse must not take new memory"
    kv.release(second)


def test_different_prompt_shares_nothing(kv: KVCacheManager) -> None:
    _fill_and_release(kv, list(range(8)), [1, 2, 3, 4])
    second = kv.allocate([9, 9, 9, 9, 9, 9, 9, 9], max_tokens=4)
    assert second.shared_prefix_blocks == 0
    kv.release(second)


def test_partially_matching_prefix_shares_only_the_common_part(kv: KVCacheManager) -> None:
    _fill_and_release(kv, list(range(8)), [1, 2, 3, 4])
    second = kv.allocate([*range(8), 99], max_tokens=4)
    assert second.shared_prefix_blocks == 2
    kv.release(second)


def test_tail_block_is_private_even_when_the_prefix_is_shared(kv: KVCacheManager) -> None:
    """Sharing a partial block would let one sequence's append corrupt another's."""
    prompt = list(range(6))  # 4 shared + 2 in a partial tail
    _fill_and_release(kv, prompt, [])
    first = kv.allocate(prompt, max_tokens=4)
    kv.append(first, 77)
    kv.append(first, 78)
    kv.append(first, 79)
    kv.append(first, 80)

    second = kv.allocate(prompt, max_tokens=4)
    assert second.shared_prefix_blocks == 1
    assert second.block_table[1] != first.block_table[1]
    kv.release(first)
    kv.release(second)


def test_shared_block_survives_one_of_two_holders_being_released(kv: KVCacheManager) -> None:
    prompt = list(range(8))
    _fill_and_release(kv, prompt, [1, 2, 3, 4])

    first = kv.allocate(prompt, max_tokens=4)
    second = kv.allocate(prompt, max_tokens=4)
    shared_id = first.block_table[0]
    assert second.block_table[0] == shared_id

    kv.release(first)
    assert kv.pool.block(shared_id).ref_count == 1, "still held by the second sequence"

    kv.release(second)
    assert kv.pool.block(shared_id).ref_count == 0


def test_evicted_block_is_forgotten_so_stale_hashes_are_not_reused(
    kv: KVCacheManager,
) -> None:
    prompt = list(range(8))
    _fill_and_release(kv, prompt, [1, 2, 3, 4])
    cached_before = kv.pool.num_cached
    assert cached_before > 0

    kv.pool.drop_cached()
    fresh = kv.allocate(prompt, max_tokens=4)
    assert fresh.shared_prefix_blocks == 0
    kv.release(fresh)


def test_prefix_hit_rate_is_counted_per_sequence(kv: KVCacheManager) -> None:
    prompt = list(range(8))
    _fill_and_release(kv, prompt, [1, 2, 3, 4])

    hit = kv.allocate(prompt, max_tokens=4)
    kv.release(hit)
    miss = kv.allocate([7] * 8, max_tokens=4)
    kv.release(miss)

    # One of the three allocates reused a cached prefix.
    assert kv.stats.prefix_hits == 1
    assert kv.stats.prefix_misses == 2
    assert kv.stats.prefix_hit_rate == pytest.approx(1 / 3)


def test_append_across_a_block_boundary(kv: KVCacheManager) -> None:
    seq = kv.allocate([1, 2, 3, 4], max_tokens=5)
    for token in [10, 11, 12, 13]:
        kv.append(seq, token)
    assert seq.num_blocks == 2
    assert kv.pool.block(seq.block_table[0]).content_hash is not None


def test_can_fit_predicts_admission(kv: KVCacheManager) -> None:
    assert kv.can_fit(list(range(8)), max_tokens=8)
    assert not kv.can_fit(list(range(8)), max_tokens=10_000)


def test_can_fit_ignores_memory_already_shared(kv: KVCacheManager) -> None:
    prompt = list(range(64))
    _fill_and_release(kv, prompt, [])
    assert kv.can_fit(prompt, max_tokens=4)
