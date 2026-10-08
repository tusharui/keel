from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from keel.errors import OutOfBlocks
from keel.ids import new_id
from keel.sim_engine.model import stable_digest


@dataclass(slots=True)
class Block:
    """One unit of KV storage, holding exactly ``block_size`` tokens."""

    id: int
    num_tokens: int = 0
    content_hash: int | None = None
    ref_count: int = 0
    last_access: int = 0


@dataclass(slots=True)
class PoolStats:
    allocated: int = 0
    reused_from_cache: int = 0
    served_from_free: int = 0
    evictions: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0

    @property
    def prefix_hit_rate(self) -> float:
        total = self.prefix_hits + self.prefix_misses
        return self.prefix_hits / total if total else 0.0


def block_content_hash(parent_hash: int | None, tokens: list[int]) -> int:
    """Chain a block's hash to its parent's so identical prefixes collide only
    when every preceding token also matches.

    Hashing tokens alone would let two unrelated sequences share a block
    whenever their *last* block happened to match, which is exactly the bug
    that makes naive prefix caches return another user's cached prompt.
    """
    return stable_digest("kvblock", parent_hash if parent_hash is not None else -1, *tokens)


class BlockPool:
    """Fixed-size pool with two retirement queues.

    A block whose reference count drops to zero is not destroyed, it is parked
    in the cached queue and stays usable as a prefix hit. Only under real memory
    pressure is the least recently used cached block reclaimed. That is what
    makes prefix reuse and eviction separate decisions instead of one.
    """

    __slots__ = ("_blocks", "_cached", "_free", "_on_evict", "_tick", "block_size", "stats")

    def __init__(
        self, num_blocks: int, block_size: int, *, on_evict: Callable[[int], None] | None = None
    ) -> None:
        self.block_size = block_size
        self._blocks = [Block(id=i) for i in range(num_blocks)]
        # Reversed so pop() yields the lowest id first and allocation order stays
        # stable, which keeps benchmark output reproducible.
        self._free = deque(range(num_blocks - 1, -1, -1))
        self._cached: OrderedDict[int, None] = OrderedDict()
        self._tick = 0
        self.stats = PoolStats()
        self._on_evict = on_evict

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_cached(self) -> int:
        return len(self._cached)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free - self.num_cached

    def block(self, block_id: int) -> Block:
        return self._blocks[block_id]

    def allocate(self) -> int:
        self._tick += 1
        self.stats.allocated += 1

        if self._free:
            block_id = self._free.popleft()
            self.stats.served_from_free += 1
        elif self._cached:
            block_id, _ = self._cached.popitem(last=False)
            self.stats.evictions += 1
            self.stats.reused_from_cache += 1
            if self._on_evict is not None:
                self._on_evict(block_id)
        else:
            raise OutOfBlocks(
                f"all {self.num_blocks} blocks are resident; cannot admit more sequences"
            )

        block = self._blocks[block_id]
        block.ref_count += 1
        block.last_access = self._tick
        block.num_tokens = 0
        block.content_hash = None
        return block_id

    def ref(self, block_id: int) -> None:
        block = self._blocks[block_id]
        if block.ref_count == 0:
            self._cached.pop(block_id, None)
        block.ref_count += 1
        self._tick += 1
        block.last_access = self._tick

    def unref(self, block_id: int) -> None:
        block = self._blocks[block_id]
        if block.ref_count == 0:
            raise RuntimeError(f"unref on block {block_id} already at zero refs")
        block.ref_count -= 1
        if block.ref_count == 0:
            self._tick += 1
            block.last_access = self._tick
            self._cached[block_id] = None

    def drop_cached(self) -> int:
        """Release the entire prefix cache. Returns how many blocks were freed."""
        released = 0
        for block_id in list(self._cached):
            self._free.append(block_id)
            if self._on_evict is not None:
                self._on_evict(block_id)
            released += 1
        self._cached.clear()
        return released


@dataclass(slots=True)
class SequenceState:
    """A request's place in the cache.

    ``block_table`` maps logical block index to physical block id. The tail
    block is always private even when the prompt shares a prefix, because more
    tokens could still be appended to it and a shared block cannot be mutated.
    """

    seq_id: str = field(default_factory=new_id)
    tokens: list[int] = field(default_factory=list)
    max_tokens: int = 0
    prompt_len: int = 0
    block_table: list[int] = field(default_factory=list)
    shared_prefix_blocks: int = 0
    prefill_completed: bool = False
    # Prompt tokens already pushed through the model. Less than prompt_len means
    # a chunked prefill is still in flight.
    prefilled: int = 0
    # Sampled from the model but not yet through it. Each decode step pushes it
    # into the KV cache and replaces it, so the cache always trails the token
    # stream by exactly one.
    pending_token: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def num_blocks(self) -> int:
        return len(self.block_table)

    @property
    def output_tokens(self) -> list[int]:
        return self.tokens[self.prompt_len :]

    @property
    def num_output_tokens(self) -> int:
        return len(self.tokens) - self.prompt_len


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self._pool = BlockPool(num_blocks, block_size, on_evict=self._forget_hash)
        self._hash_to_block: dict[int, int] = {}

    @property
    def pool(self) -> BlockPool:
        return self._pool

    @property
    def block_size(self) -> int:
        return self._pool.block_size

    @property
    def num_free_blocks(self) -> int:
        return self._pool.num_free + self._pool.num_cached

    @property
    def num_used_blocks(self) -> int:
        return self._pool.num_used

    @property
    def stats(self) -> PoolStats:
        return self._pool.stats

    def blocks_required(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def _forget_hash(self, block_id: int) -> None:
        block = self._pool.block(block_id)
        if block.content_hash is not None:
            self._hash_to_block.pop(block.content_hash, None)
            block.content_hash = None

    def _reclaim(self, seq: SequenceState) -> None:
        for block_id in seq.block_table:
            self._pool.unref(block_id)
        seq.block_table.clear()

    def _match_prefix(self, tokens: list[int]) -> list[int]:
        """Longest run of already-cached full blocks matching this prompt."""
        size = self.block_size
        matched: list[int] = []
        parent: int | None = None
        offset = 0

        while offset + size <= len(tokens):
            chunk = tokens[offset : offset + size]
            content_hash = block_content_hash(parent, chunk)
            block_id = self._hash_to_block.get(content_hash)
            if block_id is None:
                break
            self._pool.ref(block_id)
            matched.append(block_id)
            parent = content_hash
            offset += size

        # Counted per sequence rather than per block, so the rate answers the
        # question that matters: what fraction of prompts avoided recomputation.
        if matched:
            self._pool.stats.prefix_hits += 1
        else:
            self._pool.stats.prefix_misses += 1
        return matched

    def allocate(self, prompt: list[int], max_tokens: int) -> SequenceState:
        seq = SequenceState(tokens=list(prompt), max_tokens=max_tokens, prompt_len=len(prompt))
        try:
            seq.block_table.extend(self._match_prefix(seq.tokens))
            seq.shared_prefix_blocks = len(seq.block_table)
            for _ in range(self.blocks_required(len(prompt)) - seq.shared_prefix_blocks):
                seq.block_table.append(self._pool.allocate())

            # Publish hashes for the prompt blocks this sequence owns, so the
            # next request sharing this prefix can skip re-prefilling it. Without
            # this the cache only ever learns about generated tokens.
            for index in range(seq.shared_prefix_blocks, seq.num_blocks):
                if (index + 1) * self.block_size <= len(prompt):
                    self._finalize(seq, index)
        except OutOfBlocks:
            self._reclaim(seq)
            raise
        return seq

    def _finalize(self, seq: SequenceState, index: int) -> None:
        """A block just filled. Publish its hash so later sequences can reuse it."""
        size = self.block_size
        start = index * size
        tokens = seq.tokens[start : start + size]
        parent = None
        if index > 0:
            parent = self._pool.block(seq.block_table[index - 1]).content_hash

        content_hash = block_content_hash(parent, tokens)
        existing = self._hash_to_block.get(content_hash)
        block_id = seq.block_table[index]

        if existing is not None and existing != block_id:
            self._pool.unref(block_id)
            self._pool.ref(existing)
            seq.block_table[index] = existing
            block_id = existing

        block = self._pool.block(block_id)
        block.content_hash = content_hash
        block.num_tokens = size
        self._hash_to_block[content_hash] = block_id

    def append(self, seq: SequenceState, token: int) -> None:
        seq.tokens.append(token)
        index = (seq.num_tokens - 1) // self.block_size

        if index == len(seq.block_table):
            seq.block_table.append(self._pool.allocate())

        block_id = seq.block_table[index]
        self._pool.block(block_id).num_tokens += 1
        if self._pool.block(block_id).num_tokens >= self.block_size:
            self._finalize(seq, index)

    def extend(self, seq: SequenceState, tokens: list[int]) -> None:
        for token in tokens:
            self.append(seq, token)

    def release(self, seq: SequenceState) -> None:
        self._reclaim(seq)

    def can_fit(self, prompt: list[int], max_tokens: int) -> bool:
        # Shared prefix blocks are already resident, so they cost no new memory.
        needed = self.blocks_required(len(prompt) + max_tokens) - self.shared_prefix_blocks(prompt)
        return needed <= self.num_free_blocks

    def shared_prefix_blocks(self, tokens: list[int]) -> int:
        """How many leading full blocks are already cached. Read-only."""
        size = self.block_size
        count = 0
        parent: int | None = None
        offset = 0
        while offset + size <= len(tokens):
            content_hash = block_content_hash(parent, tokens[offset : offset + size])
            if content_hash not in self._hash_to_block:
                break
            count += 1
            parent = content_hash
            offset += size
        return count
