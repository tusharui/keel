from __future__ import annotations

import secrets
import threading
import time
from typing import Final

_ALPHABET: Final = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ENCODED_TIME: Final = 10
_ENCODED_RANDOM: Final = 16
_ULID_LEN: Final = _ENCODED_TIME + _ENCODED_RANDOM
_MAX_TIME: Final = (1 << 48) - 1
_MAX_RANDOM: Final = (1 << 80) - 1
_DECODE: Final[dict[str, int]] = {c: i for i, c in enumerate(_ALPHABET)}


def _encode(value: int, length: int) -> str:
    out = [""] * length
    for i in range(length - 1, -1, -1):
        out[i] = _ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(out)


class IdGenerator:
    """ULID source: 48-bit millisecond timestamp followed by 80 random bits.

    Lexicographic order matches creation order, so primary-key inserts land at
    the right-hand edge of the B-tree instead of scattering random writes across
    the whole index. On a high-write table that difference shows up as lock
    contention long before it shows up as raw throughput loss.

    Within a single millisecond the random field is incremented rather than
    redrawn, which keeps ordering total for a process that mints ids faster than
    the clock ticks.
    """

    __slots__ = ("_last_random", "_last_time", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_time = -1
        self._last_random = 0

    def _next(self) -> tuple[int, int]:
        with self._lock:
            now_ms = time.time_ns() // 1_000_000
            if now_ms == self._last_time:
                nxt = self._last_random + 1
                if nxt > _MAX_RANDOM:
                    now_ms += 1
                    nxt = 0
                self._last_time = now_ms
                self._last_random = nxt
                return now_ms, nxt
            self._last_time = now_ms
            self._last_random = secrets.randbits(80)
            return now_ms, self._last_random

    def new(self) -> str:
        ts, rand = self._next()
        return _encode(ts & _MAX_TIME, _ENCODED_TIME) + _encode(rand, _ENCODED_RANDOM)

    def new_int(self) -> int:
        """Packed 128-bit form. Cheaper as a primary key and sorts identically."""
        ts, rand = self._next()
        return (ts << 80) | rand


_generator = IdGenerator()


def new_id() -> str:
    return _generator.new()


def new_id_int() -> int:
    return _generator.new_int()


def ulid_timestamp_ms(value: str | int) -> int:
    """Recover creation time from an id.

    Lets callers range-scan a table by creation time without carrying a separate
    timestamp column on the hot path.
    """
    if isinstance(value, int):
        return value >> 80
    if len(value) != _ULID_LEN:
        raise ValueError(f"expected {_ULID_LEN} character ULID, got {len(value)}")

    upper = value.upper()
    invalid = next((c for c in upper if c not in _DECODE), None)
    if invalid is not None:
        raise ValueError(f"invalid ULID character: {invalid!r}")

    acc = 0
    for char in upper[:_ENCODED_TIME]:
        acc = (acc << 5) | _DECODE[char]
    return acc


def is_valid(value: str) -> bool:
    if len(value) != _ULID_LEN:
        return False
    return all(c.upper() in _DECODE for c in value)
