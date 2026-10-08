from __future__ import annotations

import pytest

from keel import ids
from keel.ids import IdGenerator, is_valid, new_id, new_id_int, ulid_timestamp_ms


class _FakeTime:
    def __init__(self, ns: int) -> None:
        self.ns = ns

    def time_ns(self) -> int:
        return self.ns


def test_id_shape() -> None:
    uid = new_id()
    assert len(uid) == 26
    assert is_valid(uid)


def test_ids_sort_by_creation_order() -> None:
    gen = IdGenerator()
    generated = [gen.new() for _ in range(2000)]
    assert generated == sorted(generated)


def test_ids_are_unique_under_a_frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ids, "time", _FakeTime(1_700_000_000_000_000_000))
    gen = IdGenerator()
    generated = [gen.new() for _ in range(1000)]
    assert len(set(generated)) == 1000
    assert generated == sorted(generated), "ordering must stay total within one millisecond"


def test_random_field_carries_into_the_next_millisecond(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeTime(1_700_000_000_000_000_000)
    monkeypatch.setattr(ids, "time", clock)

    gen = IdGenerator()
    first = gen.new()
    clock.ns += 1_000_000
    second = gen.new()

    assert first < second
    assert ulid_timestamp_ms(second) == ulid_timestamp_ms(first) + 1


def test_timestamp_survives_the_round_trip() -> None:
    uid = new_id()
    assert 1_700_000_000_000 < ulid_timestamp_ms(uid) < 4_000_000_000_000


def test_int_form_orders_the_same_as_string_form() -> None:
    gen = IdGenerator()
    strings = [gen.new() for _ in range(500)]
    ints = [gen.new_int() for _ in range(500)]
    assert strings == sorted(strings)
    assert ints == sorted(ints)


def test_int_form_encodes_the_same_timestamp() -> None:
    value = new_id_int()
    assert value >> 80 == ulid_timestamp_ms(value)


@pytest.mark.parametrize("bad", ["", "short", "U" * 26, "0" * 25 + "!"])
def test_malformed_ids_are_rejected(bad: str) -> None:
    assert not is_valid(bad)
    with pytest.raises(ValueError):
        ulid_timestamp_ms(bad)


def test_lowercase_is_accepted_on_read() -> None:
    uid = new_id()
    assert is_valid(uid.lower())
    assert ulid_timestamp_ms(uid.lower()) == ulid_timestamp_ms(uid)
