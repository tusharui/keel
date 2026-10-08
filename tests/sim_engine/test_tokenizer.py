from __future__ import annotations

import pytest

from keel.sim_engine import BOS, EOS, PAD, Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(vocab_size=1024)


def test_encode_decode_round_trips(tokenizer: Tokenizer) -> None:
    text = "the quick brown fox jumps"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bos_is_prepended(tokenizer: Tokenizer) -> None:
    assert tokenizer.encode("hi")[0] == BOS
    assert tokenizer.encode("hi", add_bos=False)[0] != BOS


def test_tokens_are_stable_for_repeated_text(tokenizer: Tokenizer) -> None:
    first = tokenizer.encode("hello world")
    tokenizer.encode("completely different")
    assert tokenizer.encode("hello world") == first


def test_eos_is_skipped_on_decode(tokenizer: Tokenizer) -> None:
    ids = [*tokenizer.encode("hi"), EOS]
    assert tokenizer.decode(ids) == "hi"
    assert tokenizer.decode(ids, skip_special=False).endswith("<eos>")


def test_padding_is_dropped_on_decode(tokenizer: Tokenizer) -> None:
    assert tokenizer.decode([PAD, *tokenizer.encode("ok")]) == "ok"


def test_whitespace_is_preserved(tokenizer: Tokenizer) -> None:
    text = "a  b\tc"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_punctuation_splits_from_words(tokenizer: Tokenizer) -> None:
    assert len(tokenizer.encode("hi, ok!")) >= 4


def test_unknown_ids_decode_to_nothing(tokenizer: Tokenizer) -> None:
    assert tokenizer.decode([999_999]) == ""


def test_roughly_four_characters_per_token(tokenizer: Tokenizer) -> None:
    text = "the quick brown fox jumps over the lazy dog " * 4
    ratio = len(text) / tokenizer.count_tokens(text)
    assert 3.0 < ratio < 6.0


def test_exhausted_vocabulary_fails_loudly() -> None:
    tiny = Tokenizer(vocab_size=24)
    with pytest.raises(ValueError, match="exhausted"):
        tiny.encode(" ".join(f"word{i}" for i in range(40)))


def test_vocab_size_must_exceed_reserved() -> None:
    with pytest.raises(ValueError, match="exceed"):
        Tokenizer(vocab_size=4)
