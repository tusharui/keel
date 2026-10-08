from __future__ import annotations

import re

PAD = 0
BOS = 1
EOS = 2
UNK = 3
RESERVED = 16

_CHUNK = re.compile(r"\s*\w+|\s+|[^\w\s]")


class Tokenizer:
    """Deterministic greedy tokenizer over word and punctuation chunks.

    Piece ids are assigned in the order chunks are first seen, so tokenisation
    depends on what has been encoded earlier in the process. That is fine here
    and worth being explicit about: the prefix cache in this package hashes
    token ids rather than raw text, and every consumer of it resolves ids
    through one shared tokenizer instance. Swapping in a real BPE would change
    the ids but not the cache logic.

    Leading whitespace is folded into the following word, the way production
    pre-tokenizers do. Attaching it to the previous token instead roughly halves
    the token count on ordinary prose and understates prompt lengths by about
    the same margin.
    """

    __slots__ = ("_id_to_piece", "_piece_to_id", "_vocab_size")

    def __init__(self, vocab_size: int, *, reserved: int = RESERVED) -> None:
        if vocab_size <= reserved:
            raise ValueError(f"vocab_size must exceed {reserved}")
        self._vocab_size = vocab_size
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def size(self) -> int:
        return len(self._piece_to_id)

    def _id_for(self, piece: str) -> int:
        existing = self._piece_to_id.get(piece)
        if existing is not None:
            return existing
        nxt = len(self._piece_to_id) + RESERVED
        if nxt >= self._vocab_size:
            raise ValueError(
                f"tokenizer exhausted at {self._vocab_size} pieces; a real "
                "vocabulary would fall back to UNK or merge instead"
            )
        self._piece_to_id[piece] = nxt
        self._id_to_piece[nxt] = piece
        return nxt

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        ids = [BOS] if add_bos else []
        for chunk in _CHUNK.findall(text):
            ids.append(self._id_for(chunk))
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        out: list[str] = []
        for token in ids:
            if token in (PAD, BOS):
                continue
            if token == EOS:
                if skip_special:
                    continue
                out.append("<eos>")
                continue
            out.append(self._id_to_piece.get(token, ""))
        return "".join(out)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))
