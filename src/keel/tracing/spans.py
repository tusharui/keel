from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from keel.clock import Clock, default_clock
from keel.ids import new_id


@dataclass(slots=True)
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ms: float
    duration_ms: float
    attributes: dict[str, object] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return 0 if self.parent_id is None else 1


_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "keel_current_span", default=None
)


@dataclass(slots=True)
class Tracer:
    """Records a span tree.

    The active span lives in a ContextVar rather than an instance attribute.
    Spans are created inside concurrently executing tasks, and a plain stack
    would interleave: whichever task last entered would become the parent of
    whatever started next, producing a tree that looks plausible and describes
    nothing that happened.
    """

    clock: Clock = field(default_factory=default_clock)
    trace_id: str = field(default_factory=new_id)
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        parent = _current.get()
        started = self.clock.now()
        record = Span(
            name=name,
            trace_id=self.trace_id,
            span_id=new_id(),
            parent_id=parent,
            start_ms=started * 1000.0,
            duration_ms=0.0,
            attributes=dict(attributes),
        )
        self.spans.append(record)
        token = _current.set(record.span_id)
        try:
            yield record
        finally:
            _current.reset(token)
            record.duration_ms = (self.clock.now() - started) * 1000.0

    def children_of(self, span_id: str | None) -> list[Span]:
        return [s for s in self.spans if s.parent_id == span_id]

    def roots(self) -> list[Span]:
        return [s for s in self.spans if s.parent_id is None]

    def find(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]

    def total_ms(self, name: str) -> float:
        return sum(s.duration_ms for s in self.spans if s.name == name)

    def waterfall(self) -> list[tuple[int, str, float]]:
        """Flattened tree as (depth, name, duration_ms) in start order."""
        children: dict[str | None, list[Span]] = {}
        for span in self.spans:
            children.setdefault(span.parent_id, []).append(span)
        for group in children.values():
            group.sort(key=lambda s: s.start_ms)

        out: list[tuple[int, str, float]] = []

        def walk(span: Span, depth: int) -> None:
            out.append((depth, span.name, span.duration_ms))
            for child in children.get(span.span_id, []):
                walk(child, depth + 1)

        for root in children.get(None, []):
            walk(root, 0)
        return out

    def reset(self) -> None:
        self.spans.clear()
        self.trace_id = new_id()

    def as_dicts(self) -> list[dict[str, object]]:
        return [
            {
                "name": s.name,
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "start_ms": s.start_ms,
                "duration_ms": s.duration_ms,
                "attributes": s.attributes,
            }
            for s in self.spans
        ]


def current_span_id() -> str | None:
    return _current.get()
