from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from keel.ids import new_id
from keel.sim_engine.engine import FinishReason
from keel.sim_engine.kv_cache import SequenceState


class RequestState(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    # Evicted from the running batch to free KV blocks. Distinct from FAILED
    # because nothing was lost; the request goes back on the waiting queue.
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class InferenceRequest:
    request_id: str = field(default_factory=new_id)
    prompt_tokens: list[int] = field(default_factory=list)
    max_tokens: int = 128
    stop: tuple[str, ...] = ()
    tenant_id: str = "default"
    arrival_ms: float = 0.0
    deadline_ms: float | None = None

    state: RequestState = RequestState.WAITING
    seq: SequenceState | None = None
    output: list[int] = field(default_factory=list)
    # Accumulated incrementally. Stop-sequence matching needs the rendered text,
    # and re-decoding the whole output every step turns that into quadratic work
    # on exactly the long generations that matter.
    text: str = ""
    finish_reason: FinishReason | None = None

    enqueued_at_ms: float = 0.0
    started_at_ms: float | None = None
    first_token_at_ms: float | None = None
    finished_at_ms: float | None = None
    preemptions: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_tokens)

    @property
    def estimated_output_tokens(self) -> int:
        """What the admission policy uses to rank this request.

        Bounded by max_tokens because that is the only signal available before
        the first token exists. A production scheduler would use a predictor
        trained on per-tenant length histograms; ranking on max_tokens still
        distinguishes a one-line classification from a 2000-token summary, which
        is most of the signal.
        """
        return self.max_tokens

    @property
    def queue_wait_ms(self) -> float | None:
        if self.started_at_ms is None:
            return None
        return self.started_at_ms - self.enqueued_at_ms

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at_ms is None or self.started_at_ms is None:
            return None
        return self.first_token_at_ms - self.started_at_ms

    @property
    def tpot_ms(self) -> float | None:
        if self.first_token_at_ms is None or self.finished_at_ms is None:
            return None
        decode_window = self.finished_at_ms - self.first_token_at_ms
        if len(self.output) < 2:
            return None
        return decode_window / (len(self.output) - 1)

    @property
    def latency_ms(self) -> float | None:
        if self.finished_at_ms is None:
            return None
        return self.finished_at_ms - self.arrival_ms

    @property
    def is_terminal(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.CANCELLED)
