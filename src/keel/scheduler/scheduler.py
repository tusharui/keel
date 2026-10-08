from __future__ import annotations

from keel.clock import Clock, default_clock
from keel.config import SchedulerConfig
from keel.errors import OutOfBlocks
from keel.scheduler.metrics import ServingMetrics
from keel.scheduler.policies import AdmissionPolicy, FCFSPolicy
from keel.scheduler.request import InferenceRequest, RequestState
from keel.sim_engine.engine import FinishReason, InferenceEngine, find_stop, longest_stop
from keel.sim_engine.kv_cache import SequenceState
from keel.sim_engine.tokenizer import EOS


def _seq(request: InferenceRequest) -> SequenceState:
    """Narrow the optional sequence, loudly if the invariant is broken.

    An assert would disappear under ``python -O`` and turn a scheduler bug into a
    silently malformed batch.
    """
    seq = request.seq
    if seq is None:
        raise RuntimeError(f"request {request.request_id} has no live sequence")
    return seq


class Scheduler:
    """Continuous batching with iteration-level admission.

    A static batch is decided once and executed to completion. Here the batch is
    reassembled every iteration: finished sequences leave and waiting requests
    take their slots immediately. That is the whole difference, and it is why a
    request arriving one millisecond into a long generation waits one iteration
    rather than one entire generation.
    """

    __slots__ = (
        "_batch_slots",
        "_clock",
        "_config",
        "_decode_iterations",
        "_decoding",
        "_engine",
        "_finished",
        "_inbox",
        "_iterations",
        "_policy",
        "_preemptions",
        "_prefill",
        "_tokenizer",
        "_waiting",
    )

    def __init__(
        self,
        engine: InferenceEngine,
        config: SchedulerConfig,
        *,
        policy: AdmissionPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._engine = engine
        self._config = config
        self._policy = policy or FCFSPolicy()
        self._clock = clock or default_clock()
        self._tokenizer = engine.tokenizer
        self._inbox: list[InferenceRequest] = []
        self._waiting: list[InferenceRequest] = []
        self._prefill: list[InferenceRequest] = []
        self._decoding: list[InferenceRequest] = []
        self._finished: list[InferenceRequest] = []
        self._iterations = 0
        self._preemptions = 0
        self._batch_slots = 0
        self._decode_iterations = 0

    # --- properties ---------------------------------------------------------------

    @property
    def policy(self) -> AdmissionPolicy:
        return self._policy

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def preemptions(self) -> int:
        return self._preemptions

    @property
    def mean_batch_size(self) -> float:
        """Average concurrent sequences across decode iterations.

        Measured over decoding iterations only. Counting iterations where the
        batch was empty would drag the average down without saying anything
        about how full the device actually was.
        """
        if self._decode_iterations == 0:
            return 0.0
        return self._batch_slots / self._decode_iterations

    @property
    def results(self) -> list[InferenceRequest]:
        return list(self._finished)

    @property
    def running(self) -> list[InferenceRequest]:
        return list(self._decoding)

    @property
    def waiting(self) -> list[InferenceRequest]:
        return list(self._waiting)

    @property
    def queue_depth(self) -> int:
        return len(self._inbox) + len(self._waiting) + len(self._prefill)

    @property
    def batch_size(self) -> int:
        return len(self._decoding)

    def has_work(self) -> bool:
        return bool(self._inbox or self._waiting or self._prefill or self._decoding)

    def metrics(
        self,
        *,
        ttft_slo_ms: float | None = None,
        tpot_slo_ms: float | None = None,
    ) -> ServingMetrics:
        return ServingMetrics.from_results(
            self.results,
            makespan_ms=self.makespan_ms,
            mean_batch_size=self.mean_batch_size,
            preemptions=self._preemptions,
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
        )

    @property
    def makespan_ms(self) -> float:
        if not self._finished:
            return 0.0
        return max(r.finished_at_ms or 0.0 for r in self._finished)

    # --- lifecycle ----------------------------------------------------------------

    def submit(self, request: InferenceRequest) -> None:
        """Queue a request. It becomes eligible once simulated time reaches its
        arrival time, so an arrival stream can be modelled without interleaving
        submissions into the loop."""
        self._inbox.append(request)

    def _drain_inbox(self) -> None:
        if not self._inbox:
            return
        now_ms = self._clock.now() * 1000.0
        arrived = [r for r in self._inbox if r.arrival_ms <= now_ms]
        if not arrived:
            return
        for request in arrived:
            self._inbox.remove(request)
            request.enqueued_at_ms = now_ms
            self._waiting.append(request)

    # --- admission ----------------------------------------------------------------

    def _slots(self) -> int:
        return self._config.max_num_seqs - len(self._prefill) - len(self._decoding)

    def _admit_one(self, request: InferenceRequest) -> bool:
        try:
            request.seq = self._engine.kv.allocate(request.prompt_tokens, request.max_tokens)
        except OutOfBlocks:
            return False
        request.state = RequestState.RUNNING
        request.started_at_ms = self._clock.now() * 1000.0
        self._prefill.append(request)
        self._waiting.remove(request)
        return True

    def _preempt(self, required_blocks: int) -> bool:
        """Free memory by evicting the sequence that has done the least work.

        The victim is the one whose recompute is smallest, not the one resident
        longest. Eviction loses every token generated so far because blocks go
        back to the pool rather than being swapped to host memory, so the cost of
        preemption is measured in regenerated tokens.
        """
        if not self._config.enable_preemption or not self._decoding:
            return False

        victim = min(self._decoding, key=lambda r: (len(r.output), r.started_at_ms or 0.0))
        victim_seq = victim.seq
        if victim_seq is not None and victim_seq.num_blocks <= required_blocks:
            # Evicting a sequence no larger than what the newcomer needs buys no
            # net room, so the two would simply trade places: the newcomer still
            # does not fit, the incumbent is evicted, and the next pass reverses
            # it. Preemption only earns its recompute cost when the incoming
            # request is genuinely smaller than what it displaces.
            return False

        if victim_seq is not None:
            self._engine.kv.release(victim_seq)

        victim.seq = None
        victim.output.clear()
        victim.text = ""
        victim.started_at_ms = None
        victim.first_token_at_ms = None
        victim.finish_reason = None
        victim.preemptions += 1
        victim.state = RequestState.PREEMPTED
        # Re-enter the queue with a fresh arrival stamp. Keeping the original one
        # would put it straight back at the head under FCFS.
        victim.enqueued_at_ms = self._clock.now() * 1000.0

        self._decoding.remove(victim)
        self._waiting.append(victim)
        self._preemptions += 1
        return True

    def _span_blocks(self, request: InferenceRequest) -> int:
        """Peak blocks a request occupies over its whole lifetime."""
        return self._engine.kv.blocks_required(request.num_prompt_tokens + request.max_tokens)

    def _incremental_blocks(self, request: InferenceRequest) -> int:
        """New blocks this request would actually take on admission.

        Prefix blocks already in the cache are resident, so they cost nothing
        further. Only meaningful for a request that has not been admitted yet:
        once resident, a sequence's own blocks appear in the hash map and the
        subtraction would start discounting memory it is already holding.
        """
        return self._span_blocks(request) - self._engine.kv.shared_prefix_blocks(
            request.prompt_tokens
        )

    def _committed_blocks(self) -> int:
        """Peak capacity already promised to admitted requests.

        Counted per request rather than by summing current block tables, because
        a sequence allocates as it generates and would otherwise look cheaper
        than it is going to be. Shared prefixes between admitted requests are
        counted more than once here, which over-reserves rather than
        under-reserves, and under-reserving is what starves a decode step.
        """
        held = [r for r in (*self._prefill, *self._decoding) if r.seq is not None]
        return sum(self._span_blocks(r) for r in held)

    def _can_reserve(self, request: InferenceRequest) -> bool:
        return self._committed_blocks() + self._incremental_blocks(request) <= (
            self._engine.kv.pool.num_blocks
        )

    def _admit_first_that_fits(self, ordered: list[InferenceRequest]) -> bool:
        """Bypass requests that cannot fit rather than stalling behind them.

        Without this a single oversized request parks at the head of the queue
        and every smaller request behind it starves, even though the memory to
        run them is sitting free.
        """
        for request in ordered:
            if self._slots() <= 0:
                return False
            if self._can_reserve(request):
                return self._admit_one(request)
        return False

    def _admit(self) -> int:
        if not self._waiting or self._slots() <= 0:
            return 0

        admitted = 0
        while self._waiting and self._slots() > 0:
            ordered = self._policy.order(self._waiting)
            if self._admit_first_that_fits(ordered):
                admitted += 1
                continue
            if not self._make_room():
                break
            admitted += 1

        return admitted

    def _make_room(self) -> bool:
        """Preempt until some waiting request fits. Gives up rather than livelock.

        Candidates are tried in policy order, not shortest-first. Preemption is
        only worth its recompute for a request smaller than its victim, and the
        skip on failure already handles the ones it cannot help, so the queue
        keeps its configured ordering.
        """
        candidates = self._policy.order(self._waiting)
        pool_size = self._engine.kv.pool.num_blocks

        if all(self._span_blocks(r) > pool_size for r in candidates):
            offender = max(candidates, key=self._span_blocks)
            raise OutOfBlocks(
                f"request {offender.request_id} needs {self._span_blocks(offender)} "
                f"blocks but the pool holds {pool_size}"
            )

        for candidate in candidates:
            needed = self._incremental_blocks(candidate)
            if needed > pool_size:
                continue
            while not self._can_reserve(candidate):
                if not self._preempt(needed):
                    break
            if self._can_reserve(candidate):
                return self._admit_one(candidate)
        return False

    # --- prefill ------------------------------------------------------------------

    def _run_prefill(self) -> bool:
        if not self._prefill:
            return False

        budget = self._config.max_num_batched_tokens
        did_work = False

        while self._prefill and budget > 0:
            request = self._prefill[0]
            seq = _seq(request)

            remaining = seq.prompt_len - seq.prefilled
            chunk = min(remaining, self._config.chunked_prefill_tokens, budget)
            self._engine.prefill(seq, chunk)
            seq.prefilled += chunk
            budget -= chunk
            did_work = True

            if seq.prefilled >= seq.prompt_len:
                self._prefill.remove(request)
                self._begin_decode(request)

        return did_work

    def _begin_decode(self, request: InferenceRequest) -> None:
        """First token falls out of the prefill pass, which is where TTFT ends."""
        seq = _seq(request)
        seq.prefill_completed = True
        seq.pending_token = self._engine.sample(seq)
        request.output.append(seq.pending_token)
        request.first_token_at_ms = self._clock.now() * 1000.0
        self._decoding.append(request)

    # --- decode -------------------------------------------------------------------

    def _run_decode(self) -> bool:
        if not self._decoding:
            return False

        self._decode_iterations += 1
        self._batch_slots += len(self._decoding)
        self._engine.decode_step([_seq(r) for r in self._decoding])

        completed: list[tuple[InferenceRequest, FinishReason]] = []
        for request in self._decoding:
            seq = _seq(request)
            token = seq.pending_token

            previous_len = len(request.text)
            request.output.append(token)
            request.text += self._tokenizer.decode([token])

            if token == EOS:
                completed.append((request, FinishReason.STOP))
            elif len(request.output) >= request.max_tokens:
                completed.append((request, FinishReason.LENGTH))
            elif request.stop and find_stop(
                request.text, request.stop, max(0, previous_len - longest_stop(request.stop) + 1)
            ):
                completed.append((request, FinishReason.STOP))

        for request, reason in completed:
            self._finish(request, reason)
        return True

    def _finish(self, request: InferenceRequest, reason: FinishReason) -> None:
        request.state = RequestState.FINISHED
        request.finish_reason = reason
        request.finished_at_ms = self._clock.now() * 1000.0
        if request.seq is not None:
            self._engine.kv.release(request.seq)
            request.seq = None
        self._decoding.remove(request)
        self._finished.append(request)

    # --- main loop ----------------------------------------------------------------

    def step(self) -> bool:
        self._iterations += 1
        self._drain_inbox()
        self._admit()
        self._run_prefill()
        self._run_decode()
        return self.has_work()

    def run(self, *, max_iterations: int = 5_000_000) -> None:
        guard = 0
        while self.has_work():
            self.step()
            guard += 1
            if guard > max_iterations:
                raise RuntimeError(
                    f"scheduler did not converge after {max_iterations} iterations; "
                    f"waiting={len(self._waiting)} prefill={len(self._prefill)} "
                    f"decoding={len(self._decoding)}"
                )
