from __future__ import annotations

import random
from dataclasses import dataclass

from keel.scheduler.request import InferenceRequest


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    num_requests: int = 200
    prompt_min: int = 64
    prompt_max: int = 512
    completion_min: int = 32
    completion_max: int = 256
    arrival_rate_per_s: float = 20.0
    shared_prefix_tokens: int = 0
    deadline_ms: float | None = 5_000.0
    seed: int = 7


def generate(spec: WorkloadSpec) -> list[InferenceRequest]:
    """Deterministic workload.

    Seeded so two runs of the same scenario are comparable and a regression
    traces to a change in the scheduler rather than to a different draw.
    """
    rng = random.Random(spec.seed)
    shared = list(range(spec.shared_prefix_tokens))
    requests: list[InferenceRequest] = []

    for index in range(spec.num_requests):
        prompt_len = rng.randint(spec.prompt_min, spec.prompt_max)
        # Offset past the shared prefix so distinct requests do not accidentally
        # collide on content and share more than the intended prefix.
        body = list(range(10_000 + index * 4_000, 10_000 + index * 4_000 + prompt_len))
        arrival_ms = (index / spec.arrival_rate_per_s) * 1000.0
        tenant = f"t{index % 4}"
        requests.append(
            InferenceRequest(
                prompt_tokens=shared + body,
                max_tokens=rng.randint(spec.completion_min, spec.completion_max),
                arrival_ms=arrival_ms,
                # Only two of the four tenants carry an objective. Leaving the
                # rest undated is what makes deadline ordering distinguishable
                # from first-come-first-served.
                deadline_ms=None
                if spec.deadline_ms is None or tenant not in ("t0", "t1")
                else arrival_ms + spec.deadline_ms,
                tenant_id=tenant,
            )
        )
    return requests
