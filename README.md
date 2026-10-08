# keel

A control plane for LLM inference workloads.

`keel` treats inference as a **scheduling problem**, not an API call. It sits in front of a
model backend and owns the decisions that actually determine cost and latency: how requests
get batched, when new requests are admitted, how KV cache memory is shared and reclaimed,
what a tenant is allowed to spend, and what happens when the backend degrades.

The model backend here is a deterministic simulation of an autoregressive decode loop, so
the whole system runs with no GPU, no API key, and no network. The latency figures are
simulated. The scheduling logic is not.

## Why it exists

A single-request-per-call design is the default most people build against. It is simple and
it is wrong in two ways that only show up under load:

- **Throughput collapses.** Decode is memory-bandwidth bound. Batching amortises the weight
  reads across many sequences, which is the single largest efficiency lever available.
- **Tail latency is unbounded.** Without an admission policy, one long prompt sits in front
  of everyone and TTFT stops being predictable.

`keel` addresses both directly: continuous batching with iteration-level admission, paged KV
cache with prefix sharing and preemption, and a set of interchangeable admission policies
whose behaviour can be measured rather than assumed.

## Status

Early. See the git history for the current state.