# keel

An LLM inference control plane you can actually run: continuous batching, KV paging, preemption,
quota enforcement. The benchmark shows where maximising tokens/sec and meeting a latency SLO come
apart. No GPU or API key needed.

`keel` treats inference as a **scheduling problem**, not an API call. It sits in front of a
model backend and owns the decisions that actually determine cost and latency: how requests
get batched, when new ones are admitted, how KV cache memory is shared and reclaimed, what a
tenant may spend, and what happens when the backend degrades.

The model backend here is a deterministic simulation of an autoregressive decode loop, so the
whole system runs with no GPU, no API key and no network. The latency figures are simulated.
The scheduling logic is not.

## Why it exists

One-request-per-call is the default most people build against. It is simple, and it is wrong
in two ways that only surface under load:

- **Throughput collapses.** Decode is memory-bandwidth bound. Batching amortises the weight
  reads across many sequences, which is the single largest efficiency lever available.
- **Tail latency is unbounded.** Without an admission policy one long prompt sits in front of
  everyone, and time-to-first-token stops being predictable.

`keel` addresses both directly: continuous batching with iteration-level admission, a paged KV
cache with prefix sharing and preemption, and a set of interchangeable admission policies whose
behaviour is measured rather than assumed.

## Measured behaviour

150 requests, mixed prompt and completion lengths, seeded workload, simulated A100-80GB.
Reproduce with `keel bench --requests 150`.

```
scenario                  makespan  ttft p50  ttft p95  tpot p95   out tok/s  goodput  batch
sequential                  250.29     137.5     218.1     92.32        13.4      13.4   1.00
batched max_num_seqs=8       53.56     152.3     319.7    144.95        67.4      67.3   6.95
batched max_num_seqs=32      31.41     211.5    1666.7    383.82       115.0      82.0  19.66
batched max_num_seqs=128     26.86    3678.9    8034.8   2863.12       134.4       6.3  31.46
policy sjf                   30.30     211.9    1662.1    400.68       119.2      69.6  21.63
policy deadline              28.72     211.5    1538.4    354.53       125.7      76.7  25.26
kv=256 preemption on        207.35     254.3    1155.2    313.97        17.4      16.7   9.92
kv=256 preemption off        49.09     158.8     369.1    159.74        73.6      73.4   7.99
```

Three things worth reading twice:

**Throughput and latency trade off almost exactly.** Going from one sequence to 32 buys 8.6x
throughput and costs 7.6x TTFT. That is the deal, stated numerically.

**The best throughput number in the table is the worst outcome.** At `max_num_seqs=128` the
engine pushes the most tokens per second, 134.4, and goodput collapses to 6.3 because TTFT
blows past the 2s objective. Maximising tokens per second is the wrong objective once a latency
target exists, and this table is where that stops being an opinion.

**Preemption makes things worse here.** Under memory pressure, evicting to make room cost 1138
recomputations and a 4x worse makespan than leaving it off. Preemption is a last resort, and
the project's own benchmark says so.

## Layout

```
src/keel/
  clock.py         injectable clock; ManualClock is virtual time
  errors.py        error taxonomy, retryability declared per class
  config.py        env settings with cross-field validation
  sim_engine/      tokenizer, deterministic model, roofline device, paged KV cache, decode loop
  scheduler/       continuous batching, admission policies, preemption, serving metrics
  cache/           exact-match and semantic response tiers
  reliability/     retry with jitter strategies, circuit breaker, provider fallback
  policy/          token bucket limiting, reserve-then-settle budgets
  dag/             graph validation, bounded-parallel executor, resumable state, backfill
  eval/            prompt regression harness and gate
  tracing/         span tree
  db/              typed SQLAlchemy models over SQLite or Postgres
  api/             FastAPI control plane
  bench/           workload generator and benchmark
  cli.py           keel demo | bench | db | version
```

## Running it

### With Docker

Nothing to install but Docker. The image pins the Python version and its Debian
variant, so it does not drift between machines.

```bash
docker compose up --build
open http://localhost:8000/docs
```

That is the whole setup. Migrations run before the server starts, so a fresh
stack comes up with a usable database. Data is ephemeral — fine for a demo, and
it means there is no stale state to explain.

If 8000 is already taken, which it usually is:

```bash
KEEL_PORT=8010 docker compose up --build     # macOS / Linux
$env:KEEL_PORT="8010"; docker compose up --build   # PowerShell
```

For Postgres, which is the point of running it at all: SQLite accepts schemas
Postgres rejects and vice versa, so the default profile cannot prove the models
are right on both.

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up --build
```

The api waits on Postgres's own readiness query rather than a port check, because
Postgres accepts connections slightly before it can authenticate, and that race
only shows up on a cold container.

### Without Docker

Requires Python 3.12+. SQLite by default, so there is nothing else to install.

```bash
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"

./.venv/bin/keel demo                  # smallest end-to-end scenario
./.venv/bin/keel bench --requests 150  # the table above
./.venv/bin/keel db head               # create the schema
./.venv/bin/uvicorn keel.api.app:app   # http control plane
```

`keel demo` prints per-request latency and the KV prefix-cache hit rate for six
requests sharing a system prompt.

Quality gates, all of which CI enforces on every push:

```bash
./.venv/bin/pytest -q
./.venv/bin/ruff check src tests
./.venv/bin/ruff format --check src tests
./.venv/bin/mypy src
```

## API

```
GET  /health              device, pool size, free blocks
POST /v1/completions      prompt, max_tokens, stop, tenant_id
GET  /v1/metrics          token and cost counters, TTFT percentiles, cache hit rate
POST /v1/runs             run a registered DAG over a partition
GET  /v1/traces           span tree for completed requests
```

Rate limiting returns 429 with `Retry-After`; budget exhaustion returns 402; the two need
different responses from a client, so they are not the same status.

## Limitations

Stated plainly, because they are the first questions worth asking:

- **Latency is simulated, not measured.** `DeviceProfile` is a roofline model with a fixed
  per-forward-pass overhead. It is not calibrated against a real trace, so the absolute numbers
  mean "shape of the tradeoff", not "what an A100 would do".
- **The semantic cache threshold is fitted to a hashed-trigram embedder, not a real one.** A
  sentence embedder would separate paraphrases this one cannot see. The tests pin both ends of
  the band so the assumption stays visible.
- **Only one device profile.** No multi-GPU, no tensor parallelism, no disaggregated prefill.
- **The token bucket is in-process.** Two replicas each get the full rate.
- **The semantic cache scan is linear.** Fine for the hundreds of entries it is sized for, not
  for tens of thousands; a real deployment swaps in an ANN index.