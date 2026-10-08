from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from keel import __version__


def _project_root() -> Path:
    """Where alembic.ini and the versions directory live.

    Checked against the environment first, because in a container the package is
    installed into site-packages and walking up three levels from cli.py lands
    somewhere unrelated to the project.
    """
    override = os.environ.get("KEEL_PROJECT_ROOT")
    if override:
        return Path(override)
    # parents[2] is the repository root from src/keel/cli.py. Correct for an
    # editable or source checkout, which is how this is normally run.
    return Path(__file__).resolve().parents[2]


def _demo(_args: argparse.Namespace) -> int:
    from keel.clock import ManualClock
    from keel.config import SchedulerConfig
    from keel.scheduler.policies import DeadlinePolicy
    from keel.scheduler.request import InferenceRequest
    from keel.scheduler.scheduler import Scheduler
    from keel.sim_engine import DeviceProfile, SimModel, Tokenizer
    from keel.sim_engine.engine import InferenceEngine
    from keel.sim_engine.kv_cache import KVCacheManager

    clock = ManualClock()
    engine = InferenceEngine(
        tokenizer=Tokenizer(vocab_size=32_000),
        model=SimModel(vocab_size=32_000),
        device=DeviceProfile(),
        kv=KVCacheManager(num_blocks=512, block_size=16),
        context_length=8192,
        clock=clock,
    )
    scheduler = Scheduler(
        engine,
        SchedulerConfig(max_num_seqs=8, policy="deadline"),
        policy=DeadlinePolicy(),
        clock=clock,
    )

    system_prompt = list(range(64))
    print("6 requests sharing a 64-token system prompt; first 3 carry a deadline\n")
    for index in range(6):
        scheduler.submit(
            InferenceRequest(
                request_id=f"req-{index}",
                prompt_tokens=system_prompt + list(range(500 + index * 300, 560 + index * 300)),
                max_tokens=24,
                deadline_ms=3_000.0 if index < 3 else None,
            )
        )

    scheduler.run()

    print(f"{'request':<10} {'state':<10} {'out':>4} {'ttft_ms':>9} {'tpot_ms':>8} {'preempt':>7}")
    print("-" * 54)
    for request in scheduler.results:
        tpot = "n/a" if request.tpot_ms is None else f"{request.tpot_ms:.2f}"
        print(
            f"{request.request_id:<10} {request.state!s:<10} {len(request.output):>4} "
            f"{request.ttft_ms or 0.0:>9.1f} {tpot:>8} {request.preemptions:>7}"
        )

    metrics = scheduler.metrics(ttft_slo_ms=2_000.0, tpot_slo_ms=250.0)
    print()
    for key, value in metrics.as_rows():
        print(f"{key:<22}{value}")

    stats = engine.kv.stats
    print(
        f"\nkv prefix cache: hits={stats.prefix_hits} misses={stats.prefix_misses} "
        f"hit_rate={stats.prefix_hit_rate:.2f}"
    )
    return 0


def _bench(args: argparse.Namespace) -> int:
    from keel.bench.runner import main as bench_main

    argv = ["--requests", str(args.requests), "--seed", str(args.seed)]
    if args.shared_prefix:
        argv += ["--shared-prefix", str(args.shared_prefix)]
    if args.json:
        argv.append("--json")
    return bench_main(argv)


def _db(args: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    root = _project_root()
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    if args.revision == "head":
        command.upgrade(config, "head")
    else:
        command.downgrade(config, args.revision)
    return 0


def _version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="keel", description="LLM inference control plane")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="run a small end-to-end scheduler scenario").set_defaults(
        func=_demo
    )
    sub.add_parser("version", help="print the package version").set_defaults(func=_version)

    bench = sub.add_parser("bench", help="run the scheduler benchmark")
    bench.add_argument("--requests", type=int, default=150)
    bench.add_argument("--shared-prefix", type=int, default=0)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(func=_bench)

    db = sub.add_parser("db", help="run database migrations")
    db.add_argument("revision", nargs="?", default="head")
    db.set_defaults(func=_db)

    args = parser.parse_args(argv)
    func: object = args.func
    return int(func(args))  # type: ignore[operator]


if __name__ == "__main__":
    sys.exit(main())
