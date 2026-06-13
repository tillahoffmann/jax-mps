#!/usr/bin/env python3
"""Interleaved A/B benchmark for JAX_MPS_ASYNC_DISPATCH.

The plugin normally blocks (mlx::core::eval) at the end of every executable
call, so the CPU sits idle during GPU compute and vice versa. With
JAX_MPS_ASYNC_DISPATCH=1 the final materialization uses mlx::core::async_eval,
letting Execute() return before the GPU finishes so jax can dispatch the next
step while the current one runs.

That win only shows up when many steps are dispatched *without* a host sync
between them. So each workload chains K jitted steps on-device (output feeds the
next input) and syncs exactly once at the end via a host transfer (np.asarray),
which is the real wait point under the async flag.

The driver runs a fresh worker subprocess per measurement and alternates
sync/async order across rounds, because the flag is read once per process and
because single-shot numbers on this hardware swing with thermal throttling
(interleaving balances that out). It reports the median wall-time per mode and
the speedup.

Usage:
    uv run python scripts/bench_async_dispatch.py            # run all workloads
    uv run python scripts/bench_async_dispatch.py --rounds 7
    uv run python scripts/bench_async_dispatch.py --workload mlp_small
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time


# Each workload returns (init_state, step_fn). step_fn must be shape-preserving
# so it can be chained: state = step(state) for K iterations. Workloads are
# chosen to span the gain spectrum: small matrices dominated by per-dispatch
# overhead (async should win big) up to large compute-bound matmuls (async
# should win little, since GPU time dwarfs dispatch).
def _workloads():
    import jax.numpy as jnp
    from jax import random

    def mlp(n, batch, steps):
        key = random.key(0)
        kx, kw = random.split(key)
        x0 = random.normal(kx, (batch, n))
        w = random.normal(kw, (n, n)) * (1.0 / n**0.5)

        def step(x):
            # Shape-preserving transformer-ish block: keeps values bounded so a
            # long chain stays finite and the comparison is apples-to-apples.
            return jnp.tanh(x @ w)

        return (x0, w), step, steps

    return {
        # n=64: tiny kernels, dispatch overhead dominates -> async should win.
        "mlp_small": lambda: mlp(64, 64, 2000),
        # n=256: moderate.
        "mlp_medium": lambda: mlp(256, 128, 1000),
        # n=1024: compute-bound, GPU time dominates -> async should win little.
        "mlp_large": lambda: mlp(1024, 256, 400),
    }


def run_worker(workload: str, rounds_seed: int) -> dict:
    """Time one chained workload in this (already env-configured) process."""
    import jax
    import jax.numpy as jnp  # noqa: F401  (imported for side effects / parity)
    import numpy as np

    device = jax.devices("mps")[0]
    spec = _workloads()[workload]()
    # `step` closes over its weights; only the chained state needs placing.
    (state, *_consts), step, n_steps = spec
    state = jax.device_put(state, device)

    step_fn = jax.jit(step)

    def chain(s):
        for _ in range(n_steps):
            s = step_fn(s)
        return s

    # Warm up: compile + prime caches; host transfer forces a full sync.
    np.asarray(chain(state))
    np.asarray(chain(state))

    # Measure: dispatch the whole chain, then a single host transfer forces the
    # real wait (this is the only sync point, async or not).
    t0 = time.perf_counter()
    out = chain(state)
    host = np.asarray(out)  # forces ToHostBuffer -> eval -> GPU wait
    dt = time.perf_counter() - t0

    # Return a cheap checksum so the driver can assert numerical agreement
    # between sync and async runs.
    checksum = float(host.astype(np.float64).sum())
    return {
        "workload": workload,
        "n_steps": n_steps,
        "seconds": dt,
        "checksum": checksum,
        "async": bool(os.environ.get("JAX_MPS_ASYNC_DISPATCH")),
    }


def _worker_main(argv) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--workload", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    result = run_worker(a.workload, a.seed)
    print("RESULT " + json.dumps(result))
    return 0


def _spawn(workload: str, use_async: bool, seed: int) -> dict:
    env = dict(os.environ)
    if use_async:
        env["JAX_MPS_ASYNC_DISPATCH"] = "1"
    else:
        env.pop("JAX_MPS_ASYNC_DISPATCH", None)
    # Quiet the experimental-platform warning.
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--workload",
        workload,
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"worker failed for {workload} (async={use_async})")
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def driver(workloads: list[str], rounds: int) -> int:
    print(f"Interleaved A/B over {rounds} rounds (fresh subprocess per run)\n")
    header = f"{'workload':<14}{'sync (ms)':>12}{'async (ms)':>12}{'speedup':>10}"
    print(header)
    print("-" * len(header))

    overall_ok = True
    for wl in workloads:
        sync_times: list[float] = []
        async_times: list[float] = []
        checks: list[float] = []
        for r in range(rounds):
            # Alternate which mode goes first each round to balance thermal drift.
            order = [False, True] if r % 2 == 0 else [True, False]
            for use_async in order:
                res = _spawn(wl, use_async, seed=r)
                (async_times if use_async else sync_times).append(res["seconds"])
                checks.append(res["checksum"])

        # Numerical agreement: every run of a given workload must match.
        cmin, cmax = min(checks), max(checks)
        denom = max(abs(cmin), abs(cmax), 1.0)
        agree = (cmax - cmin) / denom < 1e-5
        overall_ok &= agree

        s = statistics.median(sync_times) * 1e3
        a = statistics.median(async_times) * 1e3
        speedup = s / a if a > 0 else float("nan")
        flag = "" if agree else "  <-- CHECKSUM MISMATCH"
        print(f"{wl:<14}{s:>12.2f}{a:>12.2f}{speedup:>9.2f}x{flag}")

    print()
    print("Higher speedup = bigger async-dispatch win.")
    print("Expectation: large for small/many-dispatch, ~1x for compute-bound.")
    if not overall_ok:
        print("\nWARNING: sync vs async produced different results.")
        return 1
    return 0


def main() -> int:
    if "--worker" in sys.argv:
        return _worker_main(sys.argv[1:])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument(
        "--workload",
        default=None,
        help="run a single named workload instead of all",
    )
    a = p.parse_args()
    import jax  # noqa: F401  (ensure mps backend importable before spawning)

    all_wl = list(_workloads().keys())
    workloads = [a.workload] if a.workload else all_wl
    return driver(workloads, a.rounds)


if __name__ == "__main__":
    raise SystemExit(main())
