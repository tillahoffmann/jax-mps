#!/usr/bin/env python3
"""Interleaved A/B benchmark of JAX_MPS_ASYNC_DISPATCH on the ResNet example.

The #179 trace showed ResNet training is dispatch-bound (lots of tiny compute
encoders, GPU idle between them), which is the regime async dispatch targets.
This driver quantifies the steady-state step-time gain.

The async flag is read once per process at plugin load, so each measurement runs
a fresh `examples/resnet/main.py` subprocess. The example loop is async-friendly
(it does not call .item() per step; it syncs once at the window boundary), so a
real CPU/GPU pipeline can form under the flag. We alternate sync/async order
across rounds because single-shot numbers on this hardware swing ~2x with
thermal throttling — interleaving balances that out. We report the median
steady-state step time per mode and the speedup.

Usage:
    uv run python scripts/bench_resnet_async.py
    uv run python scripts/bench_resnet_async.py --rounds 6 --steps 25 --warmup 5
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "resnet" / "main.py"
_STEP_RE = re.compile(r"Time per step \(steady state, \d+ steps\): ([\d.]+) s")
_LOSS_RE = re.compile(r"Final training loss: ([\d.]+)")


def run_once(async_on: bool, steps: int, warmup: int) -> tuple[float, float]:
    """Run one example subprocess, return (per_step_seconds, final_loss)."""
    env = dict(os.environ)
    if async_on:
        env["JAX_MPS_ASYNC_DISPATCH"] = "1"
    else:
        env.pop("JAX_MPS_ASYNC_DISPATCH", None)
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE), "--steps", str(steps), "--warmup", str(warmup)],
        cwd=EXAMPLE.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"worker failed (async={async_on})")
    step_m = _STEP_RE.search(proc.stdout)
    loss_m = _LOSS_RE.search(proc.stdout)
    if not step_m or not loss_m:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("could not parse worker output")
    return float(step_m.group(1)), float(loss_m.group(1))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()

    sync_times: list[float] = []
    async_times: list[float] = []
    losses: list[float] = []

    for r in range(a.rounds):
        # Alternate which mode runs first to balance thermal drift.
        order = [False, True] if r % 2 == 0 else [True, False]
        for async_on in order:
            t, loss = run_once(async_on, a.steps, a.warmup)
            losses.append(loss)
            tag = "async" if async_on else "sync "
            # Round 0 runs on a cool machine and throttles mid-round; treat it
            # as a thermal warmup and exclude it from the medians.
            warm = r >= 1
            if warm:
                (async_times if async_on else sync_times).append(t)
            mark = "" if warm else "  (warmup, excluded)"
            print(f"round {r} [{tag}] step={t * 1e3:7.1f} ms  loss={loss:.3f}{mark}")
            time.sleep(1.0)  # brief cooldown between runs

    sync_med = statistics.median(sync_times)
    async_med = statistics.median(async_times)
    print("\n=== ResNet18 / CIFAR-10 steady-state step time ===")
    print(f"sync  (flag off): median {sync_med * 1e3:7.1f} ms  n={len(sync_times)}")
    print(f"async (flag on):  median {async_med * 1e3:7.1f} ms  n={len(async_times)}")
    print(f"speedup (sync/async): {sync_med / async_med:.3f}x")
    if len(set(round(x, 3) for x in losses)) == 1:
        print(f"loss identical across all runs: {losses[0]:.3f} (numerically equal)")
    else:
        print(f"loss range: {min(losses):.3f} .. {max(losses):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
