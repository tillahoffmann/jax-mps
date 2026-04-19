"""Full RNN training benchmark for jax-mps branch comparison.

Measures wall-clock time for complete GRU training steps including:
  - Forward pass through lax.scan
  - MSE loss computation
  - Backward pass (grad through scan)
  - Parameter update via SGD

Usage:
  python benchmarks/bench_rnn_training.py --save pr_training.json
  # switch branch, rebuild
  python benchmarks/bench_rnn_training.py --save main_training.json
  # compare
  python benchmarks/bench_rnn_training.py --compare main_training.json pr_training.json
"""

import argparse
import json
import os
import time
from functools import partial

os.environ.setdefault("JAX_PLATFORMS", "mps")

import jax
import jax.numpy as jnp
from jax import lax, random

BATCH_SIZE = 32
SEQ_LEN = 200
INPUT_DIM = 64

# ---------------------------------------------------------------------------
# GRU Model
# ---------------------------------------------------------------------------


def init_gru_params(key, input_dim, hidden_dim, output_dim):
    """Initialize GRU + linear readout parameters."""
    keys = random.split(key, 8)
    def scale(shape):
        return 1.0 / jnp.sqrt(shape[-1])

    params = {
        # GRU gates
        "Wz": random.normal(keys[0], (input_dim, hidden_dim)) * scale((input_dim,)),
        "Uz": random.normal(keys[1], (hidden_dim, hidden_dim)) * scale((hidden_dim,)),
        "bz": jnp.zeros(hidden_dim),
        "Wr": random.normal(keys[2], (input_dim, hidden_dim)) * scale((input_dim,)),
        "Ur": random.normal(keys[3], (hidden_dim, hidden_dim)) * scale((hidden_dim,)),
        "br": jnp.zeros(hidden_dim),
        "Wh": random.normal(keys[4], (input_dim, hidden_dim)) * scale((input_dim,)),
        "Uh": random.normal(keys[5], (hidden_dim, hidden_dim)) * scale((hidden_dim,)),
        "bh": jnp.zeros(hidden_dim),
        # Linear readout
        "Wo": random.normal(keys[6], (hidden_dim, output_dim)) * scale((hidden_dim,)),
        "bo": jnp.zeros(output_dim),
    }
    return params


def gru_forward(params, xs, unroll):
    """Forward pass: GRU scan over sequence, return final hidden → output."""
    hidden_dim = params["Wz"].shape[1]

    def gru_cell(h, x):
        z = jax.nn.sigmoid(x @ params["Wz"] + h @ params["Uz"] + params["bz"])
        r = jax.nn.sigmoid(x @ params["Wr"] + h @ params["Ur"] + params["br"])
        h_tilde = jnp.tanh(x @ params["Wh"] + (r * h) @ params["Uh"] + params["bh"])
        h_new = (1 - z) * h + z * h_tilde
        return h_new, None

    h0 = jnp.zeros((xs.shape[1], hidden_dim))  # (batch, hidden)
    h_final, _ = lax.scan(gru_cell, h0, xs, unroll=unroll)
    logits = h_final @ params["Wo"] + params["bo"]
    return logits


def mse_loss(params, xs, targets, unroll):
    """MSE loss over batch."""
    preds = gru_forward(params, xs, unroll)
    return jnp.mean((preds - targets) ** 2)


def train_step(params, xs, targets, lr, unroll):
    """Single training step: forward + backward + SGD update."""
    loss, grads = jax.value_and_grad(mse_loss)(params, xs, targets, unroll)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, loss


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

CONFIGS = [
    # (name, hidden_dim, output_dim)
    ("h512", 512, 32),
    ("h1024", 1024, 64),
    ("h2048", 2048, 128),
]

UNROLLS = [1, 10, 50]  # full unroll added per config


def bench_training(warmup=3, repeats=10):
    """Run full training step benchmark across configs and unroll factors."""
    key = random.PRNGKey(42)
    results = {}

    for name, hidden_dim, output_dim in CONFIGS:
        k1, k2, k3, key = random.split(key, 4)
        params = init_gru_params(k1, INPUT_DIM, hidden_dim, output_dim)
        xs = random.normal(k2, (SEQ_LEN, BATCH_SIZE, INPUT_DIM))
        targets = random.normal(k3, (BATCH_SIZE, output_dim))
        lr = 0.001

        unrolls = UNROLLS + [SEQ_LEN]

        print(f"\n{'=' * 78}")
        print(
            f"  {name}: in={INPUT_DIM} hidden={hidden_dim} out={output_dim} "
            f"T={SEQ_LEN} batch={BATCH_SIZE}"
        )
        print(f"{'=' * 78}")
        print(f"  {'unroll':<14}  {'step time':>12}  {'ms/seq_step':>12}  {'loss':>12}")
        print(f"  {'-' * 14}  {'-' * 12}  {'-' * 12}  {'-' * 12}")

        config_results = {}
        for unroll in unrolls:
            label = f"{unroll}" if unroll < SEQ_LEN else f"{unroll} (full)"

            step_jit = jax.jit(partial(train_step, lr=lr, unroll=unroll))

            try:
                # Warmup
                p = params
                for _ in range(warmup):
                    p, loss = step_jit(p, xs, targets)
                    loss.block_until_ready()

                # Timed runs
                times = []
                p = params
                loss = jnp.array(0.0)
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    p, loss = step_jit(p, xs, targets)
                    loss.block_until_ready()
                    times.append((time.perf_counter() - t0) * 1000)

                times.sort()
                med = times[len(times) // 2]
                final_loss = float(loss)
                config_results[str(unroll)] = round(med, 2)
                print(
                    f"  {label:<14}  {med:10.2f} ms  {med / SEQ_LEN:10.3f} ms  {final_loss:10.6f}"
                )

            except Exception as e:
                config_results[str(unroll)] = None
                print(f"  {label:<14}  {'FAILED':>12}  {'':>12}  {str(e)[:40]}")

        results[name] = config_results

    return results


def compare(baseline_path, current_path):
    """Compare two saved result files."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    print(f"\n{'=' * 84}")
    print("  Full Training Step Comparison (fwd + bwd + SGD)")
    print(f"  baseline: {baseline_path}")
    print(f"  current:  {current_path}")
    print(f"{'=' * 84}")
    print(
        f"  {'config':<10} {'unroll':>8}  {'baseline':>12}  {'current':>12}  {'speedup':>10}"
    )
    print(f"  {'-' * 10} {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

    for config in baseline:
        if config not in current:
            continue
        for unroll in baseline[config]:
            b = baseline[config].get(unroll)
            c = current[config].get(unroll)
            if b is None or c is None:
                status = "FAIL" if c is None else "N/A"
                print(f"  {config:<10} {unroll:>8}  {b or 'N/A':>12}  {status:>12}")
                continue
            speedup = b / c
            marker = " ✓" if speedup > 1.05 else (" ✗" if speedup < 0.95 else "")
            print(
                f"  {config:<10} {unroll:>8}  {b:10.2f} ms  {c:10.2f} ms  {speedup:8.2f}x{marker}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full RNN training benchmark")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CURRENT"),
        help="Compare two saved result files",
    )
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
    else:
        results = bench_training()
        if args.save:
            with open(args.save, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.save}")
        print("\nTraining benchmark complete.")
