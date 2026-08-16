#!/usr/bin/env python
"""Compile the training step on THIS machine and report what it actually needs.

Run this before any long training job. It compiles ``train_step`` for a matrix
of (payload_conv_impl, remat) settings and reports the compiled memory
footprint for each, so a configuration that would OOM three hours in is caught
in about a minute.

    python tools/preflight.py configs/xl_500m.yaml --batch 256
    python tools/preflight.py configs/base_124m.yaml --batch 32

Exits non-zero if no variant fits in device memory.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from models.config import ProtogrokConfig  # noqa: E402
from models.protogrok import count_params  # noqa: E402
from optimization.optimizer import (  # noqa: E402
    build_optimizer, create_train_state, train_step,
)

GB = 1e9


def device_memory_bytes() -> float:
    """Total memory of device 0, or 0.0 if it cannot be determined."""
    try:
        d = jax.devices()[0]
        stats = d.memory_stats() or {}
        for key in ("bytes_limit", "bytes_reservable_limit", "largest_free_block_bytes"):
            if stats.get(key):
                return float(stats[key])
    except Exception:
        pass
    return 0.0


def probe(cfg: ProtogrokConfig, batch: int, task: str = "anomaly") -> dict:
    state = create_train_state(cfg, jax.random.PRNGKey(0),
                               build_optimizer(100, peak_lr=1e-4, warmup_steps=2))
    b = {"payload": jnp.zeros((batch, cfg.max_packets, cfg.payload_max_len), jnp.int32),
         "headers": jnp.zeros((batch, 5), jnp.int32),
         "proto_id": jnp.zeros((batch,), jnp.int32),
         "labels": jnp.zeros((batch,), jnp.int32),
         "weights": jnp.ones((batch,), jnp.float32)}
    compiled = jax.jit(train_step, static_argnames=("task",)).lower(
        state, b, task=task).compile()
    m = compiled.memory_analysis()
    return {"params": count_params(state.params),
            "temp": m.temp_size_in_bytes,
            "args": m.argument_size_in_bytes,
            "total": m.temp_size_in_bytes + m.argument_size_in_bytes}


def main() -> None:
    p = argparse.ArgumentParser(description="Protogrok preflight memory check")
    p.add_argument("config")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--task", default="anomaly")
    args = p.parse_args()

    import jaxlib
    print(f"jax {jax.__version__} | jaxlib {jaxlib.__version__} | "
          f"backend {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    cap = device_memory_bytes()
    print(f"device memory: {cap/GB:.1f} GB" if cap else "device memory: unknown")

    base = ProtogrokConfig.from_yaml(args.config)
    print(f"\nconfig {os.path.basename(args.config)} | batch {args.batch} | task {args.task}")
    print(f"{'conv_impl':<10} {'remat':<7} {'params':>10} {'temp':>10} "
          f"{'args':>10} {'total':>10}  fits?")

    results, ok_any = [], False
    for conv in ("conv", "matmul"):
        for remat in (False, True):
            cfg = ProtogrokConfig.from_dict(
                {**base.to_dict(), "payload_conv_impl": conv, "remat": remat})
            try:
                r = probe(cfg, args.batch, args.task)
            except Exception as exc:
                msg = str(exc).split("\n")[0][:44]
                print(f"{conv:<10} {str(remat):<7} {'-':>10} {'-':>10} "
                      f"{'-':>10} {'-':>10}  FAILED: {msg}")
                continue
            fits = "" if not cap else ("yes" if r["total"] < cap * 0.9 else "NO")
            ok_any = ok_any or fits != "NO"
            results.append((conv, remat, r))
            print(f"{conv:<10} {str(remat):<7} {r['params']/1e6:>9.1f}M "
                  f"{r['temp']/GB:>9.2f}G {r['args']/GB:>9.2f}G "
                  f"{r['total']/GB:>9.2f}G  {fits}")

    if not results:
        print("\nEvery variant failed to compile. Do not start a training run.")
        raise SystemExit(2)

    best = min(results, key=lambda t: t[2]["total"])
    print(f"\nlowest footprint: payload_conv_impl={best[0]}, remat={best[1]} "
          f"({best[2]['total']/GB:.2f} GB)")
    if cap and not ok_any:
        print("None fit in device memory -- lower --batch or shrink the model.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
