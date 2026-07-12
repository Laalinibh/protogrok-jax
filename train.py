#!/usr/bin/env python
"""Main pretraining / fine-tuning execution script for Protogrok-JAX.

Reads a model config (``configs/*.yaml``) and training config, loads UNSW-NB15
(or runs on synthetic data for a smoke test), and trains with the Optax
optimizer defined in ``optimization/``. ``pmap``-ready: ``train_step`` is a pure
function of ``(state, batch)``.

Examples
--------
    # Smoke test (no data needed) — verifies the whole loop end to end:
    python train.py --model-config configs/large_300m.yaml --synthetic --steps 5

    # Real training on UNSW-NB15:
    python train.py --model-config configs/base_124m.yaml \
        --train-csv UNSW_NB15_training-set.csv --test-csv UNSW_NB15_testing-set.csv \
        --train-config configs/train.yaml --out checkpoints/anomaly
"""
from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from models.config import ProtogrokConfig
from models.protogrok import count_params
from data.tokenizer import iterate_batches, load_unsw_nb15, synthetic_batch
from optimization.optimizer import (
    batch_to_device, build_optimizer, create_train_state, eval_step, train_step,
)
from optimization.batching import BinPackedBatcher, PackingConfig, padding_report
from optimization import checkpointing


def _load_train_cfg(path):
    defaults = dict(epochs=3, batch_size=32, peak_lr=3e-4, warmup_steps=1000,
                    weight_decay=0.01, grad_clip=1.0, seed=0, task="anomaly")
    if path:
        import yaml
        with open(path) as f:
            defaults.update(yaml.safe_load(f) or {})
    return defaults


def run_synthetic(model_cfg: ProtogrokConfig, tcfg, steps: int) -> None:
    rng = jax.random.PRNGKey(tcfg["seed"])
    opt = build_optimizer(total_steps=steps, peak_lr=tcfg["peak_lr"],
                          warmup_steps=min(2, steps), weight_decay=tcfg["weight_decay"],
                          grad_clip=tcfg["grad_clip"])
    state = create_train_state(model_cfg, rng, opt)
    print(f"parameters: {count_params(state.params)/1e6:.1f}M")
    for step in range(steps):
        batch = batch_to_device(synthetic_batch(tcfg["batch_size"], model_cfg,
                                                 num_labels=2, seed=step))
        t0 = time.time()
        state, m = train_step(state, batch, task=tcfg["task"])
        jax.block_until_ready(m["loss"])
        print(f"step {step:03d} | loss {float(m['loss']):.4f} "
              f"| acc {float(m['accuracy']):.3f} | gnorm {float(m['grad_norm']):.2f} "
              f"| {time.time()-t0:.2f}s")


def run_training(model_cfg: ProtogrokConfig, tcfg, args) -> None:
    train, test, meta = load_unsw_nb15(args.train_csv, args.test_csv, max_rows=args.max_rows)
    model_cfg = ProtogrokConfig.from_dict({**model_cfg.to_dict(),
                                           "proto_vocab": meta["proto_vocab"]})
    rng = jax.random.PRNGKey(tcfg["seed"])
    steps_per_epoch = max(1, len(train) // tcfg["batch_size"])
    opt = build_optimizer(total_steps=steps_per_epoch * tcfg["epochs"],
                          peak_lr=tcfg["peak_lr"],
                          warmup_steps=min(tcfg["warmup_steps"], steps_per_epoch),
                          weight_decay=tcfg["weight_decay"], grad_clip=tcfg["grad_clip"])
    state = create_train_state(model_cfg, rng, opt)
    print(f"parameters: {count_params(state.params)/1e6:.1f}M | train {len(train)} "
          f"| proto_vocab {meta['proto_vocab']}")

    pack_cfg = None
    if args.pack:
        pack_cfg = PackingConfig(strategy=args.pack_strategy, num_buckets=args.num_buckets,
                                 max_batch=tcfg["batch_size"], token_budget=args.token_budget)
        rep = padding_report(train, pack_cfg)
        print(f"[bin-packing] buckets={rep['buckets']} | padding reduced "
              f"{rep['padding_reduction']*100:.1f}% | packed_efficiency={rep['packed_efficiency']} "
              f"| distinct JIT shapes={rep['distinct_shapes']}")

    def train_batches(epoch):
        if pack_cfg is not None:
            pack_cfg_epoch = PackingConfig(**{**pack_cfg.__dict__, "seed": epoch})
            return BinPackedBatcher(train, pack_cfg_epoch)
        return iterate_batches(train, tcfg["batch_size"], seed=epoch)

    for epoch in range(tcfg["epochs"]):
        losses, t0 = [], time.time()
        for b in train_batches(epoch):
            state, m = train_step(state, batch_to_device(b), task=tcfg["task"])
            losses.append(float(m["loss"]))
        msg = f"epoch {epoch+1} | train_loss {np.mean(losses):.4f} | {time.time()-t0:.1f}s"
        if test:
            accs = [float(eval_step(state, batch_to_device(b), task=tcfg["task"])["accuracy"])
                    for b in iterate_batches(test, tcfg["batch_size"], shuffle=False, drop_last=False)]
            msg += f" | val_acc {np.mean(accs):.4f}"
        print(msg)

    if args.out:
        checkpointing.save(args.out, state.params, model_cfg, step=int(state.step))
        print(f"Saved checkpoint to {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Protogrok-JAX training")
    p.add_argument("--model-config", default="configs/large_300m.yaml")
    p.add_argument("--train-config", default=None)
    p.add_argument("--train-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--synthetic", action="store_true", help="smoke test on random data")
    p.add_argument("--steps", type=int, default=5, help="steps for --synthetic")
    # bin-packing dynamic batcher (OR component)
    p.add_argument("--pack", action="store_true", help="use the bin-packing dynamic batcher")
    p.add_argument("--pack-strategy", choices=["bucket", "ffd"], default="bucket")
    p.add_argument("--num-buckets", type=int, default=4)
    p.add_argument("--token-budget", type=int, default=None)
    args = p.parse_args()

    model_cfg = ProtogrokConfig.from_yaml(args.model_config)
    tcfg = _load_train_cfg(args.train_config)

    if args.synthetic or not args.train_csv:
        run_synthetic(model_cfg, tcfg, args.steps)
    else:
        run_training(model_cfg, tcfg, args)


if __name__ == "__main__":
    main()
