#!/usr/bin/env python
"""Main pretraining / fine-tuning execution script for Protogrok-JAX.

Reads a model config (``configs/*.yaml``) and training config, loads UNSW-NB15
(or runs on synthetic data for a smoke test), and trains with the Optax
optimizer defined in ``optimization/``. ``pmap``-ready: ``train_step`` is a pure
function of ``(state, batch)``.

Two data paths
--------------
``--packet-shard`` (recommended) trains on **real payload bytes** prepared by
``tools/prepare_packets.py`` -- the same tokenization ``data/pcap.py`` uses at
inference, so there is no train/serve mismatch.

``--train-csv`` trains on the tabular UNSW-NB15 partition. That path derives
byte proxies by hashing each row (``data.tokenizer.hash_bytes_from_fields``),
which leaves the model two genuine features (protocol and packet count) and a
payload tensor with no learnable structure. It is kept for reproducibility and
comparison; it will not produce a working detector.

Examples
--------
    # Smoke test (no data needed) — verifies the whole loop end to end:
    python train.py --model-config configs/large_300m.yaml --synthetic --steps 5

    # Real training on packet bytes:
    python train.py --model-config configs/base_124m.yaml \
        --packet-shard data/shards/File_1 \
        --train-config configs/train.yaml --out checkpoints/anomaly

    # Legacy tabular path (hash-proxy payloads — see caveat above):
    python train.py --model-config configs/base_124m.yaml \
        --train-csv UNSW_NB15_training-set.csv --test-csv UNSW_NB15_testing-set.csv \
        --train-config configs/train.yaml --out checkpoints/anomaly
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from models.config import ProtogrokConfig
from models.protogrok import count_params
from data.tokenizer import iterate_batches, load_unsw_nb15, synthetic_batch
from data.packets import FlowShard, iterate_flow_batches, train_val_split
from optimization.optimizer import (
    batch_to_device, build_optimizer, create_train_state, eval_step, macro_f1,
    train_step,
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


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U); no sklearn dependency."""
    labels = np.asarray(labels)
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(np.asarray(scores), kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties so AUC stays correct on a collapsed score band
    s_sorted = np.asarray(scores)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def evaluate(state, batches, task: str, num_classes: int) -> dict:
    """Accuracy + macro-F1 + (binary) AUC and score spread over a val set."""
    logits, labels = [], []
    for b in batches:
        out = eval_step(state, batch_to_device(b), task=task)
        logits.append(np.asarray(out["logits"]))
        labels.append(np.asarray(b.labels))
    if not logits:
        return {}
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    preds = logits.argmax(-1)
    m = {"acc": float((preds == labels).mean()),
         "macro_f1": macro_f1(logits, labels, num_classes)}
    if task == "anomaly":
        e = np.exp(logits - logits.max(-1, keepdims=True))
        p1 = (e / e.sum(-1, keepdims=True))[:, 1]
        m["auc"] = roc_auc(p1, labels)
        # Score spread is the tell for a collapsed head: a model that emits the
        # same probability for every flow has a working loss but no detector.
        m["score_p05"], m["score_p95"] = float(np.percentile(p1, 5)), float(np.percentile(p1, 95))
    return m


def _fmt(m: dict) -> str:
    parts = [f"acc {m['acc']:.4f}", f"macro_f1 {m['macro_f1']:.4f}"]
    if "auc" in m:
        parts.append(f"auc {m['auc']:.4f}")
        parts.append(f"score p05-p95 {m['score_p05']:.3f}-{m['score_p95']:.3f}")
    return " | ".join(parts)


def run_packet_training(model_cfg: ProtogrokConfig, tcfg, args) -> None:
    """Train on real payload bytes from a prepared FlowShard."""
    dirs = args.packet_shard
    if len(dirs) == 1:
        shard = FlowShard.load(dirs[0], mmap=True)
    else:
        # Materialised (not memmapped) so the concatenation is one contiguous
        # array; 18 UNSW shards is ~3.2 GB, which a training box can hold.
        parts = [FlowShard.load(d, mmap=False) for d in dirs]
        shard = FlowShard.concat(parts)
        print(f"loaded {len(parts)} shards -> {len(shard):,} flows")
    train, val = train_val_split(shard, val_fraction=args.val_fraction,
                                 seed=tcfg["seed"])
    task = tcfg["task"]
    num_classes = 2 if task == "anomaly" else model_cfg.num_classes

    rng = jax.random.PRNGKey(tcfg["seed"])
    steps_per_epoch = max(1, len(train) // tcfg["batch_size"])
    opt = build_optimizer(total_steps=steps_per_epoch * tcfg["epochs"],
                          peak_lr=tcfg["peak_lr"],
                          warmup_steps=min(tcfg["warmup_steps"], steps_per_epoch),
                          weight_decay=tcfg["weight_decay"], grad_clip=tcfg["grad_clip"])
    state = create_train_state(model_cfg, rng, opt)

    # flush=True throughout: on a multi-hour GPU run the setup summary must not
    # sit in the stdout buffer until the first epoch ends.
    print(f"parameters: {count_params(state.params)/1e6:.1f}M | "
          f"train {len(train):,} flows | val {len(val):,} flows", flush=True)
    print(f"task: {task} | attack rate train {train.label_bin.mean():.3f} "
          f"val {val.label_bin.mean():.3f}", flush=True)
    print(f"class balance: {shard.class_balance()}", flush=True)

    # Inverse-frequency class weights. Without these the ~5% attack rate lets
    # the model minimise loss by always answering "normal".
    weights = None
    if not args.no_class_weights:
        train_labels = train.label_bin if task == "anomaly" else train.label_cls
        counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
        inv = np.where(counts > 0, len(train_labels) / np.maximum(counts, 1), 0.0)
        weights = jnp.asarray(inv / inv[inv > 0].mean(), jnp.float32)
        print("class weights: " + ", ".join(
            f"{i}:{float(w):.2f}" for i, w in enumerate(weights) if counts[i] > 0),
            flush=True)

    val_batches = lambda: iterate_flow_batches(  # noqa: E731
        val, tcfg["batch_size"], task=task, shuffle=False, drop_last=False)

    best = -1.0
    for epoch in range(tcfg["epochs"]):
        losses, t0 = [], time.time()
        for b in iterate_flow_batches(train, tcfg["batch_size"], task=task,
                                      seed=epoch):
            dev = batch_to_device(b)
            if weights is not None:
                dev["weights"] = weights[dev["labels"]]
            state, m = train_step(state, dev, task=task)
            losses.append(float(m["loss"]))
        msg = (f"epoch {epoch+1} | train_loss {np.mean(losses):.4f} "
               f"| {time.time()-t0:.1f}s")
        metrics = evaluate(state, val_batches(), task, num_classes)
        if metrics:
            msg += " | " + _fmt(metrics)

        # Checkpoint the best epoch as we go. Saving only at the end means an
        # interrupted run -- or one you stop early because it is good enough --
        # yields nothing at all.
        if args.out and metrics:
            score = metrics.get("auc") or metrics.get("macro_f1", 0.0)
            if score != score:  # NaN guard (single-class val split)
                score = metrics.get("macro_f1", 0.0)
            if score > best:
                best = score
                checkpointing.save(args.out, state.params, model_cfg,
                                   step=int(state.step))
                msg += f"  [saved: best {score:.4f}]"
        print(msg, flush=True)

    if args.out:
        print(f"Best checkpoint ({best:.4f}) at {args.out}")


def run_training(model_cfg: ProtogrokConfig, tcfg, args) -> None:
    print("=" * 78)
    print("WARNING: tabular CSV path. Payload tensors are hash-derived proxies")
    print("  (data.tokenizer.hash_bytes_from_fields), not real bytes: flows that")
    print("  differ slightly get uncorrelated payloads, so the payload encoder has")
    print("  nothing to generalise from. UNSW-NB15 has no sport/dsport columns, so")
    print("  both port buckets are constant. The model sees ~2 real features and")
    print("  is served real bytes at inference. Use --packet-shard for detection.")
    print("=" * 78)
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
    p.add_argument("--packet-shard", default=None, nargs="+",
                   help="one or more directories written by "
                        "tools/prepare_packets.py (real payload bytes — preferred)")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="held-out fraction for --packet-shard, split by flow")
    p.add_argument("--no-class-weights", action="store_true",
                   help="disable inverse-frequency loss weighting (real traffic "
                        "is ~5%% attack; without weights the head collapses)")
    p.add_argument("--train-csv", default=None,
                   help="legacy tabular path (hash-proxy payloads)")
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

    if args.packet_shard:
        run_packet_training(model_cfg, tcfg, args)
    elif args.synthetic or not args.train_csv:
        run_synthetic(model_cfg, tcfg, args.steps)
    else:
        run_training(model_cfg, tcfg, args)


if __name__ == "__main__":
    main()
