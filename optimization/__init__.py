"""Optimization: optimizer construction, training/eval steps, checkpointing.

This package holds the training loop and optimizer configuration. The current
optimizer is a standard AdamW with a warmup-cosine schedule and global-norm
gradient clipping (see ``optimizer.py``). It is the natural home for any
Operations-Research-style optimization component (e.g. a bin-packing dynamic
batcher or a constraint-based scheduler) should one be added.
"""
from optimization.optimizer import (
    TrainState, build_optimizer, create_train_state,
    train_step, eval_step, batch_to_device, softmax_ce, accuracy, macro_f1,
)
from optimization.batching import (
    PackingConfig, BinPackedBatcher, optimal_buckets, pack_ffd, pack_bucketed,
    padding_report, real_packet_count,
)
from optimization import checkpointing

__all__ = [
    "TrainState", "build_optimizer", "create_train_state", "train_step",
    "eval_step", "batch_to_device", "softmax_ce", "accuracy", "macro_f1",
    "PackingConfig", "BinPackedBatcher", "optimal_buckets", "pack_ffd",
    "pack_bucketed", "padding_report", "real_packet_count", "checkpointing",
]
