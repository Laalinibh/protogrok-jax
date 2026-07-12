"""Training: Optax optimizer, TrainState, jitted steps, and metrics.

DeepMind-style functional training: a pure ``train_step`` closed over static
config, a cosine-with-warmup schedule, AdamW with global-norm clipping, and a
multi-task cross-entropy loss. ``pmap``-ready (see :func:`replicate`).
"""
from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from models.config import ProtogrokConfig
from data.tokenizer import Batch
from models.protogrok import ProtogrokModel


class TrainState(train_state.TrainState):
    """TrainState with a folded-in dropout RNG."""
    dropout_rng: jax.Array


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
def build_optimizer(
    total_steps: int,
    *,
    peak_lr: float = 3e-4,
    warmup_steps: int = 1000,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=peak_lr * 0.1,
    )
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay,
                    b1=0.9, b2=0.95, eps=1e-8),
    )


def create_train_state(
    cfg: ProtogrokConfig,
    rng: jax.Array,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    params_rng, dropout_rng = jax.random.split(rng)
    model = ProtogrokModel(cfg)
    dummy = _dummy_inputs(cfg)
    variables = model.init(
        {"params": params_rng, "dropout": dropout_rng},
        *dummy, task="anomaly", deterministic=True)
    return TrainState.create(
        apply_fn=model.apply, params=variables["params"],
        tx=optimizer, dropout_rng=dropout_rng)


def _dummy_inputs(cfg: ProtogrokConfig):
    L, T = cfg.payload_max_len, cfg.max_packets
    return (jnp.zeros((2, T, L), jnp.int32),
            jnp.zeros((2, 5), jnp.int32),
            jnp.zeros((2,), jnp.int32))


# --------------------------------------------------------------------------- #
# Loss / metrics
# --------------------------------------------------------------------------- #
def softmax_ce(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return (jnp.argmax(logits, -1) == labels).mean()


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
@functools.partial(jax.jit, static_argnames=("task",))
def train_step(state: TrainState, batch: Dict[str, jnp.ndarray], task: str = "anomaly"
               ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    dropout_rng = jax.random.fold_in(state.dropout_rng, state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, batch["payload"], batch["headers"], batch["proto_id"],
            task=task, deterministic=False, rngs={"dropout": dropout_rng})
        loss = softmax_ce(logits, batch["labels"])
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, "accuracy": accuracy(logits, batch["labels"]),
               "grad_norm": optax.tree_utils.tree_l2_norm(grads)}
    return state, metrics


@functools.partial(jax.jit, static_argnames=("task",))
def eval_step(state: TrainState, batch: Dict[str, jnp.ndarray], task: str = "anomaly"
              ) -> Dict[str, jnp.ndarray]:
    logits = state.apply_fn(
        {"params": state.params}, batch["payload"], batch["headers"], batch["proto_id"],
        task=task, deterministic=True)
    return {"loss": softmax_ce(logits, batch["labels"]),
            "accuracy": accuracy(logits, batch["labels"]),
            "logits": logits}


def batch_to_device(b: Batch) -> Dict[str, jnp.ndarray]:
    return {"payload": jnp.asarray(b.payload), "headers": jnp.asarray(b.headers),
            "proto_id": jnp.asarray(b.proto_id), "labels": jnp.asarray(b.labels)}


def macro_f1(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Host-side macro-F1 for reporting (mirrors the notebook's sklearn F1)."""
    preds = np.asarray(logits).argmax(-1)
    labels = np.asarray(labels)
    f1s = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))
