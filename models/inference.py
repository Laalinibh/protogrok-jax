"""Inference / evaluation utilities: anomaly scoring over flows and metrics."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from models.config import ProtogrokConfig
from data.tokenizer import Batch, iterate_batches
from models.protogrok import ProtogrokModel


def macro_f1(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Host-side macro-averaged F1 (mirrors the notebook's sklearn F1)."""
    preds = np.asarray(logits).argmax(-1)
    labels = np.asarray(labels)
    scores = []
    for c in range(num_classes):
        tp = int(np.sum((preds == c) & (labels == c)))
        fp = int(np.sum((preds == c) & (labels != c)))
        fn = int(np.sum((preds != c) & (labels == c)))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def make_apply(cfg: ProtogrokConfig, params):
    """Return a jitted predictor: (payload, headers, proto_id, task) -> logits."""
    model = ProtogrokModel(cfg)

    from functools import partial

    @partial(jax.jit, static_argnames=("task",))
    def _apply(payload, headers, proto_id, task="anomaly"):
        return model.apply({"params": params}, payload, headers, proto_id,
                           task=task, deterministic=True)

    return _apply


def anomaly_scores(cfg: ProtogrokConfig, params, batch: Batch) -> np.ndarray:
    """Softmax P(anomaly) for each flow in the batch."""
    apply = make_apply(cfg, params)
    logits = apply(jnp.asarray(batch.payload), jnp.asarray(batch.headers),
                   jnp.asarray(batch.proto_id), task="anomaly")
    return np.asarray(jax.nn.softmax(logits, axis=-1)[:, 1])


def evaluate(cfg: ProtogrokConfig, params, examples: Sequence,
             *, task: str = "anomaly", batch_size: int = 64,
             num_classes: int = 2) -> Dict[str, float]:
    """Aggregate accuracy / macro-F1 over a dataset."""
    apply = make_apply(cfg, params)
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for b in iterate_batches(examples, batch_size, shuffle=False, drop_last=False):
        logits = apply(jnp.asarray(b.payload), jnp.asarray(b.headers),
                       jnp.asarray(b.proto_id), task=task)
        all_logits.append(np.asarray(logits))
        all_labels.append(b.labels)
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    acc = float((logits.argmax(-1) == labels).mean())
    return {"accuracy": acc, "macro_f1": macro_f1(logits, labels, num_classes),
            "n": int(len(labels))}
