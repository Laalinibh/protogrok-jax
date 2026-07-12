"""Data pipeline: byte tokenization, header featurization, and batching.

Faithful port of the notebook's tokenizer (``payload_to_token_ids``,
``hash_bytes_from_fields``, ``port_bucket``) plus a UNSW-NB15 CSV loader and
a deterministic NumPy batch iterator suitable for feeding the JAX model.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from models.config import (
    BYTE_OFFSET, MAX_PACKETS_PER_FLOW, PAD_ID, PAYLOAD_MAX_LEN,
)


# --------------------------------------------------------------------------- #
# Tokenization / featurization (parity with the PyTorch notebook)
# --------------------------------------------------------------------------- #
def payload_to_token_ids(payload: bytes, max_len: int = PAYLOAD_MAX_LEN) -> List[int]:
    toks = [BYTE_OFFSET + b for b in payload[:max_len]]
    if len(toks) < max_len:
        toks += [PAD_ID] * (max_len - len(toks))
    return toks


def port_bucket(p) -> int:
    try:
        p = int(p)
    except (TypeError, ValueError):
        return 0
    if p < 0:
        return 0
    if p < 1024:
        return 1
    if p < 49152:
        return 2
    return 3


def hash_bytes_from_fields(values: Sequence, length: int) -> bytes:
    """Deterministically derive byte proxies from tabular fields.

    Uses a stable hash (blake2b) rather than Python's salted ``hash`` so the
    mapping is reproducible across processes/runs, then a seeded RNG.
    """
    s = "|".join(str(v) for v in values).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "little") % (2 ** 32)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()


# --------------------------------------------------------------------------- #
# Batch container
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Batch:
    payload: np.ndarray   # int32 [B, T, L]
    headers: np.ndarray   # int32 [B, 5]
    proto_id: np.ndarray  # int32 [B]
    labels: np.ndarray    # int32 [B]

    def as_numpy(self) -> Dict[str, np.ndarray]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Tabular -> flow tensors
# --------------------------------------------------------------------------- #
def row_to_example(
    fields: Sequence,
    proto_id: int,
    sport,
    dport,
    label: int,
    *,
    max_packets: int = MAX_PACKETS_PER_FLOW,
    payload_max_len: int = PAYLOAD_MAX_LEN,
    packets_derived: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Turn one tabular record into (payload[T,L], header[5], proto_id, label).

    Tabular datasets (UNSW-NB15) have no raw bytes, so we derive deterministic
    byte proxies per packet slot from the row fields (matching the notebook).
    """
    payload = np.full((max_packets, payload_max_len), PAD_ID, dtype=np.int32)
    n = max(1, min(int(packets_derived), max_packets))
    for t in range(n):
        raw = hash_bytes_from_fields(list(fields) + [t], payload_max_len)
        payload[t] = np.asarray(payload_to_token_ids(raw, payload_max_len), dtype=np.int32)
    header = np.asarray(
        [int(proto_id), port_bucket(sport), port_bucket(dport), n, 0], dtype=np.int32)
    return payload, header, int(proto_id), int(label)


def load_unsw_nb15(
    train_csv: str,
    test_csv: Optional[str] = None,
    *,
    label_col: str = "label",
    proto_col: str = "proto",
    sport_col: str = "sport",
    dport_col: str = "dsport",
    max_rows: Optional[int] = None,
):
    """Load UNSW-NB15 CSVs into arrays. Requires pandas.

    Returns ``(train, test, meta)`` where each split is a list of
    ``(payload, header, proto_id, label)`` examples and ``meta`` carries
    ``proto_vocab``.
    """
    import pandas as pd

    def _prep(df: "pd.DataFrame"):
        if max_rows:
            df = df.iloc[:max_rows]
        # map protocol strings/ints to a compact contiguous id space
        protos = sorted(df[proto_col].astype(str).unique())
        proto_map = {p: i for i, p in enumerate(protos)}
        # Real per-flow packet count from UNSW-NB15's spkts/dpkts columns
        # (case-insensitive); enables the bin-packing batcher to see the true
        # length distribution. Falls back to 1 when absent.
        lower = {c.lower(): c for c in df.columns}
        spkts_c, dpkts_c = lower.get("spkts"), lower.get("dpkts")

        def _npackets(r) -> int:
            total = 0
            if spkts_c is not None:
                total += int(pd.to_numeric(r[spkts_c], errors="coerce") or 0)
            if dpkts_c is not None:
                total += int(pd.to_numeric(r[dpkts_c], errors="coerce") or 0)
            return max(1, min(total, MAX_PACKETS_PER_FLOW)) if total else 1

        examples = []
        feat_cols = [c for c in df.columns if c not in (label_col,)]
        for _, r in df.iterrows():
            pid = proto_map[str(r[proto_col])]
            ex = row_to_example(
                fields=[r[c] for c in feat_cols],
                proto_id=pid,
                sport=r.get(sport_col, 0),
                dport=r.get(dport_col, 0),
                label=int(r[label_col]),
                packets_derived=_npackets(r),
            )
            examples.append(ex)
        return examples, proto_map

    train_df = pd.read_csv(train_csv)
    train, proto_map = _prep(train_df)
    test = None
    if test_csv:
        test_df = pd.read_csv(test_csv)
        test, _ = _prep(test_df)
    meta = {"proto_vocab": max(2, len(proto_map) + 1), "proto_map": proto_map}
    return train, test, meta


# --------------------------------------------------------------------------- #
# Iteration
# --------------------------------------------------------------------------- #
def iterate_batches(
    examples: Sequence[Tuple[np.ndarray, np.ndarray, int, int]],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> Iterator[Batch]:
    """Deterministic NumPy batch iterator over pre-tokenized examples."""
    idx = np.arange(len(examples))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    n_full = (len(idx) // batch_size) * batch_size
    stop = n_full if drop_last else len(idx)
    for start in range(0, stop, batch_size):
        chunk = idx[start:start + batch_size]
        payloads = np.stack([examples[i][0] for i in chunk]).astype(np.int32)
        headers = np.stack([examples[i][1] for i in chunk]).astype(np.int32)
        protos = np.asarray([examples[i][2] for i in chunk], dtype=np.int32)
        labels = np.asarray([examples[i][3] for i in chunk], dtype=np.int32)
        yield Batch(payloads, headers, protos, labels)


def synthetic_batch(batch_size: int, cfg, *, num_labels: int = 2, seed: int = 0) -> Batch:
    """A random batch for smoke tests / benchmarking (no real data needed)."""
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, cfg.byte_vocab, (batch_size, cfg.max_packets, cfg.payload_max_len), np.int32)
    headers = np.stack([
        rng.integers(0, cfg.proto_vocab, batch_size),
        rng.integers(0, 4, batch_size),
        rng.integers(0, 4, batch_size),
        rng.integers(1, cfg.max_packets + 1, batch_size),
        np.zeros(batch_size),
    ], axis=1).astype(np.int32)
    proto_id = rng.integers(0, cfg.proto_vocab, batch_size).astype(np.int32)
    labels = rng.integers(0, num_labels, batch_size).astype(np.int32)
    return Batch(payload, headers, proto_id, labels)
