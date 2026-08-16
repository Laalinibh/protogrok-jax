"""Real packet-byte data pipeline for Protogrok-JAX.

This is the training-side counterpart to :mod:`data.pcap`, which already feeds
the model *real* payload bytes at inference time. The tabular path in
:func:`data.tokenizer.load_unsw_nb15` derives byte *proxies* from a hash of the
row (see :func:`data.tokenizer.hash_bytes_from_fields`), which is reproducible
but carries no learnable structure -- two near-identical flows get completely
uncorrelated payloads. Training on that and serving on real bytes is a
train/serve mismatch; this module removes it by using the same tokenization at
train time that ``data.pcap`` uses at inference time.

Source
------
The ``Payload-Bytes`` subset of the ``rdpahalavan/UNSW-NB15`` HuggingFace
dataset: per-packet payload bytes extracted from the original UNSW-NB15 PCAPs
and labelled. Schema (1485 columns, 2,503,016 rows per file, 18 files)::

    packet_id, flow_id, source_ip, source_port, destination_ip,
    destination_port, protocol, payload_length,
    payload_byte_1 .. payload_byte_1476, attack_label

Only ``payload_max_len`` of the 1476 byte columns are read. Parquet is
columnar, so reading 128 of 1476 costs roughly 128/1476 of the file -- which is
why :func:`load_parquet_flows` can stream straight from the Hub without
staging the full 2.7 GB locally.

Flow assembly
-------------
Packets are grouped by ``flow_id`` and ordered by ``packet_id``; the first
``max_packets`` of each flow become the ``[T, L]`` payload tensor. Flows are
stored as raw bytes plus per-packet lengths and tokenized at batch time, which
halves the on-disk footprint and keeps byte 0x00 distinguishable from padding.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from models.config import (
    BYTE_OFFSET, MAX_PACKETS_PER_FLOW, PAD_ID, PAYLOAD_MAX_LEN,
)
from data.tokenizer import Batch, port_bucket

HF_REPO = "rdpahalavan/UNSW-NB15"
HF_TEMPLATE = ("https://huggingface.co/datasets/{repo}/resolve/main/"
               "Payload-Bytes/Payload_Bytes_File_{n}.parquet")

META_COLUMNS = ["flow_id", "packet_id", "source_port", "destination_port",
                "protocol", "payload_length", "attack_label"]

#: UNSW-NB15 ``attack_label`` values -> traffic-class id. 0 is the benign class,
#: so ``label_binary = (label_class != 0)``.
ATTACK_CLASSES = ("normal", "analysis", "backdoor", "dos", "exploits",
                  "fuzzers", "generic", "reconnaissance", "shellcode", "worms")

#: Protocol name -> IANA number. ``data.pcap`` derives ``proto_id`` from the IP
#: header's numeric protocol field, so training must use the same numbering or
#: the protocol embedding means something different at train and serve time.
_PROTO_NUMBERS = {"hopopt": 0, "icmp": 1, "igmp": 2, "ggp": 3, "ipv4": 4,
                  "st": 5, "tcp": 6, "cbt": 7, "egp": 8, "igp": 9,
                  "udp": 17, "dccp": 33, "ipv6": 41, "rsvp": 46, "gre": 47,
                  "esp": 50, "ah": 51, "icmpv6": 58, "eigrp": 88, "ospf": 89,
                  "pim": 103, "vrrp": 112, "l2tp": 115, "sctp": 132}


def payload_columns(payload_max_len: int = PAYLOAD_MAX_LEN) -> List[str]:
    """The ``payload_byte_*`` column names we actually need (1-indexed)."""
    return [f"payload_byte_{i}" for i in range(1, payload_max_len + 1)]


def proto_to_number(value) -> int:
    """Map a ``protocol`` cell to its IANA number, matching ``data.pcap``.

    The column may hold either numbers or protocol names depending on the
    export; both are accepted. Unknown names fall back to a stable hash so the
    id space stays deterministic rather than silently collapsing to 0.
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).strip().lower()
    if s.isdigit():
        return int(s)
    if s in _PROTO_NUMBERS:
        return _PROTO_NUMBERS[s]
    # Deterministic fallback above the assigned range (255) so it cannot
    # collide with a real protocol number.
    return 256 + (abs(hash(s)) % 256)


def tokenize(payload: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Raw bytes + per-packet lengths -> model token ids.

    Exactly reproduces :func:`data.tokenizer.payload_to_token_ids`:
    ``BYTE_OFFSET + b`` for the first ``length`` bytes, ``PAD_ID`` after. Using
    the stored length (rather than "byte == 0 means padding") is what keeps a
    genuine 0x00 payload byte distinct from padding.

    Parameters
    ----------
    payload : uint8 ``[..., T, L]`` raw bytes.
    lengths : integer ``[..., T]`` valid byte count per packet.
    """
    L = payload.shape[-1]
    valid = np.arange(L)[None, :] < np.asarray(lengths)[..., None]
    return np.where(valid, payload.astype(np.int32) + BYTE_OFFSET, PAD_ID).astype(np.int32)


# --------------------------------------------------------------------------- #
# Shard container
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class FlowShard:
    """A preprocessed set of flows, backed by ``.npy`` files (memmap-friendly).

    Stored as raw bytes rather than tokens so a shard is half the size and the
    tokenization stays in one place (:func:`tokenize`).
    """

    payload: np.ndarray    # uint8  [N, T, L]  raw payload bytes
    lengths: np.ndarray    # uint8  [N, T]     valid bytes per packet (<= L <= 255)
    headers: np.ndarray    # int32  [N, 5]     [proto_id, sport_b, dport_b, npkts, 0]
    proto_id: np.ndarray   # int32  [N]
    label_bin: np.ndarray  # int32  [N]        0 benign / 1 attack
    label_cls: np.ndarray  # int32  [N]        index into ATTACK_CLASSES

    def __len__(self) -> int:
        return int(self.payload.shape[0])

    # -- persistence -------------------------------------------------------- #
    _ARRAYS = ("payload", "lengths", "headers", "proto_id", "label_bin", "label_cls")

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        for name in self._ARRAYS:
            np.save(os.path.join(out_dir, f"{name}.npy"), getattr(self, name))

    @classmethod
    def load(cls, out_dir: str, mmap: bool = True) -> "FlowShard":
        mode = "r" if mmap else None
        return cls(**{n: np.load(os.path.join(out_dir, f"{n}.npy"), mmap_mode=mode)
                      for n in cls._ARRAYS})

    @classmethod
    def concat(cls, shards: Sequence["FlowShard"]) -> "FlowShard":
        return cls(**{n: np.concatenate([getattr(s, n) for s in shards])
                      for n in cls._ARRAYS})

    def class_balance(self) -> dict:
        vals, counts = np.unique(self.label_cls, return_counts=True)
        return {ATTACK_CLASSES[v] if v < len(ATTACK_CLASSES) else str(v): int(c)
                for v, c in zip(vals, counts)}


# --------------------------------------------------------------------------- #
# Parquet -> flows
# --------------------------------------------------------------------------- #
def hf_url(file_no: int, repo: str = HF_REPO) -> str:
    """Direct URL of one ``Payload-Bytes`` parquet file on the Hub."""
    return HF_TEMPLATE.format(repo=repo, n=file_no)


def _open_source(source: str):
    """Return something ``pyarrow.parquet`` can read: a path or an HTTP file."""
    if source.startswith(("http://", "https://")):
        import fsspec
        return fsspec.open(source).open()
    return source


def load_parquet_flows(
    source: str,
    *,
    max_packets: int = MAX_PACKETS_PER_FLOW,
    payload_max_len: int = PAYLOAD_MAX_LEN,
    proto_vocab: int = 132,
    max_flows: Optional[int] = None,
) -> FlowShard:
    """Read one Payload-Bytes parquet file and assemble per-flow tensors.

    ``source`` may be a local path or an ``https://`` URL (streamed; only the
    needed columns are fetched). Returns a :class:`FlowShard`.
    """
    import pyarrow.parquet as pq

    cols = META_COLUMNS + payload_columns(payload_max_len)
    table = pq.read_table(_open_source(source), columns=cols)
    n_rows = table.num_rows
    if n_rows == 0:
        raise ValueError(f"No rows read from {source!r}")

    def col(name: str) -> np.ndarray:
        return table[name].to_numpy(zero_copy_only=False)

    flow = col("flow_id")
    pid = col("packet_id")
    plen = np.minimum(col("payload_length").astype(np.int64), payload_max_len)
    plen = np.clip(plen, 0, payload_max_len).astype(np.uint8)

    # Byte matrix [n_rows, L]. Parquet gives one array per column; stacking is
    # the dominant cost here, so write straight into a preallocated buffer.
    raw = np.zeros((n_rows, payload_max_len), dtype=np.uint8)
    for i, name in enumerate(payload_columns(payload_max_len)):
        raw[:, i] = np.nan_to_num(col(name), nan=0.0).astype(np.uint8)

    # Group by flow, ordered by packet_id within each flow.
    order = np.lexsort((pid, flow))
    flow_sorted = flow[order]
    starts = np.flatnonzero(np.r_[True, flow_sorted[1:] != flow_sorted[:-1]])
    counts = np.diff(np.r_[starts, n_rows])
    n_flows = len(starts)

    # Rank of each row within its flow; keep only the first `max_packets`.
    rank = np.arange(n_rows) - np.repeat(starts, counts)
    keep = rank < max_packets
    sel = order[keep]                                   # rows, in flow order
    slot = rank[keep].astype(np.int64)                  # packet index within flow
    gidx = np.repeat(np.arange(n_flows), counts)[keep]  # flow index

    if max_flows is not None and n_flows > max_flows:
        mask = gidx < max_flows
        sel, slot, gidx = sel[mask], slot[mask], gidx[mask]
        n_flows = max_flows
        starts, counts = starts[:max_flows], counts[:max_flows]

    payload = np.zeros((n_flows, max_packets, payload_max_len), np.uint8)
    lengths = np.zeros((n_flows, max_packets), np.uint8)
    payload[gidx, slot] = raw[sel]
    lengths[gidx, slot] = plen[sel]

    # Flow-level metadata comes from each flow's first packet.
    first = order[starts]
    protos = np.array([proto_to_number(v) for v in col("protocol")[first]], np.int64)
    proto_id = (protos % proto_vocab).astype(np.int32)
    sport = col("source_port")[first]
    dport = col("destination_port")[first]
    npkts = np.minimum(counts, max_packets).astype(np.int32)

    headers = np.zeros((n_flows, 5), np.int32)
    headers[:, 0] = proto_id
    headers[:, 1] = [port_bucket(p) for p in sport]
    headers[:, 2] = [port_bucket(p) for p in dport]
    headers[:, 3] = npkts

    labels_raw = col("attack_label")[first]
    label_cls = np.array([_class_id(v) for v in labels_raw], np.int32)
    label_bin = (label_cls != 0).astype(np.int32)

    return FlowShard(payload=payload, lengths=lengths, headers=headers,
                     proto_id=proto_id, label_bin=label_bin, label_cls=label_cls)


def _class_id(value) -> int:
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "-"):
        return 0
    try:
        return ATTACK_CLASSES.index(s)
    except ValueError:
        return 0 if s.startswith("normal") else 1  # unknown attack -> generic attack


# --------------------------------------------------------------------------- #
# Iteration
# --------------------------------------------------------------------------- #
def iterate_flow_batches(
    shard: FlowShard,
    batch_size: int,
    *,
    task: str = "anomaly",
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> Iterator[Batch]:
    """Batch iterator over a :class:`FlowShard`, tokenizing on the fly.

    ``task='anomaly'`` yields binary labels; ``task='class'`` yields the
    10-way traffic-class label.
    """
    labels_all = shard.label_bin if task == "anomaly" else shard.label_cls
    idx = np.arange(len(shard))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    n_full = (len(idx) // batch_size) * batch_size
    stop = n_full if drop_last else len(idx)
    for start in range(0, stop, batch_size):
        chunk = np.sort(idx[start:start + batch_size])  # sorted -> memmap friendly
        payload = tokenize(shard.payload[chunk], shard.lengths[chunk])
        yield Batch(payload=payload,
                    headers=np.asarray(shard.headers[chunk], np.int32),
                    proto_id=np.asarray(shard.proto_id[chunk], np.int32),
                    labels=np.asarray(labels_all[chunk], np.int32))


def train_val_split(shard: FlowShard, val_fraction: float = 0.1, seed: int = 0
                    ) -> Tuple[FlowShard, FlowShard]:
    """Split flows into train/val. Splitting by flow (never by packet) is what
    keeps packets of one flow from straddling the boundary and leaking."""
    n = len(shard)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    cut = int(n * (1.0 - val_fraction))
    tr, va = np.sort(idx[:cut]), np.sort(idx[cut:])
    take = lambda a, i: np.asarray(a[i])  # noqa: E731  (materializes from memmap)
    build = lambda i: FlowShard(
        payload=take(shard.payload, i), lengths=take(shard.lengths, i),
        headers=take(shard.headers, i), proto_id=take(shard.proto_id, i),
        label_bin=take(shard.label_bin, i), label_cls=take(shard.label_cls, i))
    return build(tr), build(va)
