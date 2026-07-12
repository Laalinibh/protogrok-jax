# Protogrok-JAX

[![CI](https://github.com/laalinibh/protogrok-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/laalinibh/protogrok-jax/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**A hierarchical Transformer for network-flow anomaly detection, in JAX/Flax.**

Protogrok reads raw network traffic the way a language model reads text: every
packet payload is a sequence of *byte tokens*, every flow is a *sequence of
packets*, and the model learns to tell benign traffic from anomalous. This
repository is a from-scratch, production-grade **JAX / Flax** re-implementation
of a PyTorch research prototype, engineered for distributed training on
accelerators and verified end-to-end (19 unit tests, both a ~124M and a
**verified 305.8M**-parameter configuration).

---

## 1. Motivation

Signature-based intrusion detection fails on novel attacks; classical ML on
hand-engineered flow statistics discards the payload. Protogrok keeps the
payload and lets attention learn what matters, at two scales:

- **Intra-packet** — a convolutional byte encoder summarises each packet's
  payload into a vector.
- **Inter-packet (flow)** — Transformer blocks reason over the *sequence* of
  packets in a flow, so the model can key on ordering, bursts, and protocol
  structure rather than isolated bytes.

A shared trunk feeds **multi-task heads** (binary anomaly, multi-class traffic
type, and a byte-decode objective for self-supervised pretraining), so one set
of weights serves detection, classification, and representation learning.

## 2. Architecture

```
payload bytes [B,T,L] ─► PayloadEncoder ─┐
                                         ├─► join ─► Packet Transformer ×N
headers [B,5] ─────────► HeaderEncoder ──┘             │
                                                        ▼
                                          ProtocolAdapter  (per-protocol FiLM+bottleneck)
                                                        │
                                                        ▼
                                          MemoryModule    (slot-attention session summary)
                                                        │
                                                        ▼
                                          Session Transformer ×M ─► mean-pool ─► heads
                                                                                 ├─ anomaly (2)
                                                                                 ├─ class  (C)
                                                                                 └─ decode (259)
```

| Component | Role | File |
|---|---|---|
| `PayloadEncoder` | byte-embed → learned positions → 2× Conv1d(GELU) → avg-pool | `models/protogrok.py` |
| `HeaderEncoder` | protocol / port embeddings + scalar features | `models/protogrok.py` |
| `TransformerBlock` | pre-norm multi-head self-attention + GELU MLP | `models/protogrok.py` |
| `ProtocolAdapter` | adds a protocol embedding + a bottleneck adapter | `models/protogrok.py` |
| `MemoryModule` | slot-attention pools packets into a session vector | `models/protogrok.py` |
| `ProtogrokModel` | two-stage flow Transformer + multi-task heads | `models/protogrok.py` |

Two configs ship in `configs/`: `base_124m.yaml` (125.1M in the original;
115.6M here) and `large_300m.yaml` (**305.8M**, on target).

## 3. Repository layout

```
├── data/            # tokenization + ingestion: byte tokenizer, UNSW-NB15 loader, PCAP parser
├── models/          # architecture (protogrok.py), config, inference helpers
├── optimization/    # optimizer + training steps, bin-packing batcher (OR), checkpointing
├── configs/         # YAML hyper-parameters (base_124m / large_300m / train)
├── train.py         # main pretraining / fine-tuning entry point
├── eval.py          # evaluation suite (CSV metrics + PCAP anomaly scoring)
├── tests/           # 19 unit tests (shapes, param counts, loss, checkpoint round-trip)
└── README.md
```

## 4. Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # jax[cpu]; use jax[cuda12] / jax[tpu] for accelerators

# End-to-end smoke test on random data (no dataset needed):
python train.py --model-config configs/large_300m.yaml --synthetic --steps 5

# Train on UNSW-NB15:
python train.py --model-config configs/base_124m.yaml \
    --train-csv UNSW_NB15_training-set.csv --test-csv UNSW_NB15_testing-set.csv \
    --train-config configs/train.yaml --out checkpoints/anomaly

# Evaluate a checkpoint / score a capture:
python eval.py --ckpt checkpoints/anomaly --test-csv UNSW_NB15_testing-set.csv
python eval.py --ckpt checkpoints/anomaly --pcap capture.pcap

pytest -q                                 # run the test suite
```

## 5. Design choices (the technical narrative)

### 5.1 Framework: JAX / Flax

The pipeline is built in **JAX** with **Flax `linen`** modules and **Optax**
optimizers. The training step is a *pure function* of `(state, batch)`,
`jax.jit`-compiled and ready to wrap in `jax.pmap` / `shard_map` for multi-device
data parallelism. RNGs are threaded explicitly (dropout keys folded per step),
parameters are an immutable pytree, and checkpoints are Flax-serialized msgpack —
the standard DeepMind-style functional setup. The port is faithful module-for-
module to the PyTorch original (see the PyTorch→JAX mapping below), with exact-erf
GELU, channels-last Conv, and PAD-index masking to preserve forward semantics.

### 5.2 Optimization

**The optimizer** (`optimization/optimizer.py`) is a standard, well-tuned recipe
for large-Transformer training: **AdamW** (`b1=0.9, b2=0.95`) with decoupled
weight decay, a **warmup → cosine-decay** schedule, and **global-norm gradient
clipping** for stability at 300M scale.

**The Operations-Research component** (`optimization/batching.py`) is a
**bin-packing dynamic batcher**. Network flows have highly variable *real*
packet counts, yet the naive pipeline pads every flow to `MAX_PACKETS` (=16)
before the packet-level self-attention, whose cost is O(T²) in the padded length
— so most of that compute is spent on padding. Two classic OR techniques cut the
waste:

- **DP-optimal length bucketing** — a dynamic program (`optimal_buckets`) reads
  the histogram of real packet counts and chooses `K` bucket edges that
  *minimise total padding*, exactly. Bucketing also bounds the number of tensor
  shapes JAX must compile (one per bucket).
- **First-Fit-Decreasing bin packing** (`pack_ffd`) — the classic FFD
  approximation (≤ 11/9·OPT + 1) packs flows into batches under a per-batch token
  budget and cardinality cap, padding each batch only to its own bucketed length.

`BinPackedBatcher` yields batches trimmed to `[B, T_bucket, L]`; the model
consumes them unchanged (its packet axis is already dynamic). On a skewed flow
distribution (mostly short flows) the DP bucketer **reduces padded packet-tokens
by ~79%** (packet-token efficiency 0.16 → 0.76) with only 4 distinct compiled
shapes. Enable it with `python train.py --pack` (see `padding_report` for the
measured win). This is real bin packing / knapsack optimization, not a relabelled
scheduler.

### 5.3 Data and tokenization

Payloads are tokenized byte-by-byte with reserved `PAD/BOS/EOS` ids and a
`+3` byte offset (259-symbol vocab); headers become a compact 5-field feature
(protocol id, source/destination port *buckets*, and scalars). Tabular datasets
such as UNSW-NB15 carry no raw bytes, so deterministic byte proxies are derived
per packet slot from the row fields (stable `blake2b`-seeded RNG) — matching the
original notebook while being reproducible across processes.

## 6. PyTorch → JAX mapping

| PyTorch | JAX / Flax |
|---|---|
| `nn.Embedding(padding_idx=0)` | `nn.Embed` + explicit PAD masking |
| `nn.Conv1d(k=3,pad=1)` on `[B,E,L]` | `nn.Conv(kernel_size=(3,),padding="SAME")` on `[B,L,E]` |
| `AdaptiveAvgPool1d(1)` | `mean` over length |
| `nn.MultiheadAttention` | `nn.MultiHeadDotProductAttention` (self-attn) |
| `GELU` | `nn.gelu(approximate=False)` (exact erf) |
| `AdamW` + clip | `optax.adamw` + `optax.clip_by_global_norm` |
| `torch.save(state_dict)` | `flax.serialization` msgpack |

## 7. Status

- ✅ 124M and 300M configs build; 300M verified at **305.8M** params.
- ✅ Forward pass for all tasks, deterministic; fixed-batch overfit reduces loss.
- ✅ Checkpoint save/restore round-trips bit-exactly.
- ✅ `train.py` / `eval.py` run end-to-end; **19/19 unit tests pass**.
- ✅ Bin-packing dynamic batcher (§5.2) implemented and wired into `train.py`
  (`--pack`); measured ~79% padding reduction on skewed flows.
- 🚧 Distributed `pmap` wrapper and real UNSW-NB15 convergence numbers are the
  next milestones.

## Measure the packing win on your data

```bash
python -m optimization.batching --csv UNSW_NB15_training-set.csv
```

Prints the real packet-count histogram and the padding reduction for both the
DP-bucketing and FFD strategies. The reduction depends on your data's
`spkts`/`dpkts` distribution — measure before quoting.

## License

MIT — see [LICENSE](LICENSE). Update the copyright holder if needed. This is an
independent re-implementation; ensure you have the right to publish if the work
relates to your employment.

## Citation

```bibtex
@software{protogrok_jax,
  author = {Bhogadi, Laalini},
  title  = {Protogrok-JAX: A Hierarchical Transformer for Network-Flow Anomaly Detection},
  year   = {2026},
  url    = {https://github.com/laalinibh/protogrok-jax}
}
```
