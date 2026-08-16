# Protogrok-JAX

A two-stage Transformer over network flows that reads **real packet payload
bytes** and returns a per-flow anomaly score plus deterministic structural
diagnostics. JAX/Flax, trained on UNSW-NB15.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-306M-yellow)](https://huggingface.co/Anvyon/protogrok-jax-306m)

```
capture.pcap
   -> flows (grouped by 5-tuple, first 16 packets, first 128 bytes each)
   -> PayloadEncoder + HeaderEncoder
   -> 16x TransformerBlock (packet stage)
   -> ProtocolAdapter -> MemoryModule (slot attention)
   -> 8x TransformerBlock (session stage)
   -> anomaly score  +  mismatch findings  +  root cause  +  remediation
```

**Released model:** [`Anvyon/protogrok-jax-306m`](https://huggingface.co/Anvyon/protogrok-jax-306m)
— 306,103,367 parameters, **ROC-AUC 0.9458** on 159,349 held-out flows.
Weights are not in this repo (1.2 GB); see [RESULTS.md](RESULTS.md) for the full
numbers and known limitations.

## Quick start

```bash
pip install -r requirements.txt

# score a capture with the released checkpoint
huggingface-cli download Anvyon/protogrok-jax-306m --local-dir ckpt
python handler.py ckpt capture.pcap
```

Output is one report per flow:

```json
{
  "flow": "10.0.2.50:61234 -> 10.0.0.10:445 (proto 6)",
  "anomaly_score": 0.9922,
  "anomaly": true,
  "mismatch_findings": [
    {"code": "tcp_syn_fin_violation", "severity": "high", "packet_index": 0,
     "message": "Packet 0 sets both SYN and FIN — not a legal TCP state transition."}
  ],
  "root_cause": {"tag": "scanner_or_evasion_tooling",
                 "label": "Scanner / evasion tooling",
                 "confidence": 0.92},
  "suggested_actions": ["Illegal TCP flag combinations are a strong scanner signature — block the source IP."]
}
```

Full request/response examples: [`examples/`](examples/).

## Two signals, different reliability

The response carries two independent things, and they should be trusted
differently:

**`mismatch_findings` / `root_cause`** — deterministic rule checks: illegal TCP
flag combinations, SYN floods, port scans, IP/L4 checksum mismatches, DNS
tunnelling, port/protocol mismatches. No model involved. **Reliable.**

**`anomaly_score`** — the neural model. AUC 0.9458 on held-out UNSW-NB15, but
the released checkpoint is a **single epoch** and saturates on traffic unlike its
training distribution — it scores a crafted attack capture and a benign capture
identically (~0.97 both). **Treat as a ranking hint, not a calibrated
probability.** See [RESULTS.md](RESULTS.md#known-limitation--out-of-distribution-saturation).

## Training

The model consumes **real payload bytes**, not features derived from flow
statistics. This matters: an earlier version of this pipeline derived byte
proxies by hashing tabular CSV rows, which left the payload encoder — 99% of the
parameters — consuming reproducible noise, and capped accuracy near 0.66.
[`logs/training_runs.md`](logs/training_runs.md) documents that in full.

```bash
# 1. Stream packet bytes from the Hub into compact per-flow shards.
#    Only ~130 of 1485 parquet columns are read, so nothing is staged on disk.
python tools/prepare_packets.py --files 1 2 3 --out data/shards

# 2. Check the config fits in device memory BEFORE a long run (~1 min).
python tools/preflight.py configs/large_300m.yaml --batch 256

# 3. Train.
python train.py --model-config configs/large_300m.yaml \
    --train-config configs/train_300m.yaml \
    --packet-shard data/shards/File_* --out checkpoints/anomaly
```

Data source: the `Payload-Bytes` subset of
[`rdpahalavan/UNSW-NB15`](https://huggingface.co/datasets/rdpahalavan/UNSW-NB15)
— per-packet payload bytes extracted from the original UNSW-NB15 PCAPs,
18 files, 47.9 GB, **1,593,481 flows**.

On RunPod, [`runpod/run_all.sh`](runpod/) does environment, preprocessing,
preflight and training in one command.

### Configurations

| config | params | notes |
|---|---|---|
| `base_124m.yaml` | 115.6M | fastest to train |
| `large_300m.yaml` | **306.1M** | the released checkpoint |
| `xl_500m.yaml` | 517.2M | needs a lower LR than the others |

All use `dtype: bfloat16` with `param_dtype: float32`. This is not optional —
some XLA:GPU versions expand float32 compilation of this model ~180×, requesting
193 GiB (and 2.02 TiB at batch 256) for a graph that needs 2.8 GB. Details in
[`logs/training_runs.md`](logs/training_runs.md#xlagpu-float32-memory-blowup).

## Layout

```
handler.py            HuggingFace Inference Endpoint handler; also a CLI
train.py              training entry point (packet-byte and legacy CSV paths)
eval.py               evaluation
models/               config, Flax model, inference helpers
data/
  packets.py          real payload-byte pipeline: parquet -> per-flow tensors
  pcap.py             pcap -> flow tensors (inference)
  tokenizer.py        byte tokenization; legacy tabular loader
diagnostics/          deterministic mismatch detection, root cause, remediation
optimization/         optimizer, checkpointing, bin-packed batching
tools/
  prepare_packets.py  stream Payload-Bytes parquet -> shards
  preflight.py        compile on the target device, report real memory use
  count_params.py     parameter counts without allocating them
runpod/               end-to-end GPU training scripts
examples/             request/response examples
logs/                 training runs, including the failures and why they failed
```

## Design notes

**Tokenization is shared between training and inference.** `data/packets.py`
emits `BYTE_OFFSET + byte` with `PAD_ID` padding, byte-identical to
`data/pcap.py`, enforced by a test. An earlier version trained on hashed proxies
and served real bytes — a train/serve mismatch that made the deployed model
meaningless regardless of its training.

**`proto_vocab` is fixed at 132**, not derived from data. `data/pcap.py` maps IP
protocol numbers with `% proto_vocab` at inference, so a data-derived vocabulary
would give the protocol embedding a different meaning at serve time.

**Class weighting is on by default.** Real traffic is ~3.6% attack; without
inverse-frequency weights the model reaches a low loss by always answering
"normal" — a collapsed head with a healthy-looking loss curve. `train.py` prints
`score p05-p95` every epoch for exactly this reason: a working detector shows a
wide spread, a collapsed one shows a narrow band.

**Flows, not packets, are split** between train and validation, so packets from
one flow can never straddle the boundary.

## Tests

```bash
pytest tests/ -q        # 55 tests
```

Covers tokenization parity with the inference path, flow assembly and
truncation, checkpoint round-trips, diagnostics, and the model.

## License

Apache 2.0 — see [LICENSE](LICENSE). Copyright 2026 Anvyon LLP.

The UNSW-NB15 dataset is the work of the Australian Centre for Cyber Security
and carries its own terms; cite Moustafa & Slay (2015) if you use it.
