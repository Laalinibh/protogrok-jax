# Results

All figures below are measured, with the command that produced them. Nothing is
estimated or carried over from a paper.

## Released checkpoint

[`Anvyon/protogrok-jax-306m`](https://huggingface.co/Anvyon/protogrok-jax-306m)

| | |
|---|---|
| Parameters | **306,103,367** |
| Training data | 1,434,132 UNSW-NB15 flows, real payload bytes |
| Validation | 159,349 held-out flows (split by flow, never by packet) |
| **ROC-AUC** | **0.9458** |
| Accuracy | 0.7554 |
| Macro F1 | 0.5393 |
| Epochs | 1 (5,602 steps) |
| Hardware | 1× H100 80GB, 439.6 s/epoch |

Accuracy is far below AUC by construction: inverse-frequency class weights
(`0.07 / 1.93`) bias the model toward recall on the 3.6% attack class, which
moves the optimal threshold well away from 0.5. AUC is threshold-free and is the
number to judge this model by.

## Comparison

| model | trained on | ROC-AUC |
|---|---|---|
| GBDT on all 42 tabular UNSW features | flow statistics | 0.9835 |
| Protogrok 0.8M, 1 shard, 6 epochs | real payload bytes | **0.9880** |
| **Protogrok 306M, 18 shards, 1 epoch** | real payload bytes | **0.9458** |
| Protogrok 517M, 18 shards, 3 epochs (lr too high) | real payload bytes | 0.8385 |
| Protogrok 115.6M on hashed row proxies | — | ceiling ~0.66 |

The 0.8M result is the important one: it shows the architecture reaches the
data's ceiling once fed real bytes. The 306M checkpoint is under-trained (one
epoch), not under-capacity.

## Known limitation — out-of-distribution saturation

Scored with `handler.py` on captures unlike the training distribution:

| capture | min | median | max | flagged |
|---|---|---|---|---|
| `crafted_attack_trace.pcap` | 0.9627 | 0.9714 | 0.9933 | 41/41 |
| `crafted_benign_and_scan_trace.pcap` | 0.9627 | 0.9715 | 0.9933 | 34/34 |
| `live_captured_real_trace.pcap` | 0.6797 | 0.9570 | 0.9922 | 24/24 |

The attack and benign distributions are **identical**. On these captures the
model does not separate the two, and at a 0.5 threshold it flags everything.

It ranks correctly on UNSW-like traffic (AUC 0.9458) but saturates off-
distribution. **Do not use the score as a calibrated probability.** Two fixes,
in order of impact: train past one epoch, and pick the operating point from a
validation ROC on traffic representative of the deployment.

## The structural diagnostics are independent of the model

`diagnostics/` is deterministic rule-based analysis and is reliable regardless
of the model's state. On `crafted_attack_trace.pcap`:

```
 20  syn_flood_dos
 15  port_scan_reconnaissance
  2  dns_tunneling_or_amplification
  1  scanner_or_evasion_tooling      (SYN+FIN illegal flag combination)
  1  packet_corruption_or_tampering  (IP checksum mismatch)
```

Each finding carries the evidence that triggered it and suggested remediation.

**A caveat on checksum findings.** On `live_captured_real_trace.pcap` every
TCP/UDP packet fails its L4 checksum — including inbound packets from public
hosts, which a kernel would have dropped. Every IP checksum passes, and the
local endpoint is `192.0.2.2` (RFC 5737 documentation range). That capture has
been sanitised without recomputing L4 checksums, so the resulting
`packet_corruption_or_tampering` verdict is an artifact of the fixture, not a
finding about a network. The checksum code is correct; the input is not a
faithful capture.

## Throughput

Forward+backward FLOPs per sample: **21.8 / 48.3 / 79.3 GFLOP** for 124M / 306M
/ 500M. Measured, then validated against observed step times.

| model | H100 80GB, batch 256 | Apple M3 CPU, batch 32 |
|---|---|---|
| 115.6M | — | 2.45 s/step |
| 306.1M | 439.6 s/epoch (5,602 steps) | 8.2 s/step |
| 517.2M | 476–584 s/epoch | ~65 s/step (swap-bound) |

Preprocessing all 18 Payload-Bytes files: ~20 min at `JOBS=4`, ~50 min at
`JOBS=1`. Output is deterministic — `total flows: 1,593,481` across three
independent runs in two regions.

## Reproducing

```bash
python tools/prepare_packets.py --files 1 --out data/shards      # stream one file
python tools/preflight.py configs/large_300m.yaml --batch 256    # check it fits
python train.py --model-config configs/large_300m.yaml \
    --train-config configs/train_300m.yaml \
    --packet-shard data/shards/File_1 --out checkpoints/anomaly
```

On RunPod, `runpod/run_all.sh` does all of it end to end.
