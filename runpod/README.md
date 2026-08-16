# Training Protogrok-JAX on RunPod

End-to-end recipe for training on the **real UNSW-NB15 packet-byte corpus**
(2.06M flows / 45M packets), rather than the tabular CSV partition.

## Why not the CSV path

`data/tokenizer.load_unsw_nb15` builds payload tensors by hashing each row
(`hash_bytes_from_fields`). Two flows differing only in duration get payloads
with **zero** correlation and no shared bytes, so the payload encoder — which
is ~99% of the parameters — has nothing to generalise from. UNSW-NB15 also has
no `sport`/`dsport` columns, so both port buckets are constant. Net effect: the
model sees two real features (protocol, packet count) out of 42, and is then
served *real* bytes at inference by `data/pcap.py` — a train/serve mismatch on
top of the data problem.

Measured ceiling on that path: ~0.66 accuracy. Gradient-boosted trees on the
same rows' real features reach AUC 0.9835.

## Pipeline

```
HuggingFace rdpahalavan/UNSW-NB15  (Payload-Bytes: 18 files x 2.7 GB)
        |  tools/prepare_packets.py   -- streams ~130 of 1485 columns
        v
FlowShard   payload uint8 [N,16,128] + lengths + headers + labels  (~3.2 GB)
        |  train.py --packet-shard
        v
checkpoint  config.json + params.msgpack   -> handler.py / HF endpoint
```

Only `payload_byte_1..128` of 1476 are read. Parquet is columnar, so this costs
roughly 9% of each file and **nothing is staged on disk**.

## Pod spec

| | |
|---|---|
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (any CUDA 12.x; Hopper needs ≥ 11.8) |
| GPU | H100 80GB **SXM** preferred → H100 PCIe → A100 80GB. All use `batch_size: 256`; 40 GB cards drop to 128 |
| Volume | network volume at `/workspace`, ≥ 50 GB |
| RAM | ≥ 32 GB — preprocessing holds one 2.5M-row Arrow row group (~3 GB) |

## Two-phase run (saves GPU rental)

Preprocessing uses **no GPU** — it is network + CPU bound. Doing it on an H100
wastes roughly $2–3. Because the network volume outlives the pod, split it:

```bash
# --- Phase A: cheapest available pod (CPU-only or RTX 4090), volume attached
cd /workspace && runpodctl receive <code> && tar xzf protogrok-code.tar.gz
MODE=cpu bash repo/runpod/setup.sh
source /workspace/venv/bin/activate
bash repo/runpod/preprocess_all.sh    # ~30-60 min, resumable
# then TERMINATE this pod

# --- Phase B: H100 pod, same volume attached
cd /workspace && bash repo/runpod/setup.sh    # asserts the GPU is visible
source /workspace/venv/bin/activate
tmux new -s train 'bash repo/runpod/train_xl.sh 2>&1 | tee /workspace/logs/train.log'
```

Shards are already on the volume, so Phase B starts training immediately. It
also de-risks: a download that stalls at file 14 fails on a cheap box.

Running both phases on the H100 is fine too — it costs a few dollars more.

Copy the checkpoint off **before terminating** — `runpodctl send`, scp, or push
to a HF model repo. `/workspace` persists only if it is a network volume.

## Expected cost and time

Measured FLOPs/sample: 21.8 / 48.3 / **79.3 GFLOP** for 124M / 306M / 500M
(forward+backward). At ~30% MFU on 1.85M training flows, using **dense** BF16
throughput (not the sparsity-inflated headline numbers):

| GPU | dense BF16 | 500M per epoch | 500M × 10 epochs |
|---|---|---|---|
| A100 80GB SXM | ~312 TFLOPS | ~26 min | ~4.3 h |
| H100 80GB SXM | ~495 TFLOPS | ~17 min | ~2.8 h |
| H100 80GB PCIe | ~378 TFLOPS | ~22 min | ~3.6 h |

H100 is ~1.6× an A100, not 2×. At roughly $2.40–3.50/hr for H100 — **verify
current rates** — a 10-epoch run is about **$7–13**. Budget **$40–80** overall
for 2–3 runs while tuning. Phase A adds ~$1–2 on a cheap pod.

## Gotchas

- **The loss plateaus before it learns.** On a synthetic byte-signal task the
  model sat at AUC ~0.62 for ~560 steps, then went to 1.000 within two epochs.
  A flat start is expected — do not kill the run or lower the LR. The original
  2-epoch CSV run never escaped this plateau.
- **Watch score spread, not just loss.** `train.py` prints `score p05-p95`.
  A collapsed head shows a healthy loss with a spread like `0.518-0.519`;
  a working detector opens toward `0.00-1.00`. This is the single most
  diagnostic number in the log.
- **Real traffic is ~5% attack.** Accuracy is meaningless at that ratio
  (95% by always answering "normal"). Inverse-frequency class weighting is on
  by default; `--no-class-weights` disables it. Judge runs by **AUC** and
  **macro-F1**.
- **Rare classes are very rare.** File 1 has 17 worms and 18 analysis flows out
  of 89,337. For `task: class`, expect those to stay weak regardless of scale;
  consider merging them or reporting per-class recall.
- **`max_packets=16` discards packets.** Real flows average 28 packets; the cap
  keeps the first 16. `payload_max_len=128` keeps 128 bytes of a median 1300-byte
  payload. Both are tunable in `models/config.py`, but changing either changes
  the model input shape — retrain, don't fine-tune, and keep `data/pcap.py` in
  sync or you reintroduce train/serve skew.
- **`proto_vocab` must stay fixed at 132.** `data/pcap.py` maps the IP protocol
  number with `% proto_vocab` at inference; deriving it from data (as the CSV
  path does) gives the protocol embedding a different meaning at serve time.
