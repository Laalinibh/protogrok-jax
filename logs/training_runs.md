# Training runs

Verbatim logs from every run, including the failures. Kept because the failures
are the useful part: three of them explain design decisions that would otherwise
look arbitrary.

---

## Run 0 — tabular CSV path (the original approach, superseded)

```
parameters: 115.6M | train 4000 | proto_vocab 107
epoch 1 | train_loss 0.8328 | 779.4s | val_acc 0.7471
epoch 2 | train_loss 0.6332 | 754.5s | val_acc 0.7471
Saved checkpoint to /tmp/ckpt_unsw_real
accuracy=0.7500 | macro_f1=0.7468 | n=1000
```

Validation accuracy is **identical across both epochs** — the model stopped
learning immediately.

Root cause: `data.tokenizer.load_unsw_nb15` builds payload tensors by hashing
each CSV row (`hash_bytes_from_fields`). Measured on two flows differing only in
duration (0.100 s vs 0.101 s):

```
near-identical flows -> payload correlation : -0.101
identical bytes                             : 0 / 128
same row twice       -> payload correlation : 1.0
```

Zero correlation, not one shared byte. The payload encoder — ~99% of the
parameters — was consuming reproducible noise. UNSW-NB15 also has no
`sport`/`dsport` columns, so `port_bucket()` returned a constant, leaving the
model two real features out of 42.

Measured ceiling of that pipeline, via gradient-boosted trees on the full
175k-row partition:

| input | accuracy | macro F1 | ROC-AUC |
|---|---|---|---|
| all real UNSW features | 0.8728 | 0.8672 | **0.9835** |
| only `[proto, npkts<=16]` (what the model saw) | 0.6630 | 0.5864 | — |
| majority class | 0.5506 | — | — |

The GBDT was a measuring stick only — never a deliverable model.

---

## Run 1 — real packet bytes, pipeline validation (local, CPU)

0.8M-parameter model, one shard (89,337 flows), to prove the new pipeline learns.

```
parameters: 0.8M | train 80,403 flows | val 8,934 flows
task: anomaly | attack rate train 0.071 val 0.067
class weights: 0:0.14, 1:1.86
epoch 1 | train_loss 0.3497 | 191.5s | acc 0.9400 | macro_f1 0.8220 | auc 0.9689 | score p05-p95 0.026-0.970
epoch 2 | train_loss 0.1901 | 225.9s | acc 0.9609 | macro_f1 0.8732 | auc 0.9736 | score p05-p95 0.012-0.967
epoch 3 | train_loss 0.1488 | 206.3s | acc 0.9665 | macro_f1 0.8883 | auc 0.9843 | score p05-p95 0.004-0.950
epoch 4 | train_loss 0.1188 | 202.6s | acc 0.9710 | macro_f1 0.9006 | auc 0.9853 | score p05-p95 0.005-0.979
epoch 5 | train_loss 0.0995 | 203.2s | acc 0.9457 | macro_f1 0.8390 | auc 0.9858 | score p05-p95 0.003-0.974
epoch 6 | train_loss 0.0865 | 189.7s | acc 0.9715 | macro_f1 0.9033 | auc 0.9880 | score p05-p95 0.003-0.976
```

**AUC 0.9880 from a 0.8M model on 1/18th of the data**, beating the GBDT ceiling
of 0.9835. The architecture was never the problem — its input was.

Note epoch 5: accuracy fell (0.9710 → 0.9457) while AUC *rose*. That is threshold
drift, not degradation, and it is why AUC is the metric of record here.

---

## Run 2 — 517M, learning rate too high (abandoned at epoch 3)

H100 80GB, `xl_500m.yaml` (517,221,903 params), `peak_lr: 6e-4`, batch 256.

```
parameters: 517.2M | train 1,434,132 flows | val 159,349 flows
task: anomaly | attack rate train 0.036 val 0.036
class weights: 0:0.07, 1:1.93
epoch 1 | train_loss 0.5336 | 583.9s | acc 0.8792 | macro_f1 0.6075 | auc 0.8348 | score p05-p95 0.301-0.926
epoch 2 | train_loss 0.5177 | 475.9s | acc 0.9324 | macro_f1 0.6831 | auc 0.8048 | score p05-p95 0.279-0.926
epoch 3 | train_loss 0.4978 | 477.9s | acc 0.9030 | macro_f1 0.6390 | auc 0.8385 | score p05-p95 0.239-0.863
```

AUC oscillates (0.835 → 0.805 → 0.839) while loss crawls — a learning rate too
high for the model width. Stopped rather than spend another hour.

---

## Run 3 — 306M, corrected learning rate (the released checkpoint)

H100 80GB, `large_300m.yaml` (306,103,367 params), `peak_lr: 2.5e-4`,
`warmup_steps: 1500`, batch 256, bfloat16 compute / float32 params.

```
parameters: 306.1M | train 1,434,132 flows | val 159,349 flows
task: anomaly | attack rate train 0.036 val 0.036
class balance: {'normal': 1535664, 'analysis': 435, 'backdoor': 401, 'dos': 3450,
                'exploits': 25098, 'fuzzers': 11638, 'generic': 3552,
                'reconnaissance': 11578, 'shellcode': 1510, 'worms': 155}
class weights: 0:0.07, 1:1.93
epoch 1 | train_loss 0.3438 | 439.6s | acc 0.7554 | macro_f1 0.5393 | auc 0.9458 | score p05-p95 0.007-0.969  [saved: best 0.9458]
```

Halving the learning rate moved epoch-1 AUC from 0.8348 to **0.9458** and opened
the score spread from `0.301–0.926` to `0.007–0.969`.

This is the published checkpoint. It is **one epoch** — training was interrupted
by a full disk, not by convergence.

---

## Preprocessing

Identical output across three independent runs on two regions:

```
flows            : 89,337 per file (mean 10.2 packets/flow after the 16-packet cap)
total flows      : 1,593,481
shard size       : ~178 MB per file, ~3.2 GB total (from 47.9 GB of parquet)
```

---

## Preflight (H100 80GB, 517M @ batch 256)

```
jax 0.10.2 | jaxlib 0.10.2 | backend gpu
devices: [CudaDevice(id=0)]
device memory: 63.8 GB

conv_impl  remat       params       temp       args      total  fits?
conv       False       517.2M      9.59G      6.21G     15.79G  yes
conv       True        517.2M      4.72G      6.21G     10.93G  yes
matmul     False       517.2M     12.27G      6.21G     18.48G  yes
matmul     True        517.2M      8.72G      6.21G     14.93G  yes
```

`remat` halves activation memory on GPU (9.59 → 4.72 GB). On the CPU backend it
does nothing (20.13 → 20.41 GB) — the two backends schedule very differently,
which is why `tools/preflight.py` measures on the target device.

---

## XLA:GPU float32 memory blowup

Three runs, one variable:

| config | compute dtype | outcome |
|---|---|---|
| `base_124m`, batch 32 | float32 | OOM, **193.78 GiB** requested |
| `large_300m`, batch 256 | float32 | OOM, **2.02 TiB** requested |
| `xl_500m`, batch 256 | bfloat16 | compiled at 15.79 GB, trained fine |

The same `train_step` compiles to **2.8 GB** on the CPU backend at batch 32, so
the model never needed that memory — some XLA:GPU versions expand it ~180× under
float32. All shipped configs therefore use `dtype: bfloat16` with
`param_dtype: float32`.

`tools/preflight.py` exists to catch this in one minute rather than hours in.
