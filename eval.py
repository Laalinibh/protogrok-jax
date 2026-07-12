#!/usr/bin/env python
"""Evaluation suite and convergence checks for Protogrok-JAX.

Restores a checkpoint and reports accuracy / macro-F1 on a labelled CSV, or
runs anomaly detection over a PCAP file.

Examples
--------
    python eval.py --ckpt checkpoints/anomaly \
        --test-csv UNSW_NB15_testing-set.csv

    python eval.py --ckpt checkpoints/anomaly --pcap capture.pcap
"""
from __future__ import annotations

import argparse
import json

import jax

from models.config import ProtogrokConfig
from models.protogrok import init_params
from models.inference import evaluate
from data.tokenizer import load_unsw_nb15
from optimization import checkpointing


def _restore(ckpt: str):
    with open(f"{ckpt}/config.json") as f:
        cfg = ProtogrokConfig.from_dict(json.load(f)["config"])
    _, variables = init_params(cfg, jax.random.PRNGKey(0))
    params, cfg = checkpointing.restore(ckpt, params_template=variables["params"])
    return cfg, params


def main() -> None:
    p = argparse.ArgumentParser(description="Protogrok-JAX evaluation")
    p.add_argument("--ckpt", required=True, help="checkpoint directory")
    p.add_argument("--test-csv", default=None)
    p.add_argument("--pcap", default=None)
    p.add_argument("--task", default="anomaly", choices=["anomaly", "class"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num-classes", type=int, default=2)
    args = p.parse_args()

    cfg, params = _restore(args.ckpt)
    print(f"Restored {cfg.d_model}-d model from {args.ckpt}")

    if args.test_csv:
        _, test, meta = load_unsw_nb15(args.test_csv, None)
        metrics = evaluate(cfg, params, test, task=args.task,
                           batch_size=args.batch, num_classes=args.num_classes)
        print(f"accuracy={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} "
              f"| n={metrics['n']}")

    if args.pcap:
        from data.pcap import evaluate_pcap
        results = evaluate_pcap(args.pcap, cfg, params)
        for r in results:
            tag = "[ANOMALY]" if r["anomaly"] else "[NORMAL] "
            print(f"{tag} score={r['score']:.4f} | {r['flow']} | packets={r['packets']}")


if __name__ == "__main__":
    main()
