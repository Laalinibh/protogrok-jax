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
    p.add_argument("--diagnose", action="store_true",
                   help="with --pcap: also run mismatch detection, root-cause "
                        "analysis, and suggested fixes (see diagnostics/)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--json", action="store_true",
                   help="print --diagnose output as JSON instead of text")
    args = p.parse_args()

    cfg, params = _restore(args.ckpt)
    print(f"Restored {cfg.d_model}-d model from {args.ckpt}")

    if args.test_csv:
        # load_unsw_nb15(csv, None) loads `csv` into the *first* (train) slot
        # of its return tuple and leaves the second (test) slot as None --
        # take the first slot here. (Pre-existing bug: this previously did
        # `_, test, meta = load_unsw_nb15(args.test_csv, None)`, which always
        # unpacked `test` as None and crashed in `evaluate()`.)
        test, _, meta = load_unsw_nb15(args.test_csv, None)
        metrics = evaluate(cfg, params, test, task=args.task,
                           batch_size=args.batch, num_classes=args.num_classes)
        print(f"accuracy={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} "
              f"| n={metrics['n']}")

    if args.pcap and args.diagnose:
        from diagnostics.report import diagnose_pcap
        reports = diagnose_pcap(args.pcap, cfg, params, threshold=args.threshold)
        if args.json:
            print(json.dumps(reports, indent=2))
        else:
            for r in reports:
                tag = "[ANOMALY]" if r["anomaly"] else "[NORMAL] "
                print(f"\n{tag} score={r['anomaly_score']:.4f} | {r['flow']} | "
                      f"packets={r['packets']}")
                cause = r["root_cause"]
                print(f"  root cause : {cause['label']} "
                      f"(tag={cause['tag']}, confidence={cause['confidence']:.2f})")
                for line in cause["reasoning"]:
                    print(f"    reason   : {line}")
                if r["mismatch_findings"]:
                    print(f"  mismatches : {len(r['mismatch_findings'])} finding(s)")
                    for f in r["mismatch_findings"]:
                        print(f"    [{f['severity']}] {f['code']}: {f['message']}")
                else:
                    print("  mismatches : none")
                print("  suggested actions:")
                for a in r["suggested_actions"]:
                    print(f"    - {a}")
    elif args.pcap:
        from data.pcap import evaluate_pcap
        results = evaluate_pcap(args.pcap, cfg, params)
        for r in results:
            tag = "[ANOMALY]" if r["anomaly"] else "[NORMAL] "
            print(f"{tag} score={r['score']:.4f} | {r['flow']} | packets={r['packets']}")


if __name__ == "__main__":
    main()
