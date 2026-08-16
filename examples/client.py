#!/usr/bin/env python
"""Call a deployed Protogrok endpoint with a .pcap file.

The token is read from the credentials `huggingface-cli login` already stored,
so it never appears in this file, in shell history, or in a process listing.

    python examples/client.py <endpoint-url> <capture.pcap> [threshold]
"""
from __future__ import annotations

import base64
import json
import sys
import time

import requests
from huggingface_hub import get_token


def analyze(url: str, pcap_path: str, threshold: float = 0.5) -> dict:
    token = get_token()
    if not token:
        raise SystemExit("No HF token found. Run: huggingface-cli login")

    with open(pcap_path, "rb") as f:
        raw = f.read()

    payload = {"inputs": base64.b64encode(raw).decode(),
               "parameters": {"threshold": threshold, "diagnose": True,
                              "max_flows": 200}}
    resp = requests.post(url.rstrip("/"),
                         headers={"Authorization": f"Bearer {token}",
                                  "Content-Type": "application/json"},
                         json=payload, timeout=600)

    if resp.status_code == 503:
        raise SystemExit("503 — endpoint is cold-starting (scale-to-zero). "
                         "Retry in ~60s.")
    resp.raise_for_status()
    out = resp.json()
    if "error" in out:
        raise SystemExit(f"endpoint error: {out['error']}")
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    url, pcap = sys.argv[1], sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    t0 = time.time()
    out = analyze(url, pcap, threshold)
    results = out.get("results", [])
    scores = [r["anomaly_score"] for r in results]

    print(f"responded in {time.time()-t0:.1f}s")
    print(f"model  : {out.get('model')}")
    print(f"flows  : {out.get('n_flows')} (truncated: {out.get('truncated')})")
    if scores:
        print(f"scores : {min(scores):.4f} - {max(scores):.4f}")
        print(f"flagged: {sum(1 for r in results if r['anomaly'])}/{len(results)}"
              f" at threshold {threshold}")

    print("\ntop flows:")
    for r in sorted(results, key=lambda r: -r["anomaly_score"])[:10]:
        cause = (r.get("root_cause") or {}).get("label", "-")
        print(f"  {r['anomaly_score']:.4f}  {r['flow'][:52]:52}  {cause}")

    with open("endpoint_response.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nfull response -> endpoint_response.json")


if __name__ == "__main__":
    main()
