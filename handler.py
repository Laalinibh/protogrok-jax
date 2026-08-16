"""Custom Hugging Face Inference Endpoint handler for Protogrok-JAX.

Deploy layout
-------------
Push this whole repository (models/, data/, diagnostics/, optimization/,
handler.py, requirements.txt) to a Hugging Face model repo, PLUS a trained
checkpoint's ``config.json`` and ``params.msgpack`` at the repo root (i.e.
copy the contents of your ``checkpointing.save(...)`` output directory into
the repo root, or set ``CKPT_SUBDIR`` below if you'd rather keep them in a
subdirectory such as ``checkpoint/``).

    your-hf-repo/
    ├── handler.py            <- this file
    ├── requirements.txt
    ├── config.json           <- from checkpointing.save()
    ├── params.msgpack        <- from checkpointing.save()
    ├── models/
    ├── data/
    ├── diagnostics/
    └── optimization/

Then create a Hugging Face Inference Endpoint (or run locally with
``huggingface_hub.InferenceClient`` against a custom container) pointing at
that repo. HF's endpoint runtime instantiates ``EndpointHandler(path)`` once
at startup (``path`` = the local snapshot dir of the repo) and calls
``handler(data)`` per request.

Request format
---------------
``data["inputs"]`` may be, in order of preference:

1. Raw ``.pcap``/``.pcapng`` bytes (e.g. posted as
   ``application/octet-stream``, or a multipart file upload that the client
   forwards as bytes).
2. A base64-encoded string of the same.
3. ``{"pcap_base64": "<base64 .pcap bytes>"}``.

Optional ``data["parameters"]``:

- ``threshold`` (float, default 0.5) — anomaly-probability cutoff.
- ``diagnose`` (bool, default True) — include mismatch findings / root
  cause / suggested fixes, not just the raw anomaly score.
- ``max_flows`` (int, default 200) — cap on how many flow reports to return
  (protects the response size on very large captures).

Response
--------
``{"model": {...}, "n_flows": int, "results": [ {...per-flow report...}, ... ]}``

Each per-flow report (when ``diagnose`` is true) has the shape produced by
``diagnostics.report.diagnose_flow``: ``flow``, ``packets``,
``anomaly_score``, ``anomaly``, ``mismatch_findings``, ``mismatch_detected``,
``root_cause`` (tag / label / confidence / reasoning), and
``suggested_actions``. With ``diagnose=False`` you get just
``flow``/``score``/``anomaly``/``packets`` (the plain anomaly-scoring path).
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

# Make the repo root importable regardless of the runtime's cwd.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CKPT_SUBDIR = ""  # set to e.g. "checkpoint" if config.json/params.msgpack
                  # live in a subdirectory instead of the repo root.


class EndpointHandler:
    def __init__(self, path: str = "") -> None:
        import jax

        from models.config import ProtogrokConfig
        from models.protogrok import init_params
        from optimization import checkpointing

        ckpt_dir = os.path.join(path or _REPO_ROOT, CKPT_SUBDIR) if CKPT_SUBDIR else (path or _REPO_ROOT)
        if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
            raise FileNotFoundError(
                f"No config.json found under '{ckpt_dir}'. Copy your "
                f"checkpointing.save(...) output (config.json + "
                f"params.msgpack) into the repo root, or set CKPT_SUBDIR in "
                f"handler.py to match where you put them.")

        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg = ProtogrokConfig.from_dict(json.load(f)["config"])
        _, variables = init_params(cfg, jax.random.PRNGKey(0))
        params, cfg = checkpointing.restore(ckpt_dir, params_template=variables["params"])

        self.cfg = cfg
        self.params = params
        self._n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    # ------------------------------------------------------------------ #
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        inputs = data.get("inputs", data)
        parameters = data.get("parameters", {}) or {}
        threshold = float(parameters.get("threshold", 0.5))
        diagnose = bool(parameters.get("diagnose", True))
        max_flows = int(parameters.get("max_flows", 200))

        pcap_bytes = self._extract_pcap_bytes(inputs)
        if pcap_bytes is None:
            return {"error": (
                "Could not find pcap bytes in the request. Send raw "
                "pcap/pcapng bytes as `inputs`, a base64 string, or "
                "{'pcap_base64': '...'}.")}

        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
            tmp.write(pcap_bytes)
            tmp_path = tmp.name

        try:
            if diagnose:
                from diagnostics.report import diagnose_pcap
                results = diagnose_pcap(tmp_path, self.cfg, self.params, threshold=threshold)
            else:
                from data.pcap import evaluate_pcap
                results = evaluate_pcap(tmp_path, self.cfg, self.params, threshold=threshold)
        except Exception as exc:  # surface a clean error instead of a 500 traceback
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            os.unlink(tmp_path)

        truncated = len(results) > max_flows
        return {
            "model": {"d_model": self.cfg.d_model, "params": self._n_params},
            "n_flows": len(results),
            "truncated": truncated,
            "results": results[:max_flows],
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_pcap_bytes(inputs: Any) -> Optional[bytes]:
        if isinstance(inputs, (bytes, bytearray)):
            return bytes(inputs)
        if isinstance(inputs, dict):
            b64 = inputs.get("pcap_base64") or inputs.get("pcap")
            if isinstance(b64, str):
                return EndpointHandler._maybe_b64_decode(b64)
            if isinstance(b64, (bytes, bytearray)):
                return bytes(b64)
            return None
        if isinstance(inputs, str):
            return EndpointHandler._maybe_b64_decode(inputs)
        return None

    @staticmethod
    def _maybe_b64_decode(s: str) -> Optional[bytes]:
        try:
            return base64.b64decode(s, validate=True)
        except (binascii.Error, ValueError):
            return None


# ---------------------------------------------------------------------- #
# Local smoke test: `python handler.py /path/to/ckpt_dir /path/to/file.pcap`
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <checkpoint_dir> <pcap_path> [threshold]")
        raise SystemExit(1)
    ckpt_dir, pcap_path = sys.argv[1], sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    handler = EndpointHandler(ckpt_dir)
    with open(pcap_path, "rb") as f:
        raw = f.read()
    out = handler({"inputs": raw, "parameters": {"threshold": threshold}})
    print(json.dumps(out, indent=2))
