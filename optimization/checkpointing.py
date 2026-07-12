"""Checkpointing: params (msgpack via flax.serialization) + config (JSON).

Uses ``flax.serialization`` — the stable, framework-native way to persist a
Flax parameter pytree — so checkpoints are portable and version-robust. A
checkpoint directory contains ``params.msgpack`` and ``config.json``; the
config makes it self-describing (no need to re-specify hyper-parameters).

For large-scale / sharded training an Orbax backend can be slotted in behind
the same ``save`` / ``restore`` interface; see :func:`save_orbax`.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional, Tuple

from flax import serialization

from models.config import ProtogrokConfig


def save(path: str, params: Any, cfg: ProtogrokConfig, step: Optional[int] = None) -> str:
    """Save params + config to directory ``path``. Returns the directory."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "params.msgpack"), "wb") as f:
        f.write(serialization.to_bytes(params))
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "step": step}, f, indent=2)
    return path


def restore(path: str, params_template: Any) -> Tuple[Any, ProtogrokConfig]:
    """Restore (params, config). ``params_template`` is a pytree of the correct
    structure/shapes (e.g. from a fresh ``model.init``)."""
    with open(os.path.join(path, "config.json")) as f:
        cfg = ProtogrokConfig.from_dict(json.load(f)["config"])
    with open(os.path.join(path, "params.msgpack"), "rb") as f:
        params = serialization.from_bytes(params_template, f.read())
    return params, cfg


def save_orbax(path: str, params: Any, cfg: ProtogrokConfig,
               step: Optional[int] = None) -> str:
    """Optional Orbax backend for sharded / async checkpointing at scale."""
    import orbax.checkpoint as ocp
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "step": step}, f, indent=2)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(os.path.abspath(os.path.join(path, "params")),
               args=ocp.args.StandardSave(params))
    ckptr.wait_until_finished()
    return path
