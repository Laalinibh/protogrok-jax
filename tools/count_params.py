#!/usr/bin/env python
"""Report parameter counts for a model config without allocating them.

Uses ``jax.eval_shape``, so a 500M-parameter config costs no memory to inspect.

    python tools/count_params.py configs/xl_500m.yaml
    python tools/count_params.py configs/*.yaml
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from models.config import ProtogrokConfig  # noqa: E402
from models.protogrok import ProtogrokModel  # noqa: E402


def param_shapes(cfg: ProtogrokConfig):
    model = ProtogrokModel(cfg)
    L, T = cfg.payload_max_len, cfg.max_packets
    args = (jnp.zeros((2, T, L), jnp.int32), jnp.zeros((2, 5), jnp.int32),
            jnp.zeros((2,), jnp.int32))
    return jax.eval_shape(
        lambda: model.init(jax.random.PRNGKey(0), *args,
                           task="anomaly", deterministic=True))["params"]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for path in sys.argv[1:]:
        cfg = ProtogrokConfig.from_yaml(path)
        shapes = param_shapes(cfg)
        leaves = jax.tree_util.tree_leaves_with_path(shapes)
        total = sum(int(v.size) for _, v in leaves)

        groups: dict = {}
        for path_, v in leaves:
            top = path_[0].key if hasattr(path_[0], "key") else str(path_[0])
            groups[top] = groups.get(top, 0) + int(v.size)
        blocks = sum(n for k, n in groups.items() if k.startswith(("pblock", "sblock")))

        print(f"\n{os.path.basename(path)}: d_model={cfg.d_model} "
              f"layers={cfg.packet_layers}+{cfg.session_layers} heads={cfg.nhead}")
        print(f"  total parameters : {total/1e6:.1f}M  ({total:,})")
        print(f"  transformer blocks: {blocks/1e6:.1f}M "
              f"({100*blocks/total:.0f}%)")
        for k in sorted(groups, key=lambda k: -groups[k]):
            if not k.startswith(("pblock", "sblock")) and groups[k] > 10_000:
                print(f"  {k:22} {groups[k]/1e6:7.2f}M")
        gb = total * 4 / 1e9
        print(f"  fp32 weights {gb:.2f}GB | +Adam m,v {3*gb:.2f}GB "
              f"| +grads {4*gb:.2f}GB total")


if __name__ == "__main__":
    main()
