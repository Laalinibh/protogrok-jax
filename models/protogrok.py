"""Protogrok model in Flax (linen).

A faithful, production-grade JAX port of the PyTorch ``ProtogrokModel``:
a two-stage (packet-level then session-level) Transformer over network
flows, with a convolutional payload encoder, a structured header encoder,
a protocol adapter, and a slot-attention "memory" module, feeding
multi-task heads (anomaly / traffic-class / byte-decode).

Shapes
------
payload_toks : int32 [B, T, L]   byte-token ids per packet
headers      : int32 [B, 5]      [proto_id, sport_bucket, dport_bucket, s0, s1]
proto_id     : int32 [B]         protocol id for the protocol adapter
returns      : float [B, C]      task logits (C depends on ``task``)
"""
from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from models.config import PAD_ID, ProtogrokConfig

Task = Literal["anomaly", "class", "protocol", "pooled"]


def _dtypes(cfg: ProtogrokConfig):
    return jnp.dtype(cfg.dtype), jnp.dtype(cfg.param_dtype)


class Conv1dMatmul(nn.Module):
    """``nn.Conv(kernel_size=(3,), padding="SAME")`` expressed as 3 matmuls.

    Parameter names and shapes are identical to ``nn.Conv`` (``kernel``
    ``[k, in, out]``, ``bias`` ``[out]``), so checkpoints interoperate freely
    with the ``nn.Conv`` implementation and either can be swapped in.

    Why this exists: on some XLA:GPU versions the 1D convolution over the
    ``[B*T, L, E]`` payload tensor triggers a catastrophic im2col-style
    expansion (observed: a 193 GiB buffer request for a model whose compiled
    footprint is 2.8 GB on CPU). Writing it as explicit shifted matmuls keeps
    the maths identical while bypassing convolution algorithm selection.
    """

    features: int
    kernel_size: int = 3
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:      # [N, L, C] -> [N, L, F]
        k, C, F = self.kernel_size, x.shape[-1], self.features
        kernel = self.param("kernel", nn.initializers.lecun_normal(),
                            (k, C, F), self.param_dtype).astype(self.dtype)
        bias = self.param("bias", nn.initializers.zeros_init(),
                          (F,), self.param_dtype).astype(self.dtype)
        # "SAME" for an odd kernel and stride 1 pads (k-1)//2 on each side.
        pad = (k - 1) // 2
        xp = jnp.pad(x, ((0, 0), (pad, pad), (0, 0)))
        L = x.shape[1]
        out = sum(xp[:, i:i + L, :] @ kernel[i] for i in range(k))
        return out + bias


class PayloadEncoder(nn.Module):
    """Per-packet byte encoder: embed -> +positional -> 2x Conv1d(GELU) -> avg-pool.

    PyTorch parity: ``nn.Embedding(padding_idx=PAD_ID)`` + learned positions +
    two ``Conv1d(k=3, pad=1)`` with GELU + ``AdaptiveAvgPool1d(1)`` (mean over L).
    """

    cfg: ProtogrokConfig

    @nn.compact
    def __call__(self, toks: jnp.ndarray) -> jnp.ndarray:  # [B, T, L] -> [B, T, E]
        cfg = self.cfg
        compute_dtype, param_dtype = _dtypes(cfg)
        B, T, L = toks.shape
        E = cfg.payload_dim

        embed = nn.Embed(
            num_embeddings=cfg.byte_vocab, features=E,
            embedding_init=nn.initializers.normal(stddev=0.02),
            param_dtype=param_dtype, name="emb",
        )
        pos = self.param("pos", nn.initializers.normal(stddev=1.0),
                         (cfg.payload_max_len, E), param_dtype)

        x = embed(toks.reshape(B * T, L)).astype(compute_dtype)          # [B*T, L, E]
        # Emulate padding_idx: zero the embedding at PAD positions (forward parity).
        pad_mask = (toks.reshape(B * T, L) != PAD_ID)[..., None].astype(compute_dtype)
        x = x * pad_mask
        x = x + pos[:L][None].astype(compute_dtype)                      # [B*T, L, E]

        if cfg.payload_conv_impl == "matmul":
            make_conv = lambda name: Conv1dMatmul(  # noqa: E731
                features=E, kernel_size=3, param_dtype=param_dtype,
                dtype=compute_dtype, name=name)
        else:
            make_conv = lambda name: nn.Conv(  # noqa: E731
                features=E, kernel_size=(3,), padding="SAME",
                param_dtype=param_dtype, dtype=compute_dtype, name=name)
        x = nn.gelu(make_conv("conv0")(x), approximate=False)
        x = nn.gelu(make_conv("conv1")(x), approximate=False)
        x = jnp.mean(x, axis=1)                                          # AdaptiveAvgPool1d(1)
        return x.reshape(B, T, E)


class HeaderEncoder(nn.Module):
    """Structured header encoder: proto/port embeddings + scalar projection."""

    cfg: ProtogrokConfig

    @nn.compact
    def __call__(self, headers: jnp.ndarray) -> jnp.ndarray:            # [B, 5] -> [B, H]
        cfg = self.cfg
        compute_dtype, param_dtype = _dtypes(cfg)
        H = cfg.header_dim
        half = H // 2

        proto = nn.Embed(cfg.proto_vocab, H, param_dtype=param_dtype, name="proto_emb")(headers[:, 0])
        port_emb = nn.Embed(4, half, param_dtype=param_dtype, name="port_emb")
        sport = port_emb(headers[:, 1])
        dport = port_emb(headers[:, 2])
        scal = nn.Dense(half, param_dtype=param_dtype, dtype=compute_dtype, name="scalar")(
            headers[:, 3:].astype(compute_dtype))
        x = jnp.concatenate([proto.astype(compute_dtype), sport.astype(compute_dtype),
                             dport.astype(compute_dtype), scal], axis=-1)
        return nn.Dense(H, param_dtype=param_dtype, dtype=compute_dtype, name="out")(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with full multi-head self-attention."""

    cfg: ProtogrokConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # `deterministic` is positional-or-keyword (not keyword-only) so this
        # block can be wrapped in nn.remat with static_argnums.
        cfg = self.cfg
        compute_dtype, param_dtype = _dtypes(cfg)
        hidden = cfg.d_model * cfg.mlp_ratio

        h = nn.LayerNorm(param_dtype=param_dtype, dtype=compute_dtype, name="norm1")(x)
        a = nn.MultiHeadDotProductAttention(
            num_heads=cfg.nhead, qkv_features=cfg.d_model,
            dropout_rate=cfg.dropout_rate, deterministic=deterministic,
            param_dtype=param_dtype, dtype=compute_dtype, name="attn",
        )(h)  # inputs_kv defaults to inputs_q -> self-attention
        x = x + nn.Dropout(cfg.dropout_rate, deterministic=deterministic)(a)

        h = nn.LayerNorm(param_dtype=param_dtype, dtype=compute_dtype, name="norm2")(x)
        h = nn.Dense(hidden, param_dtype=param_dtype, dtype=compute_dtype, name="mlp0")(h)
        h = nn.gelu(h, approximate=False)
        h = nn.Dense(cfg.d_model, param_dtype=param_dtype, dtype=compute_dtype, name="mlp1")(h)
        x = x + nn.Dropout(cfg.dropout_rate, deterministic=deterministic)(h)
        return x


class ProtocolAdapter(nn.Module):
    """Adds a protocol embedding and a bottleneck adapter (down-ReLU-up)."""

    cfg: ProtogrokConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, proto_id: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        compute_dtype, param_dtype = _dtypes(cfg)
        pe = nn.Embed(cfg.proto_vocab, cfg.d_model, param_dtype=param_dtype,
                      name="proto")(proto_id)[:, None, :].astype(compute_dtype)
        ad = nn.Dense(cfg.bottleneck, param_dtype=param_dtype, dtype=compute_dtype, name="down")(x)
        ad = nn.relu(ad)
        ad = nn.Dense(cfg.d_model, param_dtype=param_dtype, dtype=compute_dtype, name="up")(ad)
        return x + pe + ad


class MemoryModule(nn.Module):
    """Slot-attention session memory: pool packets into slots, broadcast back."""

    cfg: ProtogrokConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:                 # [B, T, D]
        cfg = self.cfg
        compute_dtype, param_dtype = _dtypes(cfg)
        D = cfg.d_model
        slots = self.param("slots", nn.initializers.normal(stddev=1.0),
                           (cfg.memory_slots, D), param_dtype).astype(compute_dtype)
        S = jnp.broadcast_to(slots[None], (x.shape[0],) + slots.shape)  # [B, S, D]
        att = jnp.einsum("bsd,btd->bst", S, x) / jnp.sqrt(D).astype(compute_dtype)
        w = jax.nn.softmax(att, axis=-1)
        ctx = jnp.einsum("bst,btd->bsd", w, x)                          # [B, S, D]
        session = jnp.mean(ctx, axis=1)                                 # [B, D]
        y = x + session[:, None, :]
        return nn.LayerNorm(param_dtype=param_dtype, dtype=compute_dtype, name="norm")(y)


class ProtogrokModel(nn.Module):
    """Full two-stage flow Transformer with multi-task heads."""

    cfg: ProtogrokConfig

    @nn.compact
    def __call__(self, payload_toks: jnp.ndarray, headers: jnp.ndarray,
                 proto_id: jnp.ndarray, *, task: Task = "anomaly",
                 deterministic: bool = True) -> jnp.ndarray:
        cfg = self.cfg
        compute_dtype, param_dtype = _dtypes(cfg)

        p = PayloadEncoder(cfg, name="payload")(payload_toks)          # [B, T, P]
        h = HeaderEncoder(cfg, name="header")(headers)                 # [B, H]
        h_expand = jnp.broadcast_to(h[:, None, :], (h.shape[0], p.shape[1], h.shape[1]))
        x = jnp.concatenate([p, h_expand], axis=-1)                    # [B, T, P+H]
        x = nn.Dense(cfg.d_model, param_dtype=param_dtype, dtype=compute_dtype, name="join")(x)

        # nn.remat recomputes each block's activations during the backward pass
        # instead of keeping them live. It bounds peak activation memory to one
        # block, which stops a bad XLA fusion from ballooning the whole stack.
        # static_argnums=(2,) marks `deterministic` (self=0, x=1) as static.
        Block = nn.remat(TransformerBlock, static_argnums=(2,)) if cfg.remat \
            else TransformerBlock
        for i in range(cfg.packet_layers):
            x = Block(cfg, name=f"pblock_{i}")(x, deterministic)
        x = ProtocolAdapter(cfg, name="adapter")(x, proto_id)
        x = MemoryModule(cfg, name="memory")(x)
        for i in range(cfg.session_layers):
            x = Block(cfg, name=f"sblock_{i}")(x, deterministic)

        pooled = jnp.mean(x, axis=1)                                   # [B, D]

        # Build ALL heads on every call so a single `init` materializes every
        # task's parameters (shared trunk, multi-task heads). Unused heads are
        # tiny and their extra compute is negligible.
        def head(width: int, name: str) -> jnp.ndarray:
            z = nn.LayerNorm(param_dtype=param_dtype, dtype=compute_dtype, name=f"{name}_ln")(pooled)
            return nn.Dense(width, param_dtype=param_dtype, dtype=compute_dtype, name=f"{name}_out")(z)

        outputs = {
            "pooled": pooled,
            "anomaly": head(2, "anom_head"),
            "class": head(cfg.num_classes, "class_head"),
            # byte-decode head (no LayerNorm in the original), on pooled state
            "protocol": nn.Dense(cfg.byte_vocab, param_dtype=param_dtype,
                                 dtype=compute_dtype, name="decode_head")(pooled),
        }
        if task not in outputs:
            raise ValueError(f"Unknown task: {task!r}")
        return outputs[task]


def init_params(cfg: ProtogrokConfig, rng: jax.Array, batch: int = 2):
    """Initialise parameters with a dummy batch; returns the params pytree."""
    L, T = cfg.payload_max_len, cfg.max_packets
    payload = jnp.zeros((batch, T, L), jnp.int32)
    headers = jnp.zeros((batch, 5), jnp.int32)
    proto = jnp.zeros((batch,), jnp.int32)
    model = ProtogrokModel(cfg)
    variables = model.init(rng, payload, headers, proto, task="anomaly", deterministic=True)
    return model, variables


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
