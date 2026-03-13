"""Longcat Flash N-gram model — longcat_flash backbone + NgramEmbedding, ModelProtocol-compliant.

Ported from mlx_lm longcat_flash_ngram. Reuses LongcatFlashDecoderLayer from longcat_flash;
adds NgramEmbedding (word + n-gram embedders) and NgramContextCache for decode context.
Norms: nn.RMSNorm only (via longcat_flash layers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.cache_list import CacheList
from mlxs.cache.kv import KVCache
from mlxs.models.base import BaseModelArgs
from mlxs.models.longcat_flash import (
    LongcatFlashDecoderLayer,
)
from mlxs.models.longcat_flash import (
    Model as LongcatFlashModel,
)


class NgramContextCache:
    """Holds sliding context of token ids for NgramEmbedding; implements CacheProtocol.

    Used as the first element of the model cache list. The model reads/writes
    .context (mx.array | None) and calls update_context() so the embedding can
    use previous tokens for n-gram lookups across decode steps.
    """

    def __init__(self) -> None:
        self._context: mx.array | None = None

    @property
    def context(self) -> mx.array | None:
        return self._context

    def update_context(self, arr: mx.array) -> None:
        self._context = arr

    @property
    def offset(self) -> int:
        if self._context is None:
            return 0
        return int(self._context.shape[-1])

    @property
    def keys(self) -> mx.array | None:
        return None

    @property
    def values(self) -> mx.array | None:
        return None

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return keys, values

    def trim(self, n: int) -> int:
        if self._context is None or n <= 0:
            return 0
        T = self._context.shape[-1]
        removed = min(n, T)
        if removed >= T:
            self._context = None
        else:
            self._context = self._context[..., :-removed]
        return removed

    @property
    def state_size_bytes(self) -> int:
        if self._context is None:
            return 0
        return int(self._context.nbytes)

    def reset(self) -> None:
        self._context = None


@dataclass
class ModelArgs(BaseModelArgs):
    """Longcat Flash N-gram config; from_dict aligned to config.json.

    Same as longcat_flash plus ngram_vocab_size_ratio, emb_neighbor_num, emb_split_num.
    """

    model_type: str = "longcat_flash_ngram"
    attention_method: str = "flash"
    zero_expert_type: str = "identity"
    hidden_size: int = 2048
    ffn_hidden_size: int = 5632
    moe_topk: int = 1
    expert_ffn_hidden_size: int = 1408
    n_routed_experts: int = 8
    zero_expert_num: int = 1
    num_layers: int = 32
    vocab_size: int = 128256
    max_position_embeddings: int = 32768
    num_attention_heads: int = 16
    kv_lora_rank: int = 256
    q_lora_rank: int | None = 768
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 128
    routed_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    mla_scale_q_lora: bool = False
    mla_scale_kv_lora: bool = False
    attention_bias: bool = False
    norm_topk_prob: bool = False
    router_bias: bool = False
    rope_scaling: dict[str, Any] | None = None
    ngram_vocab_size_ratio: int = 78
    emb_neighbor_num: int = 4
    emb_split_num: int = 4


class NgramEmbedding(nn.Module):
    """Word embedding plus k*(n-1) n-gram embedders with vocab hashing and post projections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.m = args.ngram_vocab_size_ratio * args.vocab_size
        self.k = args.emb_split_num
        self.n = args.emb_neighbor_num

        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)

        num_embedders = self.k * (self.n - 1)
        emb_dim = args.hidden_size // num_embedders

        self.embedders: list[nn.Embedding] = []
        self.post_projs: list[nn.Linear] = []
        for i in range(num_embedders):
            emb_vocab_size = int(self.m + i * 2 + 1)
            self.embedders.append(nn.Embedding(emb_vocab_size, emb_dim))
            self.post_projs.append(nn.Linear(emb_dim, args.hidden_size, bias=False))
        self._vocab_mods = self._compute_vocab_mods()

    def _compute_vocab_mods(self) -> dict[tuple[int, int], list[int]]:
        vocab_mods: dict[tuple[int, int], list[int]] = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                power_mod = 1
                mods = []
                for _ in range(i - 1):
                    power_mod = (power_mod * self.vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[(i, j)] = mods
        return vocab_mods

    @staticmethod
    def _shift_right(x: mx.array, n: int) -> mx.array:
        if n <= 0:
            return x
        batch_size, seq_len = x.shape
        if seq_len <= n:
            return mx.zeros_like(x)
        return mx.concatenate(
            [mx.zeros((batch_size, n), dtype=x.dtype), x[..., :-n]],
            axis=-1,
        )

    @staticmethod
    def _get_ngram_ids(
        context: mx.array,
        shifted_ids: dict[int, mx.array],
        vocab_mods: list[int],
        ngram: int,
    ) -> mx.array:
        ngram_ids = context
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[Any] | None = None,
    ) -> mx.array:
        seq_len = input_ids.shape[-1]
        input_ids = input_ids.astype(mx.int64)

        if cache is not None:
            ctx_cache = cache[0]
            if ctx_cache.context is None:
                context = input_ids
            else:
                context = mx.concatenate([ctx_cache.context, input_ids], axis=-1)
            ctx_cache.update_context(context[..., max(0, context.shape[-1] - self.n + 1) :])
        else:
            context = input_ids

        x = self.word_embeddings(input_ids)
        shifted_ids = {i: self._shift_right(context, i - 1) for i in range(2, self.n + 1)}

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, self._vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                x_ngram = self.embedders[index](new_ids)
                x_proj = self.post_projs[index](x_ngram)
                x = x + x_proj

        return x / (1 + self.k * (self.n - 1))


class LongcatFlashNgramModel(nn.Module):
    """N-gram embedding + stacked LongcatFlash decoder layers + final RMSNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_layers = args.num_layers
        self.ngram_embeddings = NgramEmbedding(args)
        self.layers = [LongcatFlashDecoderLayer(args) for _ in range(args.num_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[CacheList | NgramContextCache] | None = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * (1 + self.num_layers)  # type: ignore[list-item]

        h = self.ngram_embeddings(input_ids, cache=cache)
        layer_caches = cache[1:]
        mask = create_attention_mask(h, layer_caches[0], return_array=True)

        for layer, c in zip(self.layers, layer_caches, strict=True):
            h = layer(
                h,
                mask=mask,
                cache=(c[0], c[1]) if c is not None else None,
            )

        return self.norm(h)


class Model(nn.Module):
    """Longcat Flash N-gram LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LongcatFlashNgramModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[CacheList | NgramContextCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        del mask
        return self.lm_head(self.model(input_ids, cache=cache))

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[NgramContextCache | CacheList]:
        ngram = NgramContextCache()
        layer_caches = [CacheList(KVCache(), KVCache(), kv_index=0) for _ in self.model.layers]
        return [ngram, *layer_caches]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Run longcat_flash sanitize then remap embed_tokens -> ngram word_embeddings."""
        import inspect

        from mlxs.models.longcat_flash import ModelArgs as FlashArgs

        flash_params = inspect.signature(FlashArgs).parameters
        flash_arg_dict = {k: getattr(self.args, k) for k in flash_params if hasattr(self.args, k)}
        flash_args = FlashArgs.from_dict(flash_arg_dict)
        weights = LongcatFlashModel(flash_args).sanitize(weights)
        if "model.embed_tokens.weight" in weights:
            weights["model.ngram_embeddings.word_embeddings.weight"] = weights.pop(
                "model.embed_tokens.weight"
            )
        return weights
