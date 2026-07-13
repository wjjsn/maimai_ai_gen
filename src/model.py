from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


@dataclass
class ModelDimensions:
    n_audio_ctx: int
    n_vocab: int
    n_text_ctx: int
    n_state: int
    n_head: int
    n_layer: int
    n_audio_adapter_layer: int
    audio_adapter_kernel: int
    audio_adapter_dropout: float


@dataclass
class DecoderKVCache:
    """单个推理窗口的 decoder KV 缓存，不跨窗口复用。"""

    text_len: int
    self_kv: list[tuple[Tensor, Tensor]]
    cross_kv: list[Optional[tuple[Tensor, Tensor]]]


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        is_causal: bool = False,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
        cache_len: Optional[int] = None,
    ):
        q = self.query(x)

        if xa is None:
            k = self.key(x)
            v = self.value(x)
            if kv_cache is not None:
                if cache_len is None:
                    raise ValueError("self-attention 缓存缺少文本长度")
                key_cache, value_cache = kv_cache
                end = cache_len + x.shape[1]
                if end > key_cache.shape[1]:
                    raise ValueError("self-attention 缓存容量不足")
                key_cache[:, cache_len:end].copy_(k)
                value_cache[:, cache_len:end].copy_(v)
                k = key_cache[:, :end]
                v = value_cache[:, :end]
            new_kv = None
        elif kv_cache is None:
            k = self.key(xa)
            v = self.value(xa)
            new_kv = (k, v)
        else:
            k, v = kv_cache
            new_kv = kv_cache

        wv, qk = self.qkv_attention(
            q, k, v, mask, key_padding_mask, is_causal=is_causal
        )
        return self.out(wv), qk, new_kv

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = key_padding_mask[:, None, None, :]
            a = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if is_causal:
                if mask is None or n_ctx != k.shape[-2]:
                    raise ValueError("causal attention 的 query/key 长度不一致")
                qk = qk + mask[:n_ctx, :n_ctx]
            if key_padding_mask is not None:
                qk = qk.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        xa_mask: Optional[Tensor] = None,
        self_kv: Optional[tuple[Tensor, Tensor]] = None,
        cross_kv: Optional[tuple[Tensor, Tensor]] = None,
        cache_len: Optional[int] = None,
    ):
        x = x + self.attn(
            self.attn_ln(x),
            mask=mask,
            is_causal=cache_len is None or cache_len == 0,
            kv_cache=self_kv,
            cache_len=cache_len,
        )[0]
        new_cross_kv = None
        if self.cross_attn:
            cross, _, new_cross_kv = self.cross_attn(
                self.cross_attn_ln(x),
                xa,
                key_padding_mask=xa_mask,
                kv_cache=cross_kv,
            )
            x = x + cross
        x = x + self.mlp(self.mlp_ln(x))
        return x, new_cross_kv


class AudioAdapterBlock(nn.Module):
    """保留 MERT 帧率的轻量局部节奏适配器。"""

    def __init__(self, n_state: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(n_state)
        self.depthwise = nn.Conv1d(
            n_state, n_state, kernel_size, padding=kernel_size // 2, groups=n_state
        )
        self.pointwise = Linear(n_state, n_state)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        local = self.depthwise(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(self.pointwise(F.gelu(local)))


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        nn.init.normal_(self.positional_embedding, std=0.02)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(float("-inf")).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        xa_mask: Optional[Tensor] = None,
        kv_cache: Optional[DecoderKVCache] = None,
    ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, MERT帧数, 状态维度)
            the encoded audio features to be attended on
        xa_mask : torch.BoolTensor, shape = (batch_size, n_audio_ctx)
            True 表示真实 MERT 帧，False 表示窗口补零。
        """
        if xa_mask is not None:
            if xa_mask.dtype != torch.bool or xa_mask.shape != xa.shape[:2]:
                raise ValueError(
                    f"音频 mask 形状或类型错误: {xa_mask.shape}/{xa_mask.dtype}，"
                    f"需要 {xa.shape[:2]}/torch.bool"
                )
            if not xa_mask.any(dim=1).all():
                raise ValueError("每个音频窗口必须至少包含一个真实 MERT 帧")
        offset = 0 if kv_cache is None else kv_cache.text_len
        if offset + x.shape[1] > self.positional_embedding.shape[0]:
            raise ValueError("decoder token 长度超过上下文上限")
        if kv_cache is not None:
            if x.shape[0] != kv_cache.self_kv[0][0].shape[0]:
                raise ValueError("decoder 缓存 batch 大小不匹配")
            if x.device != kv_cache.self_kv[0][0].device:
                raise ValueError("decoder 缓存的设备或类型不匹配")
            if offset and x.shape[1] != 1:
                raise ValueError("已有 decoder 缓存时每次只能输入一个 token")
            for layer_i, (self_k, self_v) in enumerate(kv_cache.self_kv):
                if self_k.shape != self_v.shape or self_k.shape[1] != self.positional_embedding.shape[0]:
                    raise ValueError(f"第 {layer_i} 层 self-attention 缓存形状错误")
            for layer_i, cross in enumerate(kv_cache.cross_kv):
                if cross is not None and (
                    cross[0].shape != cross[1].shape
                    or cross[0].shape[:2] != xa.shape[:2]
                ):
                    raise ValueError(f"第 {layer_i} 层 cross-attention 缓存形状错误")
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)
        if kv_cache is not None and x.dtype != kv_cache.self_kv[0][0].dtype:
            raise ValueError("decoder 缓存的设备或类型不匹配")

        for layer_i, block in enumerate(self.blocks):
            x, cross_kv = block(
                x,
                xa,
                mask=self.mask,
                xa_mask=xa_mask,
                self_kv=None if kv_cache is None else kv_cache.self_kv[layer_i],
                cross_kv=None if kv_cache is None else kv_cache.cross_kv[layer_i],
                cache_len=None if kv_cache is None else offset,
            )
            if kv_cache is not None:
                kv_cache.cross_kv[layer_i] = cross_kv

        if kv_cache is not None:
            kv_cache.text_len += x.shape[1]

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits

    def new_kv_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> DecoderKVCache:
        shape = (batch_size, self.positional_embedding.shape[0], self.positional_embedding.shape[1])
        return DecoderKVCache(
            text_len=0,
            self_kv=[
                (torch.empty(shape, device=device, dtype=dtype), torch.empty(shape, device=device, dtype=dtype))
                for _ in self.blocks
            ],
            cross_kv=[None] * len(self.blocks),
        )


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.audio_ln = LayerNorm(self.dims.n_state)
        self.audio_adapter: Iterable[AudioAdapterBlock] = nn.ModuleList(
            [
                AudioAdapterBlock(
                    self.dims.n_state,
                    self.dims.audio_adapter_kernel,
                    self.dims.audio_adapter_dropout,
                )
                for _ in range(self.dims.n_audio_adapter_layer)
            ]
        )
        self.audio_position = nn.Parameter(
            torch.empty(self.dims.n_audio_ctx, self.dims.n_state)
        )
        nn.init.normal_(self.audio_position, std=0.02)
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_state,
            self.dims.n_head,
            self.dims.n_layer,
        )

    def embed_audio(
        self, features: torch.Tensor, audio_mask: Optional[torch.Tensor] = None
    ):
        if features.shape[1:] != self.audio_position.shape:
            raise ValueError(
                f"MERT 特征形状错误: {features.shape[1:]}，需要 {self.audio_position.shape}"
            )
        if audio_mask is not None:
            if audio_mask.dtype != torch.bool or audio_mask.shape != features.shape[:2]:
                raise ValueError("音频 mask 形状或类型错误")
            # 卷积会访问相邻帧，因此要在适配器前消除 padding 的任意原始值。
            features = features.masked_fill(~audio_mask.unsqueeze(-1), 0)
        x = self.audio_ln(features)
        for block in self.audio_adapter:
            x = block(x)
        return x + self.audio_position.to(features.dtype)

    def logits(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[DecoderKVCache] = None,
    ):
        return self.decoder(tokens, audio_features, xa_mask=audio_mask, kv_cache=kv_cache)

    def new_kv_cache(self, audio_features: torch.Tensor) -> DecoderKVCache:
        return self.decoder.new_kv_cache(
            audio_features.shape[0], audio_features.device, audio_features.dtype
        )

    def forward(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.embed_audio(features, audio_mask), xa_mask=audio_mask)


def _self_check() -> None:
    dims = ModelDimensions(100, 3072, 64, 768, 12, 4, 2, 5, 0.05)
    model = Whisper(dims)
    audio = torch.randn(2, 100, 768)
    tokens = torch.randint(0, 3072, (2, 16))
    logits = model(audio, tokens)
    assert logits.shape == (2, 16, 3072)
    logits.mean().backward()
    grads = [p.grad for p in model.decoder.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    model.eval()
    audio_mask = torch.ones(2, 100, dtype=torch.bool)
    audio_mask[0, :10] = False
    audio_mask[1, -10:] = False
    changed_padding = audio.clone()
    changed_padding[~audio_mask] = 1e4
    with torch.no_grad():
        masked_logits = model(audio, tokens, audio_mask)
        changed_logits = model(changed_padding, tokens, audio_mask)
    assert torch.allclose(masked_logits, changed_logits, atol=1e-5, rtol=1e-5)
    with disable_sdpa(), torch.no_grad():
        fallback_logits = model(audio, tokens, audio_mask)
        fallback_changed_logits = model(changed_padding, tokens, audio_mask)
    assert torch.allclose(fallback_logits, fallback_changed_logits, atol=1e-5, rtol=1e-5)

    def check_cached_logits() -> None:
        with torch.no_grad():
            audio_features = model.embed_audio(audio, audio_mask)
            reference = model.logits(tokens, audio_features, audio_mask)
            cache = model.new_kv_cache(audio_features)
            prefix_len = 5
            parts = [model.logits(tokens[:, :prefix_len], audio_features, audio_mask, cache)]
            for token_i in range(prefix_len, tokens.shape[1]):
                parts.append(
                    model.logits(tokens[:, token_i : token_i + 1], audio_features, audio_mask, cache)
                )
            cached = torch.cat(parts, dim=1)
            assert cache.text_len == tokens.shape[1]
            assert all(k.shape == v.shape == (2, 64, 768) for k, v in cache.self_kv)
            assert all(kv is not None and kv[0].shape == kv[1].shape == (2, 100, 768) for kv in cache.cross_kv)
            # 不同 query 长度的 SDPA 归约顺序略有不同，但 greedy 结果必须一致。
            assert torch.allclose(reference, cached, atol=3e-4, rtol=3e-4)
            assert torch.equal(reference.argmax(dim=-1), cached.argmax(dim=-1))

            changed_features = model.embed_audio(changed_padding, audio_mask)
            changed_cache = model.new_kv_cache(changed_features)
            changed_parts = [model.logits(tokens[:, :prefix_len], changed_features, audio_mask, changed_cache)]
            for token_i in range(prefix_len, tokens.shape[1]):
                changed_parts.append(
                    model.logits(tokens[:, token_i : token_i + 1], changed_features, audio_mask, changed_cache)
                )
            assert torch.allclose(cached, torch.cat(changed_parts, dim=1), atol=1e-5, rtol=1e-5)

            try:
                model.logits(tokens[:, :2], audio_features, audio_mask, cache)
            except ValueError:
                pass
            else:
                raise AssertionError("已有缓存时必须拒绝多个 token")

    check_cached_logits()
    with disable_sdpa():
        check_cached_logits()
    print("[model] 自检通过")


if __name__ == "__main__":
    _self_check()
