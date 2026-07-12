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
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask, key_padding_mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
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
                is_causal=mask is not None and n_ctx > 1,
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
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
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(
                self.cross_attn_ln(x),
                xa,
                key_padding_mask=xa_mask,
                kv_cache=kv_cache,
            )[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


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
        kv_cache: Optional[dict] = None,
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
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, xa_mask=xa_mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.audio_ln = LayerNorm(self.dims.n_state)
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

    def embed_audio(self, features: torch.Tensor):
        if features.shape[1:] != self.audio_position.shape:
            raise ValueError(
                f"MERT 特征形状错误: {features.shape[1:]}，需要 {self.audio_position.shape}"
            )
        return self.audio_ln(features) + self.audio_position.to(features.dtype)

    def logits(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ):
        return self.decoder(tokens, audio_features, xa_mask=audio_mask)

    def forward(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.embed_audio(features), xa_mask=audio_mask)


def _self_check() -> None:
    dims = ModelDimensions(100, 3072, 64, 768, 12, 4)
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
    audio_mask[:, :10] = False
    changed_padding = audio.clone()
    changed_padding[:, :10] = 1e4
    with torch.no_grad():
        masked_logits = model(audio, tokens, audio_mask)
        changed_logits = model(changed_padding, tokens, audio_mask)
    assert torch.allclose(masked_logits, changed_logits, atol=1e-5, rtol=1e-5)
    with disable_sdpa(), torch.no_grad():
        fallback_logits = model(audio, tokens, audio_mask)
        fallback_changed_logits = model(changed_padding, tokens, audio_mask)
    assert torch.allclose(fallback_logits, fallback_changed_logits, atol=1e-5, rtol=1e-5)
    print("[model] 自检通过")


if __name__ == "__main__":
    _self_check()
