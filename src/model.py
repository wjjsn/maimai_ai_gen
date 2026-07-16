"""基于 Hugging Face BERT 编码器的逐帧音符事件模型。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import BertConfig, BertModel


@dataclass(frozen=True)
class ModelDimensions:
    n_mels: int
    hidden_dim: int
    layers: int
    dropout: float
    window_frames: int
    attention_heads: int = 8


class NoteTimingTransformer(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        if dims.hidden_dim % dims.attention_heads:
            raise ValueError("隐藏维度必须能被注意力头数整除")
        self.dims = dims
        self.input_norm = nn.LayerNorm(dims.n_mels)
        self.audio_projection = nn.Linear(dims.n_mels, dims.hidden_dim)
        self.difficulty_projection = nn.Sequential(
            nn.Linear(1, dims.hidden_dim),
            nn.GELU(),
            nn.Linear(dims.hidden_dim, dims.hidden_dim),
        )
        config = BertConfig(
            vocab_size=1,
            hidden_size=dims.hidden_dim,
            num_hidden_layers=dims.layers,
            num_attention_heads=dims.attention_heads,
            intermediate_size=dims.hidden_dim * 4,
            hidden_dropout_prob=dims.dropout,
            attention_probs_dropout_prob=dims.dropout,
            max_position_embeddings=dims.window_frames,
            type_vocab_size=1,
            pad_token_id=0,
            use_cache=False,
        )
        self.encoder = BertModel(config, add_pooling_layer=False)
        self.tap_head = nn.Linear(dims.hidden_dim, 9)
        self.hold_head = nn.Linear(dims.hidden_dim, 3)
        self.duration_head = nn.Linear(dims.hidden_dim, 2)

    def forward(
        self, features: Tensor, difficulty: Tensor, mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.dims.n_mels:
            raise ValueError(f"音频特征形状错误: {tuple(features.shape)}")
        if features.shape[1] > self.dims.window_frames:
            raise ValueError("输入帧数超过模型窗口")
        if difficulty.shape != features.shape[:1] or not difficulty.is_floating_point():
            raise ValueError("浮点难度张量形状或类型错误")
        if not torch.isfinite(difficulty).all():
            raise ValueError("浮点难度必须是有限数字")
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if mask.dtype != torch.bool or mask.shape != features.shape[:2]:
            raise ValueError("音频 mask 形状或类型错误")
        audio = self.audio_projection(self.input_norm(features))
        condition = self.difficulty_projection((difficulty / 15.0).unsqueeze(-1)).unsqueeze(1)
        hidden = self.encoder(inputs_embeds=audio + condition, attention_mask=mask).last_hidden_state
        return self.tap_head(hidden), self.hold_head(hidden), torch.sigmoid(self.duration_head(hidden))


def _self_check() -> None:
    torch.manual_seed(0)
    model = NoteTimingTransformer(ModelDimensions(8, 16, 2, 0.0, 32, 4))
    features = torch.randn(2, 32, 8)
    mask = torch.ones(2, 32, dtype=torch.bool)
    output = model(features, torch.tensor([12.3, 14.7]), mask)
    assert tuple(value.shape for value in output) == ((2, 32, 9), (2, 32, 3), (2, 32, 2))
    sum(value.mean() for value in output).backward()
    assert model.audio_projection.weight.grad is not None
    assert model.encoder.encoder.layer[0].attention.self.query.weight.grad is not None
    assert model.tap_head.weight.grad is not None
    model.eval()
    one = model(features[:1], torch.tensor([13.0]), mask[:1])
    harder = model(features[:1], torch.tensor([14.0]), mask[:1])
    assert any(not torch.allclose(left, right) for left, right in zip(one, harder))
    print("[model] 自检通过")


if __name__ == "__main__":
    _self_check()
