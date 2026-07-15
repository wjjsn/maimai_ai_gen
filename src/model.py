from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelDimensions:
    input_dim: int
    hidden_dim: int
    layers: int
    kernel_size: int
    dropout: float


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels * 2, 1)
        self.out = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm(x.transpose(1, 2)).transpose(1, 2)
        value, gate = self.pointwise(self.depthwise(normalized)).chunk(2, dim=1)
        return x + self.dropout(self.out(F.gelu(value) * torch.sigmoid(gate)))


class ChartCNN(nn.Module):
    """整曲 CNN：分类预测事件数量，回归预测两个持续音时长。"""
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.input_norm = nn.LayerNorm(dims.input_dim)
        self.input_projection = nn.Conv1d(dims.input_dim, dims.hidden_dim, 1)
        self.level_projection = nn.Linear(1, dims.hidden_dim)
        self.level_embedding = nn.Embedding(7, dims.hidden_dim)
        self.blocks = nn.ModuleList([ResidualConvBlock(dims.hidden_dim, dims.kernel_size, 2 ** (i % 5), dims.dropout) for i in range(dims.layers)])
        self.output_norm = nn.LayerNorm(dims.hidden_dim)
        self.tap_head = nn.Conv1d(dims.hidden_dim, 9, 1)
        self.hold_head = nn.Conv1d(dims.hidden_dim, 3, 1)
        self.duration_head = nn.Conv1d(dims.hidden_dim, 2, 1)

    def forward(self, features: Tensor, level_queries: Tensor, level_indices: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.dims.input_dim:
            raise ValueError(f"音频特征形状错误: {tuple(features.shape)}")
        if mask is not None:
            if mask.dtype != torch.bool or mask.shape != features.shape[:2]:
                raise ValueError("音频 mask 形状或类型错误")
            features = features.masked_fill(~mask.unsqueeze(-1), 0)
        if level_queries.shape != features.shape[:1] or not level_queries.is_floating_point() or not torch.isfinite(level_queries).all():
            raise ValueError("浮点难度张量形状或数值错误")
        if level_indices.shape != features.shape[:1] or level_indices.dtype != torch.long or ((level_indices < 2) | (level_indices > 6)).any():
            raise ValueError("难度编号张量形状、类型或范围错误")
        x = self.input_projection(self.input_norm(features).transpose(1, 2))
        condition = self.level_projection((level_queries / 15).unsqueeze(-1)) + self.level_embedding(level_indices)
        x = x + condition.unsqueeze(-1)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), 0)
        for block in self.blocks:
            x = block(x)
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(1), 0)
        x = self.output_norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        return (
            self.tap_head(x).transpose(1, 2),
            self.hold_head(x).transpose(1, 2),
            torch.sigmoid(self.duration_head(x).transpose(1, 2)),
        )


def _self_check() -> None:
    model = ChartCNN(ModelDimensions(8, 16, 3, 5, 0.0))
    output = model(torch.randn(2, 37, 8), torch.tensor([12.3, 14.7]), torch.tensor([4, 5]), torch.ones(2, 37, dtype=torch.bool))
    assert tuple(value.shape for value in output) == ((2, 37, 9), (2, 37, 3), (2, 37, 2))
    assert (0 <= output[2]).all() and (output[2] <= 1).all()
    sum(value.mean() for value in output).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    model.eval()
    features = torch.randn(1, 17, 8)
    level_query, level_idx = torch.tensor([13.7]), torch.tensor([5])
    alone = model(features, level_query, level_idx)
    padded = model(torch.cat((features, torch.zeros(1, 9, 8)), dim=1), level_query, level_idx, torch.tensor([[True] * 17 + [False] * 9]))
    assert all(torch.allclose(left, right[:, :17], atol=1e-6) for left, right in zip(alone, padded))
    harder = model(features, torch.tensor([14.7]), level_idx)
    assert any(not torch.allclose(left, right) for left, right in zip(alone, harder))
    print("[model] 自检通过")


if __name__ == "__main__":
    _self_check()
