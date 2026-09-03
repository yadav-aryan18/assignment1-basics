from __future__ import annotations

import torch
import torch.nn as nn

from einops import rearrange


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None =None, dtype: torch.dtype | None =None) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.std = (2/(in_features + out_features))**0.5
        self.W = nn.Parameter(torch.nn.init.trunc_normal_(
                                                        torch.empty(out_features, in_features, device=self.device, dtype=self.dtype),
                                                        mean = 0.0,
                                                        std = self.std,
                                                        a = -3.0 * self.std,
                                                        b = 3.0 * self.std,
                                                        ),
                                                        requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.W.T
        return x


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.embedding_matrix = nn.Parameter(torch.nn.init.trunc_normal_(
                                                                torch.empty(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype),
                                                                mean = 0.0,
                                                                std = 1.0,
                                                                a = -3.0,
                                                                b = 3.0
                                                                ),
                                                                requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embds =  self.embedding_matrix[token_ids]
        return embds



class CustomRMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):

        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.gain = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(torch.float32)

        squared_sum_x = (x * x).sum(dim=-1, keepdim=True)
        rms_x = torch.sqrt((squared_sum_x / self.d_model) + self.eps)

        x = (x / rms_x) * self.gain

        return x.to(in_dtype)


def customSiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.std = (2/(self.d_ff + self.d_model))**0.5
        self.W1 = nn.Parameter(torch.nn.init.trunc_normal_(
                                                            torch.empty(self.d_ff, self.d_model, device=self.device, dtype=self.dtype),
                                                            mean = 0.0,
                                                            std = self.std,
                                                            a = -3.0 * self.std,
                                                            b = 3.0 * self.std
                                                            )
                                                            , requires_grad=True)

        self.W2 = nn.Parameter(torch.nn.init.trunc_normal_(
                                                            torch.empty(self.d_model, self.d_ff, device=self.device, dtype=self.dtype),
                                                            mean = 0.0,
                                                            std = self.std,
                                                            a = -3.0 * self.std,
                                                            b = 3.0 * self.std
                                                            )
                                                            , requires_grad=True)

        self.W3 = nn.Parameter(torch.nn.init.trunc_normal_(
                                                            torch.empty(self.d_ff, self.d_model, device=self.device, dtype=self.dtype),
                                                            mean = 0.0,
                                                            std = self.std,
                                                            a = -3.0 * self.std,
                                                            b = 3.0 * self.std
                                                            )
                                                            , requires_grad=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = customSiLU(x @ self.W1.T) * (x @ self.W3.T)
        x = x @ self.W2.T
        return x


class RotaryPositionalEmbeddding(nn.Module):

    RoPE_cos: torch.Tensor
    RoPE_sin: torch.Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.rope_init = torch.logspace(start=0, end=(d_k//2) - 1 , steps=d_k//2, base=1/(theta**(2/d_k)), device=self.device)
        self.rope_init = torch.linspace(0, max_seq_len-1, steps=max_seq_len, device=self.device).unsqueeze(-1) * self.rope_init.unsqueeze(0)

        self.cosine_rope = torch.cos(self.rope_init)
        self.sine_rope = torch.sin(self.rope_init)

        self.register_buffer("RoPE_cos", self.cosine_rope, persistent=False)
        self.register_buffer("RoPE_sin", self.sine_rope, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        cos_rope = self.RoPE_cos[token_positions]
        sin_rope = self.RoPE_sin[token_positions]

        x1 = x_even * cos_rope - x_odd * sin_rope
        x2 = x_odd * cos_rope + x_even * sin_rope

        x = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2, end_dim=-1)
        return x


def customSoftmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    y, _ = x.max(dim=dim, keepdim=True)
    x = x - y
    x = torch.exp(x)
    sumX_dims = x.sum(dim=dim, keepdim=True)
    x = x / sumX_dims
    return x



def scaledDotProductAttention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:

    d_k = K.shape[-1]
    x = Q @ K.transpose(-2, -1)
    x = x / (d_k)**0.5

    if mask is not None:
        x = torch.masked_fill(x, ~mask, -float('inf'))

    x = customSoftmax(x, dim=-1)
    x = x @ V
    return x


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model//num_heads

        self.Wq = Linear(in_features=num_heads * self.d_k, out_features=d_model)
        self.Wk = Linear(in_features=num_heads * self.d_k, out_features=d_model)
        self.Wv = Linear(in_features=num_heads * self.d_v, out_features=d_model)
        self.Wo = Linear(in_features=d_model, out_features=num_heads * self.d_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = rearrange(Q, "... (heads dk) -> ... heads dk", heads=self.num_heads)
        Q = rearrange(Q, "... seq_len heads d_k -> ... heads seq_len d_k")

        K = rearrange(K, "... (heads d_k) -> ... heads d_k", heads=self.num_heads)
        K = rearrange(K, "... seq_len heads d_k -> ... heads seq_len d_k")

        V = rearrange(V, "... (heads d_v) -> ... heads d_v", heads=self.num_heads)
        V = rearrange(V, "... seq_len heads d_v -> ... heads seq_len d_v")

        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], dtype=torch.bool))

        x = scaledDotProductAttention(Q, K, V, mask)
        x = rearrange(x, "... heads seq_len d_v -> ... seq_len (heads d_v)")

        x = self.Wo(x)
        return x




class CausalMHSAwithRoPE(nn.Module):

    mask: torch.Tensor

    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model//num_heads

        self.device = device
        self.dtype = dtype

        self.Wq = Linear(in_features=num_heads * self.d_k, out_features=d_model, device=self.device, dtype=self.dtype)
        self.Wk = Linear(in_features=num_heads * self.d_k, out_features=d_model, device=self.device, dtype=self.dtype)
        self.Wv = Linear(in_features=num_heads * self.d_v, out_features=d_model, device=self.device, dtype=self.dtype)
        self.Wo = Linear(in_features=d_model, out_features=num_heads * self.d_v, device=self.device, dtype=self.dtype)
        self.RoPE = RotaryPositionalEmbeddding(theta, self.d_k, max_seq_len, device=self.device, dtype=self.dtype)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len, device=self.device, dtype=torch.bool)), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = rearrange(Q, "... (heads dk) -> ... heads dk", heads=self.num_heads)
        Q = rearrange(Q, "... seq_len heads d_k -> ... heads seq_len d_k")

        K = rearrange(K, "... (heads d_k) -> ... heads d_k", heads=self.num_heads)
        K = rearrange(K, "... seq_len heads d_k -> ... heads seq_len d_k")

        V = rearrange(V, "... (heads d_v) -> ... heads d_v", heads=self.num_heads)
        V = rearrange(V, "... seq_len heads d_v -> ... heads seq_len d_v")

        Q = self.RoPE(Q, token_positions)
        K = self.RoPE(K, token_positions)

        mask = self.mask[:x.shape[-2], :x.shape[-2]]

        x = scaledDotProductAttention(Q, K, V, mask)
        x = rearrange(x, "... heads seq_len d_v -> ... seq_len (heads d_v)")

        x = self.Wo(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.l1s1_rms_norm = CustomRMSNorm(d_model, device=self.device, dtype=self.dtype)
        self.l1s2_rms_norm = CustomRMSNorm(d_model, device=self.device, dtype=self.dtype)
        self.attention = CausalMHSAwithRoPE(d_model, num_heads, theta, max_seq_len, device=self.device, dtype=self.dtype)
        self.pw_ffn = PositionWiseFFN(d_model, d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y = self.l1s1_rms_norm(x)

        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)

        y = self.attention(y, token_positions)
        x = x + y

        y = self.l1s2_rms_norm(x)
        y = self.pw_ffn(y)
        x = x + y
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.embeddings = Embedding(vocab_size, d_model, device=self.device, dtype=self.dtype)

        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device=self.device, dtype=self.dtype) for _ in range(num_layers)]
        )

        self.rms_norm = CustomRMSNorm(d_model, device=self.device, dtype=self.dtype)
        self.fc = Linear(d_model, vocab_size, device=self.device, dtype=self.dtype)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(token_indices)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.rms_norm(x)
        x = self.fc(x)
        return x
