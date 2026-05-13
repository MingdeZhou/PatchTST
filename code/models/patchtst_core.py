from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PatchTSTConfig:
    c_in: int
    context_window: int
    target_window: int
    patch_len: int
    stride: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    head_dropout: float
    revin: bool
    affine: bool
    subtract_last: bool


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False, subtract_last: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"unknown RevIN mode: {mode}")

    def _normalize(self, x: Tensor) -> Tensor:
        if self.subtract_last:
            self.center = x[:, -1:, :].detach()
        else:
            self.center = x.mean(dim=1, keepdim=True).detach()
        self.scale = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self.center) / self.scale
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * self.scale + self.center


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.BatchNorm1d(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.BatchNorm1d(d_model)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self._batch_norm(x + self.attn_dropout(attn_out), self.attn_norm)
        ff_out = self.ff(x)
        x = self._batch_norm(x + self.ff_dropout(ff_out), self.ff_norm)
        return x

    @staticmethod
    def _batch_norm(x: Tensor, norm: nn.BatchNorm1d) -> Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)


class PatchTSTCore(nn.Module):
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        if cfg.context_window < cfg.patch_len:
            raise ValueError("seq_len must be at least patch_len")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.cfg = cfg
        self.patch_num = (cfg.context_window - cfg.patch_len) // cfg.stride + 2
        self.revin = RevIN(cfg.c_in, affine=cfg.affine, subtract_last=cfg.subtract_last) if cfg.revin else None
        self.patch_embedding = nn.Linear(cfg.patch_len, cfg.d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, self.patch_num, cfg.d_model))
        self.encoder = nn.Sequential(
            *[EncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.head_dropout = nn.Dropout(cfg.head_dropout)
        self.head = nn.Linear(cfg.d_model * self.patch_num, cfg.target_window)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.position_embedding, -0.02, 0.02)
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        nn.init.zeros_(self.patch_embedding.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B, L, C], got {tuple(x.shape)}")
        if x.size(1) != self.cfg.context_window:
            raise ValueError(f"expected seq_len={self.cfg.context_window}, got {x.size(1)}")
        if x.size(2) != self.cfg.c_in:
            raise ValueError(f"expected channels={self.cfg.c_in}, got {x.size(2)}")

        if self.revin is not None:
            x = self.revin(x, "norm")

        batch_size, _, n_vars = x.shape
        x = x.transpose(1, 2)
        x = torch.nn.functional.pad(x, (0, self.cfg.stride), mode="replicate")
        x = x.unfold(dimension=-1, size=self.cfg.patch_len, step=self.cfg.stride)
        x = x.reshape(batch_size * n_vars, self.patch_num, self.cfg.patch_len)

        x = self.patch_embedding(x) * math.sqrt(self.cfg.d_model)
        x = x + self.position_embedding
        x = self.encoder(x)

        x = x.reshape(batch_size, n_vars, self.patch_num, self.cfg.d_model)
        x = x.transpose(2, 3).flatten(start_dim=-2)
        x = self.head(self.head_dropout(x))
        x = x.transpose(1, 2)

        if self.revin is not None:
            x = self.revin(x, "denorm")
        return x


class Model(PatchTSTCore):
    def __init__(self, configs):
        if getattr(configs, "decomposition", 0):
            raise NotImplementedError("This project implements supervised PatchTST without decomposition.")
        cfg = PatchTSTConfig(
            c_in=int(configs.enc_in),
            context_window=int(configs.seq_len),
            target_window=int(configs.pred_len),
            patch_len=int(configs.patch_len),
            stride=int(configs.stride),
            d_model=int(configs.d_model),
            n_heads=int(configs.n_heads),
            n_layers=int(configs.e_layers),
            d_ff=int(configs.d_ff),
            dropout=float(configs.dropout),
            head_dropout=float(getattr(configs, "head_dropout", 0.0)),
            revin=bool(getattr(configs, "revin", 1)),
            affine=bool(getattr(configs, "affine", 0)),
            subtract_last=bool(getattr(configs, "subtract_last", 0)),
        )
        super().__init__(cfg)
