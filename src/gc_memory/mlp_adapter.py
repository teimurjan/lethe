"""Learned MLP adapter: delta = f(query, embedding, feedback).

Replaces blind Gaussian noise with a tiny neural network that learns
which direction to push embeddings based on cross-encoder feedback.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class DeltaPredictor(nn.Module):
    """Tiny MLP: concat(query, embedding, xenc_score) → delta vector.

    Input: 384 + 384 + 1 = 769 dims
    Hidden: configurable (default 128)
    Output: 384 dims (delta to add to embedding)
    """

    def __init__(self, embed_dim: int = 384, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        # Initialize with small weights so initial deltas are near-zero
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        query: torch.Tensor,
        embedding: torch.Tensor,
        xenc_score: torch.Tensor,
    ) -> torch.Tensor:
        """Predict delta given query, embedding, and cross-encoder feedback.

        All inputs should be 1D or batched. xenc_score is a scalar (or batch of scalars).
        Returns delta vector(s) of same dim as embedding.
        """
        x = torch.cat([query, embedding, xenc_score.unsqueeze(-1)], dim=-1)
        return self.net(x)


def train_step(
    predictor: DeltaPredictor,
    optimizer: torch.optim.Optimizer,
    query: npt.NDArray[np.float32],
    embedding: npt.NDArray[np.float32],
    xenc_score: float,
    max_norm: float,
) -> tuple[npt.NDArray[np.float32], float]:
    """One training step. Returns (clipped_delta, loss_value).

    Loss: -(sigmoid(xenc) - 0.5) * cos(normalize(emb + delta), query)
    When xenc is high (relevant): drives delta toward query.
    When xenc is low (irrelevant): drives delta away from query.
    """
    q = torch.from_numpy(query).float()
    e = torch.from_numpy(embedding).float()
    s = torch.tensor(xenc_score).float()

    predictor.train()
    optimizer.zero_grad()

    delta = predictor(q, e, s)

    # Compute effective embedding
    new_eff = e + delta
    new_eff_norm = torch.nn.functional.normalize(new_eff, dim=-1)

    # Differentiable loss: cross-encoder feedback directs the gradient
    relevance_signal = torch.sigmoid(s) - 0.5  # range [-0.5, 0.5]
    cosine = torch.dot(new_eff_norm, q)
    loss = -relevance_signal * cosine

    loss.backward()
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
    optimizer.step()

    # Clip delta norm for safety
    with torch.no_grad():
        delta_np = delta.detach().numpy().astype(np.float32)
        delta_norm = float(np.linalg.norm(delta_np))
        if delta_norm > max_norm:
            delta_np = delta_np * (max_norm / delta_norm)

    return delta_np, float(loss.item())


def predict_delta(
    predictor: DeltaPredictor,
    query: npt.NDArray[np.float32],
    embedding: npt.NDArray[np.float32],
    xenc_score: float,
    max_norm: float,
) -> npt.NDArray[np.float32]:
    """Inference only: predict delta without training."""
    predictor.eval()
    with torch.no_grad():
        q = torch.from_numpy(query).float()
        e = torch.from_numpy(embedding).float()
        s = torch.tensor(xenc_score).float()
        delta = predictor(q, e, s).numpy().astype(np.float32)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_norm:
            delta = delta * (max_norm / delta_norm)
    return delta
