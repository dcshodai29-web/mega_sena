"""Tensor causal de features para os modelos de machine learning.

``feature_tensor(draws)`` devolve ``F`` com forma ``(n + 1, 60, K)``: a linha
``F[s]`` descreve cada dezena usando **somente** os concursos ``0..s-1``
(0-based) e serve para prever o concurso ``s``. A linha ``F[n]`` serve para
prever o próximo concurso, ainda não sorteado.

A causalidade é verificada em ``tests/test_no_leakage.py``: alterar os
concursos ``s..n-1`` não pode alterar ``F[0..s]``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from megalab.config import N_NUMBERS, P_NUMBER

FEATURE_NAMES = [
    "freq_10", "freq_50", "freq_200", "freq_total", "log_atraso",
    "saiu_t-1", "saiu_t-2", "ewma_hl25", "afinidade_t-1", "transicao_t-1",
]


def onehot(draws: np.ndarray) -> np.ndarray:
    X = np.zeros((len(draws), N_NUMBERS), dtype=np.float64)
    if len(draws):
        X[np.arange(len(draws))[:, None], np.asarray(draws) - 1] = 1.0
    return X


def feature_tensor(draws: np.ndarray) -> np.ndarray:
    draws = np.asarray(draws)
    n = len(draws)
    X = onehot(draws)
    C = np.vstack([np.zeros((1, N_NUMBERS)), np.cumsum(X, axis=0)])  # C[s] = Σ_{r<s} X[r]
    s_idx = np.arange(n + 1)
    F = np.zeros((n + 1, N_NUMBERS, len(FEATURE_NAMES)))

    for k, w in enumerate((10, 50, 200)):
        lo = np.maximum(s_idx - w, 0)
        F[:, :, k] = (C - C[lo]) / w
    F[:, :, 3] = np.where(s_idx[:, None] > 0, C / np.maximum(s_idx, 1)[:, None], P_NUMBER)

    # Último índice r < s em que a dezena saiu
    seen = np.where(X > 0, np.arange(n)[:, None], -1)
    last = np.vstack([np.full((1, N_NUMBERS), -1), np.maximum.accumulate(seen, axis=0)]) if n else \
        np.full((1, N_NUMBERS), -1)
    gap = np.where(last >= 0, s_idx[:, None] - 1 - last, s_idx[:, None])
    F[:, :, 4] = np.log1p(gap)

    F[1:, :, 5] = X
    F[2:, :, 6] = X[:-1] if n >= 1 else 0

    lam = 0.5 ** (1 / 25)
    if n:
        ew = lfilter([1 - lam], [1, -lam], X, axis=0)  # ew[r] inclui X[r]
        F[1:, :, 7] = ew

    if n >= 2:
        O = X @ X.T                                     # sobreposição entre concursos
        # afinidade para s ≥ 2: Σ_{r < s-1} O[s-1, r] X[r]
        A = np.tril(O, k=-1) @ X                        # A[a] = Σ_{r<a} O[a,r] X[r]
        F[2:, :, 8] = A[1:] / np.arange(1, n)[:, None]
        # transição para s ≥ 2: Σ_{r=1}^{s-1} O[s-1, r-1] X[r]
        N = np.zeros_like(O)
        N[:, 1:] = O[:, :-1]
        T = np.tril(N, k=0) @ X                         # T[a] = Σ_{1≤r≤a} O[a,r-1] X[r]
        F[2:, :, 9] = T[1:] / np.arange(1, n)[:, None]
    return F


def labels(draws: np.ndarray) -> np.ndarray:
    """Y[s, i] = 1 se a dezena i+1 saiu no concurso s."""
    return onehot(draws)
