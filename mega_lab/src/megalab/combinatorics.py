"""Probabilidades exatas da Mega-Sena e utilidades combinatórias."""

from __future__ import annotations

from math import comb, log

import numpy as np
from scipy import stats

from megalab.config import K_DRAWN, N_NUMBERS

TOTAL_COMBINATIONS = comb(N_NUMBERS, K_DRAWN)  # 50.063.860


def hits_pmf(n: int = N_NUMBERS, k: int = K_DRAWN, ticket: int = K_DRAWN) -> np.ndarray:
    """P(acertos = j) para um bilhete de ``ticket`` dezenas — hipergeométrica."""
    return stats.hypergeom(n, k, ticket).pmf(np.arange(0, min(k, ticket) + 1))


def prize_odds() -> dict[str, float]:
    """Probabilidade de sena, quina e quadra com uma aposta simples."""
    pmf = hits_pmf()
    return {"sena": pmf[6], "quina": pmf[5], "quadra": pmf[4]}


def sum_of_iid_pmf(pmf: np.ndarray, times: int) -> np.ndarray:
    """Distribuição exata da soma de ``times`` cópias i.i.d. (convolução por quadrados)."""
    result = np.array([1.0])
    base = np.asarray(pmf, dtype=float)
    t = times
    while t > 0:
        if t & 1:
            result = np.convolve(result, base)
        t >>= 1
        if t:
            base = np.convolve(base, base)
        # Remove ruído numérico desprezível nas caudas
        result = np.clip(result, 0, None)
        base = np.clip(base, 0, None)
    return result / result.sum()


def upper_tail_p(total_hits: int, n_draws: int) -> float:
    """P(soma dos acertos ≥ total_hits) em ``n_draws`` concursos sob H0 (exato)."""
    pmf = sum_of_iid_pmf(hits_pmf(), n_draws)
    if total_hits <= 0:
        return 1.0
    if total_hits >= len(pmf):
        return 0.0
    return float(min(1.0, pmf[total_hits:].sum()))


def elementary_symmetric(weights: np.ndarray, k: int = K_DRAWN) -> np.ndarray:
    """Polinômios simétricos elementares e_0..e_k dos pesos (programação dinâmica).

    Aceita matriz (m, 60): calcula para cada linha.
    """
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    e = np.zeros((w.shape[0], k + 1))
    e[:, 0] = 1.0
    for j in range(w.shape[1]):
        wj = w[:, j][:, None]
        e[:, 1:] = e[:, 1:] + wj * e[:, :-1]
    return e


def set_log_likelihood_ratio(p: np.ndarray, drawn_idx: np.ndarray) -> np.ndarray:
    """log [P_modelo(S) / P_uniforme(S)] sob o modelo de Bernoulli condicional.

    ``p``: (m, 60) probabilidades marginais (soma 6 por linha, 0 < p < 1).
    ``drawn_idx``: (m, 6) índices 0..59 das dezenas sorteadas.
    O modelo de conjunto é P(S) = Π_{i∈S} w_i / e_6(w), w_i = p_i / (1 - p_i).
    Retorna log natural (m,).
    """
    p = np.clip(np.atleast_2d(p), 1e-9, 1 - 1e-9)
    w = p / (1 - p)
    w = w / w.mean(axis=1, keepdims=True)  # escala não altera P(S); evita overflow
    e6 = elementary_symmetric(w)[:, K_DRAWN]
    logw = np.log(w)
    num = np.take_along_axis(logw, np.atleast_2d(drawn_idx), axis=1).sum(axis=1)
    return num - np.log(e6) + log(TOTAL_COMBINATIONS)


def set_llr_null_moments(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Média e variância exatas de :func:`set_log_likelihood_ratio` sob H0.

    Sob sorteio uniforme sem reposição, Σ_{i∈S} log w_i tem média 6·mean(log w)
    e variância 6·var(log w)·(N-6)/(N-1) (correção de população finita).
    """
    p = np.clip(np.atleast_2d(p), 1e-9, 1 - 1e-9)
    w = p / (1 - p)
    w = w / w.mean(axis=1, keepdims=True)
    logw = np.log(w)
    e6 = elementary_symmetric(w)[:, K_DRAWN]
    n = w.shape[1]
    mean = K_DRAWN * logw.mean(axis=1) - np.log(e6) + log(TOTAL_COMBINATIONS)
    var = K_DRAWN * logw.var(axis=1) * (n - K_DRAWN) / (n - 1)
    return mean, var


def pair_overlap_pmf() -> np.ndarray:
    """P(dois concursos independentes compartilharem j dezenas), j = 0..6."""
    return hits_pmf()
