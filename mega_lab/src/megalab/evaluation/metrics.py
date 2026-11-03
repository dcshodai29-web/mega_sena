"""Métricas de previsão e suas distribuições nulas exatas."""

from __future__ import annotations

from math import log

import numpy as np
from scipy import stats as sstats

from megalab.combinatorics import (hits_pmf, set_llr_null_moments, set_log_likelihood_ratio,
                                   upper_tail_p)
from megalab.config import EVALUE_MIX

LN2 = log(2)


def hits(tickets: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Acertos por concurso: |bilhete ∩ sorteio|."""
    return (tickets[:, :, None] == draws[:, None, :]).any(axis=2).sum(axis=1)


def bits(p: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Log-score de conjunto em bits contra o uniforme (positivo = melhor que o acaso)."""
    return set_log_likelihood_ratio(p, draws - 1) / LN2


def log10_evalue(p: np.ndarray, draws: np.ndarray, mix: float = EVALUE_MIX) -> np.ndarray:
    """Trajetória do log10 do e-valor Π (1-mix + mix·LR_t). Supermartingal sob H0."""
    lr = np.exp(set_log_likelihood_ratio(p, draws - 1))
    return np.cumsum(np.log10((1 - mix) + mix * lr))


def hits_test(h: np.ndarray) -> dict:
    """Teste exato: soma de acertos de um bilhete fixado antes do sorteio é soma de
    hipergeométricas i.i.d. sob H0, qualquer que seja a estratégia."""
    T = len(h)
    tot = int(h.sum())
    pmf = hits_pmf()
    return {"concursos": T, "acertos_total": tot, "acertos_media": float(h.mean()),
            "acertos_esperado": float(T * 0.6), "p_unilateral": upper_tail_p(tot, T),
            "quadras": int((h == 4).sum()), "quinas": int((h == 5).sum()), "senas": int((h == 6).sum()),
            "quadras_esperado": float(T * pmf[4]), "quinas_esperado": float(T * pmf[5]),
            "senas_esperado": float(T * pmf[6])}


def bits_test(p: np.ndarray, draws: np.ndarray) -> dict:
    """Teste condicional do log-score: as previsões ficam fixas e o sorteio é uniforme
    sob H0. Usa momentos exatos por concurso e aproximação normal da soma."""
    b = bits(p, draws)
    mu, var = set_llr_null_moments(p)
    mu, var = mu / LN2, var / LN2 ** 2
    sd = np.sqrt(var.sum())
    z = (b.sum() - mu.sum()) / sd if sd > 0 else 0.0
    return {"bits_media": float(b.mean()), "bits_total": float(b.sum()),
            "bits_nulo_media": float(mu.mean()), "z": float(z),
            "p_unilateral": float(sstats.norm.sf(z)) if sd > 0 else 1.0}
