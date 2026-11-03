"""Interface comum dos modelos.

Um modelo recebe o histórico ``(t, 6)`` de concursos anteriores (cópia
somente-leitura) e devolve um vetor ``p`` de 60 probabilidades marginais que
soma 6. Isso basta para gerar o bilhete (6 maiores ``p``) e uma distribuição
sobre conjuntos (ver :mod:`megalab.combinatorics`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from megalab.config import HEURISTIC_SHRINK, K_DRAWN, N_NUMBERS, P_NUMBER

P_MIN, P_MAX = 1e-4, 0.95


class Model(ABC):
    name: str = "modelo"
    description: str = ""

    @abstractmethod
    def predict_proba(self, history: np.ndarray) -> np.ndarray:
        """Vetor (60,) de probabilidades, soma 6, usando apenas ``history``."""


def normalize(p: np.ndarray) -> np.ndarray:
    """Projeta em [P_MIN, P_MAX] com soma 6 (iterativo, poucas passadas)."""
    p = np.asarray(p, dtype=float)
    p = np.where(np.isfinite(p), p, P_NUMBER)
    p = np.clip(p, P_MIN, None)
    p = p * (K_DRAWN / p.sum())
    for _ in range(20):
        over = p > P_MAX
        if not over.any():
            break
        excess = (p[over] - P_MAX).sum()
        p[over] = P_MAX
        free = ~over
        p[free] += excess * p[free] / p[free].sum()
    return p


def scores_to_proba(scores: np.ndarray, shrink: float = HEURISTIC_SHRINK) -> np.ndarray:
    """Converte escores não-negativos em probabilidades (soma 6), encolhidas para 0,1."""
    s = np.clip(np.asarray(scores, dtype=float), 0, None)
    if s.sum() <= 0:
        return np.full(N_NUMBERS, P_NUMBER)
    raw = K_DRAWN * s / s.sum()
    return normalize(P_NUMBER + shrink * (raw - P_NUMBER))


def ticket_from_proba(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """6 dezenas de maior probabilidade; empates desfeitos aleatoriamente."""
    order = np.lexsort((rng.random(N_NUMBERS), -np.asarray(p)))
    return np.sort(order[:K_DRAWN] + 1)
