"""Estatística descritiva (sem inferência): frequências, atrasos, somas, pares."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from megalab.combinatorics import hits_pmf
from megalab.config import K_DRAWN, N_NUMBERS


def current_gaps(draws: np.ndarray) -> np.ndarray:
    """Atraso atual de cada dezena: concursos desde a última aparição."""
    n = len(draws)
    last = np.full(N_NUMBERS, -1)
    for t, row in enumerate(draws):
        last[row - 1] = t
    return np.where(last >= 0, n - 1 - last, n)


def summary(draws: np.ndarray) -> dict:
    n = len(draws)
    sums = draws.sum(axis=1)
    evens = (draws % 2 == 0).sum(axis=1)
    rep = np.array([len(set(a) & set(b)) for a, b in zip(draws[1:], draws[:-1])])
    pairs = Counter(p for row in draws for p in combinations(row.tolist(), 2))
    top_pairs = pairs.most_common(10)
    return {
        "n": int(n),
        "soma": {"media": float(sums.mean()), "dp": float(sums.std(ddof=1)),
                 "teorica_media": K_DRAWN * (N_NUMBERS + 1) / 2,
                 "min": int(sums.min()), "max": int(sums.max())},
        "pares_por_concurso": {str(k): int(v) for k, v in
                               zip(*np.unique(evens, return_counts=True))},
        "repeticao_anterior": {"observado": {str(k): int(v) for k, v in
                                             zip(*np.unique(rep, return_counts=True))},
                               "esperado": {str(k): float((n - 1) * p)
                                            for k, p in enumerate(hits_pmf())}},
        "pares_mais_frequentes": [{"par": list(p), "vezes": int(c)} for p, c in top_pairs],
        "pares_esperado_por_par": float(n * K_DRAWN * (K_DRAWN - 1) / (N_NUMBERS * (N_NUMBERS - 1))),
        "combinacoes_repetidas_6": int(n - len({tuple(r) for r in draws.tolist()})),
    }


def frequency_table(draws: np.ndarray) -> pd.DataFrame:
    counts = np.bincount(draws.ravel() - 1, minlength=N_NUMBERS)
    return pd.DataFrame({"dezena": np.arange(1, N_NUMBERS + 1), "frequencia": counts,
                         "atraso_atual": current_gaps(draws)})
