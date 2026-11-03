"""Simulador de sorteios (justos ou com bolas viciadas) e utilidades de Monte Carlo."""

from __future__ import annotations

import numpy as np

from megalab.config import K_DRAWN, N_NUMBERS


def simulate_draws(n: int, rng: np.random.Generator, weights: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Simula ``n`` concursos. Retorna (ordenadas (n,6), ordem de extração (n,6)).

    Sem ``weights``: amostragem uniforme sem reposição. Com ``weights`` (60,):
    extração sequencial proporcional ao peso de cada bola entre as restantes
    (modelo de Plackett-Luce, implementado pelo truque Gumbel-top-k) — um
    modelo simples de bola mais "pesada" ou "leve".
    """
    keys = rng.gumbel(size=(n, N_NUMBERS))
    if weights is not None:
        keys = keys + np.log(np.asarray(weights, dtype=float))[None, :]
    top = np.argpartition(-keys, K_DRAWN - 1, axis=1)[:, :K_DRAWN]
    order_idx = np.argsort(-np.take_along_axis(keys, top, axis=1), axis=1)
    order = np.take_along_axis(top, order_idx, axis=1) + 1
    return np.sort(order, axis=1), order


def mc_pvalue(observed: float, null_samples: np.ndarray) -> float:
    """p-valor de Monte Carlo unilateral (cauda superior), com correção +1."""
    null_samples = np.asarray(null_samples, dtype=float)
    return float((1 + np.sum(null_samples >= observed - 1e-12)) / (1 + len(null_samples)))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Ajuste de Holm-Bonferroni (controle de FWER)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return adjusted


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """q-valores de Benjamini-Hochberg (controle de FDR)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out
