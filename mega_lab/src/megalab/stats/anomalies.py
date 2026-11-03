"""Detecção de anomalias: fatores conhecidos (H15), varredura por dezena,
mudança de regime e viés de posição na ordem de extração."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

from megalab.config import K_DRAWN, N_NUMBERS, P_NUMBER
from megalab.data.integrity import is_year_end_special, schedule_regime
from megalab.stats.montecarlo import benjamini_hochberg, mc_pvalue
from megalab.stats.randomness import _chi2_homogeneity, changepoint_profile, onehot

FACTOR_DESCRIPTIONS = {
    "H15a": "Dia da semana × dezena (permutação)",
    "H15b": "Sorteio de fim de ano (31/12 ou 01/01) × dezena (permutação)",
    "H15c": "Concurso acumulado × dezena (permutação)",
    "H15d": "Regime de calendário (1x/2x/3x por semana) × dezena (permutação)",
}


def factor_labels(df: pd.DataFrame, prizes: pd.DataFrame | None) -> dict[str, pd.Series]:
    """Rótulos de fatores conhecidos ANTES do sorteio, para cada concurso de ``df``."""
    d = pd.to_datetime(df["data"].replace("", pd.NA), errors="coerce")
    labels = {
        "H15a": d.dt.weekday.astype("Int64").astype(str),
        "H15b": is_year_end_special(df["data"]).map({True: "fim_de_ano", False: "regular"}),
        "H15d": schedule_regime(df["data"]),
    }
    if prizes is not None:
        # Acumulado = o concurso anterior não teve ganhador da sena.
        prev = prizes.set_index("concurso")["ganhadores_6"]
        acc = df["concurso"].map(lambda c: prev.get(c - 1, np.nan))
        labels["H15c"] = acc.map(lambda g: "nd" if pd.isna(g) else ("acumulado" if g == 0 else "normal"))
    return labels


def factor_test(X: np.ndarray, labels: pd.Series, permutations: int,
                rng: np.random.Generator) -> dict:
    """χ² de homogeneidade fator × dezena com p-valor por permutação dos rótulos."""
    lab = labels.to_numpy()
    keep = (lab != "nd") & pd.notna(lab) & (lab != "<NA>")
    Xk, lk = X[keep], lab[keep]
    groups, codes = np.unique(lk, return_inverse=True)
    if len(groups) < 2:
        return {"estatistica": float("nan"), "p": float("nan"), "grupos": {}}
    G = np.zeros((len(lk), len(groups)), dtype=np.float32)
    G[np.arange(len(lk)), codes] = 1
    obs = _chi2_homogeneity(G.T @ Xk)
    null = np.empty(permutations)
    for b in range(permutations):
        Gp = G[rng.permutation(len(lk))]
        null[b] = _chi2_homogeneity(Gp.T @ Xk)
    return {"estatistica": obs, "p": mc_pvalue(obs, null),
            "grupos": {str(g): int(c) for g, c in zip(groups, np.bincount(codes))}}


def number_scan(draws: np.ndarray) -> pd.DataFrame:
    """Frequência, z-escore, p-valor bilateral e q-valor (BH) de cada dezena.

    Sob H0 a contagem de cada dezena é Binomial(n, 0,1) exata (cada concurso
    inclui a dezena com probabilidade 0,1, independentemente dos outros).
    """
    n = len(draws)
    counts = onehot(draws).sum(axis=0).astype(int)
    E = n * P_NUMBER
    z = (counts - E) / np.sqrt(n * P_NUMBER * (1 - P_NUMBER))
    binom = sstats.binom(n, P_NUMBER)
    p = np.minimum(1.0, 2 * np.minimum(binom.cdf(counts), binom.sf(counts - 1)))
    return pd.DataFrame({"dezena": np.arange(1, N_NUMBERS + 1), "frequencia": counts,
                         "esperado": E, "z": z, "p_bilateral": p,
                         "q_bh": benjamini_hochberg(p)})


def position_table(order: np.ndarray) -> pd.DataFrame:
    """Contagem dezena × posição de extração e z-escores (H11)."""
    tab = np.zeros((N_NUMBERS, K_DRAWN), dtype=int)
    for pos in range(K_DRAWN):
        tab[:, pos] = np.bincount(order[:, pos] - 1, minlength=N_NUMBERS)
    m = len(order)
    e = m / N_NUMBERS
    sd = np.sqrt(m * (1 / N_NUMBERS) * (1 - 1 / N_NUMBERS))
    df = pd.DataFrame(tab, columns=[f"pos{i}" for i in range(1, 7)])
    df.insert(0, "dezena", np.arange(1, N_NUMBERS + 1))
    df["posicao_media"] = (tab * np.arange(1, 7)).sum(axis=1) / tab.sum(axis=1)
    df["z_max"] = (np.abs(tab - e) / sd).max(axis=1)
    return df


def changepoint_detail(draws: np.ndarray, concursos: np.ndarray) -> dict:
    X = onehot(draws)
    prof, best, cut = changepoint_profile(X)
    return {"chi2_max": best, "concurso_corte": int(concursos[cut]) if cut >= 0 else None,
            "perfil": prof.tolist()}
