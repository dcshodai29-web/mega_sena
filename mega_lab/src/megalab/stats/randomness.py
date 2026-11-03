"""Bateria pré-registrada de testes de aleatoriedade (H01–H14).

Cada estatística é definida de modo que **valores maiores = mais anômalo**.
Os p-valores vêm de Monte Carlo sob H0 (:func:`run_battery`), usando o mesmo
código para os dados reais e para os históricos simulados — o que garante
que qualquer artefato de cálculo afeta igualmente os dois lados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np
from scipy import stats as sstats

from megalab.combinatorics import hits_pmf
from megalab.config import K_DRAWN, N_NUMBERS, P_NUMBER
from megalab.stats.montecarlo import mc_pvalue, simulate_draws

DESCRIPTIONS = {
    "H01": "Uniformidade das 60 dezenas (χ²)",
    "H02": "Dezena individual extrema (max |z|)",
    "H03": "Co-ocorrência de pares (χ², 1.770 pares)",
    "H04": "Trio mais frequente (máximo entre 34.220 trios)",
    "H05": "Pares de concursos com ≥ 4 dezenas em comum",
    "H06": "Repetição de dezenas do concurso anterior (χ²)",
    "H07": "Transições dezena→dezena entre concursos (χ², 60×60)",
    "H08": "Intervalos entre aparições vs. geométrica (χ²)",
    "H09": "Memória na soma das dezenas (Ljung-Box, 10 lags)",
    "H10a": "Quantidade de dezenas pares por concurso (χ²)",
    "H10b": "Quantidade de dezenas baixas 1–30 por concurso (χ²)",
    "H10c": "Contagens por faixa 1–10 … 51–60 (χ²)",
    "H10d": "Contagens por dígito final 0–9 (χ²)",
    "H11": "Dezena × posição na ordem de extração (χ², 60×6)",
    "H12": "Pares de dezenas consecutivas por concurso (χ²)",
    "H13": "Mudança de regime: máximo χ² antes×depois",
    "H14": "Deriva: variância do χ² em janelas de 200",
}

CHANGEPOINT_MARGIN = 200
CHANGEPOINT_STEP = 10
DRIFT_WINDOW = 200
LJUNG_BOX_LAGS = 10
GAP_MAX = 30


def onehot(draws: np.ndarray) -> np.ndarray:
    n = len(draws)
    X = np.zeros((n, N_NUMBERS), dtype=np.float32)
    X[np.arange(n)[:, None], draws - 1] = 1.0
    return X


def _chi2(obs: np.ndarray, exp: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    exp = np.asarray(exp, dtype=float)
    m = exp > 0
    return float(np.sum((obs[m] - exp[m]) ** 2 / exp[m]))


def _chi2_homogeneity(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    rows = table.sum(axis=1, keepdims=True)
    cols = table.sum(axis=0, keepdims=True)
    exp = rows @ cols / table.sum()
    m = exp > 0
    return float(np.sum((table[m] - exp[m]) ** 2 / exp[m]))


# --- distribuições exatas auxiliares -----------------------------------------------

_HG = hits_pmf()  # P(j dezenas em comum) entre dois conjuntos de 6 — também para H06
_HALF = sstats.hypergeom(N_NUMBERS, 30, K_DRAWN).pmf(np.arange(7))  # H10a/H10b
_GEOM = np.array([(1 - P_NUMBER) ** (k - 1) * P_NUMBER for k in range(1, GAP_MAX + 1)]
                 + [(1 - P_NUMBER) ** GAP_MAX])
# nº de subconjuntos de 6 em {1..60} com exatamente j adjacências: C(5,j)·C(55,6-j)
_CONSEC = np.array([comb(K_DRAWN - 1, j) * comb(N_NUMBERS - K_DRAWN + 1, K_DRAWN - j)
                    for j in range(K_DRAWN)], dtype=float)
_CONSEC = _CONSEC / _CONSEC.sum()

_TRI = np.array([(i, j, k) for i in range(6) for j in range(i + 1, 6) for k in range(j + 1, 6)])


# --- estatísticas -----------------------------------------------------------------

def statistics(draws: np.ndarray, order: np.ndarray | None = None) -> dict[str, float]:
    """Calcula H01–H14 para uma matriz (n, 6) de dezenas ordenadas."""
    draws = np.asarray(draws, dtype=np.int64)
    n = len(draws)
    X = onehot(draws)
    counts = X.sum(axis=0)
    E = n * P_NUMBER
    out: dict[str, float] = {}

    out["H01"] = _chi2(counts, np.full(N_NUMBERS, E))
    out["H02"] = float(np.max(np.abs(counts - E) / np.sqrt(n * P_NUMBER * (1 - P_NUMBER))))

    P = X.T @ X
    iu = np.triu_indices(N_NUMBERS, 1)
    e_pair = n * (K_DRAWN * (K_DRAWN - 1)) / (N_NUMBERS * (N_NUMBERS - 1))
    out["H03"] = _chi2(P[iu], np.full(len(iu[0]), e_pair))

    d0 = draws - 1
    tri = d0[:, _TRI[:, 0]] * 3600 + d0[:, _TRI[:, 1]] * 60 + d0[:, _TRI[:, 2]]
    out["H04"] = float(np.bincount(tri.ravel(), minlength=216000).max())

    O = X @ X.T
    out["H05"] = float((np.count_nonzero(O >= 4) - n) / 2)

    rep = (X[1:] * X[:-1]).sum(axis=1).astype(int)
    rep_c = np.bincount(np.minimum(rep, 3), minlength=4)
    rep_e = (n - 1) * np.array([_HG[0], _HG[1], _HG[2], _HG[3:].sum()])
    out["H06"] = _chi2(rep_c, rep_e)

    T = X[:-1].T @ X[1:]
    out["H07"] = _chi2(T.ravel(), np.full(N_NUMBERS ** 2, (n - 1) * P_NUMBER ** 2))

    gaps = []
    for j in range(N_NUMBERS):
        idx = np.flatnonzero(X[:, j])
        if len(idx) > 1:
            gaps.append(np.diff(idx))
    g = np.concatenate(gaps) if gaps else np.array([], dtype=int)
    g_c = np.bincount(np.minimum(g, GAP_MAX + 1), minlength=GAP_MAX + 2)[1:]
    out["H08"] = _chi2(g_c, len(g) * _GEOM)

    s = draws.sum(axis=1).astype(float)
    s = s - s.mean()
    denom = np.sum(s ** 2)
    r = np.array([np.sum(s[k:] * s[:-k]) / denom for k in range(1, LJUNG_BOX_LAGS + 1)])
    out["H09"] = float(n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, LJUNG_BOX_LAGS + 1))))

    evens = (draws % 2 == 0).sum(axis=1)
    out["H10a"] = _chi2(np.bincount(evens, minlength=7), n * _HALF)
    lows = (draws <= 30).sum(axis=1)
    out["H10b"] = _chi2(np.bincount(lows, minlength=7), n * _HALF)
    dec = np.bincount(((draws - 1) // 10).ravel(), minlength=6)
    out["H10c"] = _chi2(dec, np.full(6, n * K_DRAWN / 6))
    unit = np.bincount((draws % 10).ravel(), minlength=10)
    out["H10d"] = _chi2(unit, np.full(10, n * K_DRAWN / 10))

    if order is not None and len(order):
        tab = np.zeros((N_NUMBERS, K_DRAWN))
        for pos in range(K_DRAWN):
            tab[:, pos] = np.bincount(order[:, pos] - 1, minlength=N_NUMBERS)
        out["H11"] = _chi2(tab.ravel(), np.full(tab.size, len(order) / N_NUMBERS))
    else:
        out["H11"] = float("nan")

    consec = (np.diff(draws, axis=1) == 1).sum(axis=1)
    out["H12"] = _chi2(np.bincount(consec, minlength=6)[:6], n * _CONSEC)

    out["H13"] = changepoint_profile(X)[1]

    nw = n // DRIFT_WINDOW
    if nw >= 2:
        w = counts_by_window(X, DRIFT_WINDOW)
        chis = ((w - DRIFT_WINDOW * P_NUMBER) ** 2 / (DRIFT_WINDOW * P_NUMBER)).sum(axis=1)
        out["H14"] = float(np.var(chis))
    else:
        out["H14"] = float("nan")
    return out


def counts_by_window(X: np.ndarray, window: int) -> np.ndarray:
    nw = len(X) // window
    return X[: nw * window].reshape(nw, window, N_NUMBERS).sum(axis=1)


def changepoint_profile(X: np.ndarray) -> tuple[np.ndarray, float, int]:
    """χ² de homogeneidade antes × depois para cada ponto de corte.

    Retorna (perfil, máximo, índice do corte de máximo — 0-based, primeiro
    concurso do segmento "depois").
    """
    n = len(X)
    cuts = np.arange(CHANGEPOINT_MARGIN, n - CHANGEPOINT_MARGIN + 1, CHANGEPOINT_STEP)
    if len(cuts) == 0:
        return np.array([]), float("nan"), -1
    C = np.vstack([np.zeros((1, N_NUMBERS)), np.cumsum(X, axis=0)])
    total = C[-1]
    before = C[cuts]
    after = total[None, :] - before
    tot = before.sum(axis=1, keepdims=True) + after.sum(axis=1, keepdims=True)
    col = (before + after)
    eb = before.sum(axis=1, keepdims=True) * col / tot
    ea = after.sum(axis=1, keepdims=True) * col / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.nansum((before - eb) ** 2 / eb + (after - ea) ** 2 / ea, axis=1)
    k = int(np.argmax(chi))
    return chi, float(chi[k]), int(cuts[k])


# --- execução com Monte Carlo ---------------------------------------------------------

@dataclass
class BatteryResult:
    n: int
    observed: dict[str, float]
    pvalues: dict[str, float]
    null: dict[str, np.ndarray] = field(repr=False)


def run_battery(draws: np.ndarray, order: np.ndarray | None, replicates: int,
                rng: np.random.Generator) -> BatteryResult:
    """Estatísticas observadas + p-valores de Monte Carlo com ``replicates`` réplicas.

    ``order``: matriz (m, 6) da ordem de extração para os concursos que a têm
    (os últimos m concursos sem ordem são ignorados apenas no H11). As
    réplicas simulam ordem para o mesmo número m de concursos.
    """
    observed = statistics(draws, order)
    m = 0 if order is None else len(order)
    null = {k: np.empty(replicates) for k in observed}
    for b in range(replicates):
        sim, sim_order = simulate_draws(len(draws), rng)
        st = statistics(sim, sim_order[:m] if m else None)
        for k, v in st.items():
            null[k][b] = v
    pvals = {k: (mc_pvalue(v, null[k]) if np.isfinite(v) else float("nan"))
             for k, v in observed.items()}
    return BatteryResult(len(draws), observed, pvals, null)
