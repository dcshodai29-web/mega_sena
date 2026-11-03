"""Motor de backtest walk-forward (cronológico, sem vazamento).

Para cada concurso de índice ``t`` (0-based) a partir do aquecimento:

- modelos estatísticos recebem ``draws[:t].copy()`` marcado como
  somente-leitura — não há como acessar o futuro;
- modelos de ML recebem ``F[:t+1]`` e ``Y[:t]`` (cópias), onde ``F`` é o
  tensor causal de features (ver :mod:`megalab.models.features`);
- o modelo ``hedge`` combina os demais com pesos ∝ exp(log-verossimilhança
  acumulada até ``t-1``) — aprende com os erros sem tocar no passado.

Nenhum hiperparâmetro é ajustado aqui.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import zlib

import numpy as np
import pandas as pd

from megalab import config
from megalab.combinatorics import set_log_likelihood_ratio
from megalab.evaluation import metrics
from megalab.models.base import Model, normalize, ticket_from_proba
from megalab.models.features import feature_tensor, labels
from megalab.models.ml import MLModel, ml_models
from megalab.models.statistical import statistical_models
from megalab.stats.montecarlo import holm

HEDGE = "hedge"


def default_models(seed: int = config.SEED) -> list[Model]:
    return statistical_models() + ml_models(seed)


@dataclass
class BacktestResult:
    concursos: np.ndarray                 # concursos avaliados
    draws: np.ndarray                     # resultados reais avaliados (T, 6)
    proba: dict[str, np.ndarray]          # (T, 60) por modelo
    tickets: dict[str, np.ndarray]        # (T, 6) por modelo
    random_players_hits: np.ndarray = field(repr=False, default=None)  # (R, T)

    @property
    def models(self) -> list[str]:
        return list(self.proba)

    def frame(self) -> pd.DataFrame:
        rows = []
        for m in self.models:
            h = metrics.hits(self.tickets[m], self.draws)
            b = metrics.bits(self.proba[m], self.draws)
            rows.append(pd.DataFrame({"concurso": self.concursos, "modelo": m, "acertos": h, "bits": b}))
        return pd.concat(rows, ignore_index=True)


def hedge_weights(llr_history: np.ndarray) -> np.ndarray:
    """Pesos de mistura bayesiana a partir do log-score acumulado (nats)."""
    s = llr_history.sum(axis=0)
    w = np.exp(s - s.max())
    return w / w.sum()


def run_backtest(draws: np.ndarray, concursos: np.ndarray, start_concurso: int = config.BACKTEST_WARMUP_END + 1,
                 models: list[Model] | None = None, seed: int = config.SEED,
                 random_players: int = 1000, progress=None) -> BacktestResult:
    draws = np.asarray(draws, dtype=np.int64)
    models = models if models is not None else default_models(seed)
    n = len(draws)
    t0 = int(np.searchsorted(concursos, start_concurso))
    T = n - t0
    rng = np.random.default_rng(seed)
    F_full = feature_tensor(draws)
    Y_full = labels(draws)

    proba = {m.name: np.empty((T, config.N_NUMBERS)) for m in models}
    for m in models:
        if isinstance(m, MLModel):
            m.reset()
    for i, t in enumerate(range(t0, n)):
        hist = draws[:t].copy()
        hist.setflags(write=False)
        for m in models:
            if isinstance(m, MLModel):
                p = m.predict_from_features(F_full[: t + 1].copy(), Y_full[:t].copy(), t)
            else:
                p = m.predict_proba(hist)
            proba[m.name][i] = p
        if progress and i % 250 == 0:
            progress(i, T)

    # Hedge: mistura dos modelos não-baseline, pesos pelo desempenho passado
    members = [m.name for m in models if m.name != "aleatorio"]
    if members:
        target = draws[t0:]
        llr = np.stack([set_log_likelihood_ratio(proba[k], target - 1) for k in members], axis=1)
        cum = np.vstack([np.zeros((1, len(members))), np.cumsum(llr, axis=0)])
        stack = np.stack([proba[k] for k in members], axis=1)  # (T, K, 60)
        ph = np.empty((T, config.N_NUMBERS))
        for i in range(T):
            w = hedge_weights(cum[i][None, :])  # só concursos < i
            ph[i] = normalize(w @ stack[i])
        proba[HEDGE] = ph

    tickets = {}
    for name, P in proba.items():
        trng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        if name == "aleatorio":
            tickets[name] = np.sort(np.stack([trng.choice(60, 6, replace=False) + 1 for _ in range(T)]), axis=1)
        else:
            tickets[name] = np.stack([ticket_from_proba(P[i], trng) for i in range(T)])

    target = draws[t0:]
    rp = None
    if random_players:
        prng = np.random.default_rng([seed, 7])
        Xt = np.zeros((T, config.N_NUMBERS), dtype=np.int8)
        Xt[np.arange(T)[:, None], target - 1] = 1
        rp = np.empty((random_players, T), dtype=np.int8)
        for r in range(random_players):
            tk = np.argpartition(prng.random((T, config.N_NUMBERS)), 5, axis=1)[:, :6]
            rp[r] = np.take_along_axis(Xt, tk, axis=1).sum(axis=1)
    return BacktestResult(concursos[t0:], target, proba, tickets, rp)


def evaluate(res: BacktestResult, periods: dict[str, tuple[int, int | None]] | None = None) -> dict:
    """Testes de significância por modelo e período + Holm no holdout."""
    periods = periods or config.BACKTEST_PERIODS
    out: dict = {}
    last = int(res.concursos[-1])
    for pname, (a, b) in periods.items():
        b = last if b is None else b
        mask = (res.concursos >= a) & (res.concursos <= b)
        if not mask.any():
            continue
        d = res.draws[mask]
        rows = {}
        rp_tot = res.random_players_hits[:, mask].sum(axis=1) if res.random_players_hits is not None else None
        for m in res.models:
            h = metrics.hits(res.tickets[m][mask], d)
            ht = metrics.hits_test(h)
            bt = metrics.bits_test(res.proba[m][mask], d)
            ev = metrics.log10_evalue(res.proba[m][mask], d)
            row = {**ht, **{f"bits_{k}" if not k.startswith("bits") else k: v for k, v in bt.items()},
                   "log10_evalor_final": float(ev[-1]), "log10_evalor_max": float(ev.max())}
            if rp_tot is not None:
                row["percentil_entre_aleatorios"] = float((rp_tot < ht["acertos_total"]).mean()
                                                          + 0.5 * (rp_tot == ht["acertos_total"]).mean())
            rows[m] = row
        out[pname] = {"concursos": [a, b], "n": int(mask.sum()), "modelos": rows}
        if rp_tot is not None:
            out[pname]["jogadores_aleatorios"] = {
                "n": int(len(rp_tot)), "acertos_media": float(rp_tot.mean()),
                "p2_5": float(np.percentile(rp_tot, 2.5)), "p97_5": float(np.percentile(rp_tot, 97.5)),
                "min": int(rp_tot.min()), "max": int(rp_tot.max())}
    if "holdout" in out:
        fam = {}
        for m, r in out["holdout"]["modelos"].items():
            if m == "aleatorio":
                continue
            fam[f"{m}:acertos"] = r["p_unilateral"]
            fam[f"{m}:bits"] = r["bits_p_unilateral"]
        adj = holm(fam)
        out["holdout"]["holm"] = adj
        winners = sorted({k.split(":")[0] for k, v in adj.items() if v < config.ALPHA})
        out["holdout"]["modelos_significativos"] = winners
    return out
