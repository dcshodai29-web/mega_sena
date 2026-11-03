"""Família 1 do protocolo: aleatoriedade, anomalias, Monte Carlo e poder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from megalab import config
from megalab.data.store import Base, O_COLS
from megalab.stats import anomalies, descriptive, randomness
from megalab.stats.montecarlo import holm, simulate_draws

TEST_IDS = list(randomness.DESCRIPTIONS) + list(anomalies.FACTOR_DESCRIPTIONS)
POWER_WEIGHTS = (1.05, 1.10, 1.15, 1.20, 1.30)


def _order_block(df: pd.DataFrame) -> np.ndarray | None:
    """Ordem de extração dos concursos que a têm (devem formar um prefixo contíguo)."""
    has = df[O_COLS].notna().all(axis=1).to_numpy()
    m = int(has.sum())
    if m == 0:
        return None
    if not has[:m].all():
        raise ValueError("Concursos com ordem de extração não formam um bloco inicial contíguo")
    return df.loc[: m - 1, O_COLS].to_numpy(dtype=np.int64)


def analyze_period(base: Base, prizes: pd.DataFrame | None, replicates: int,
                   permutations: int, rng: np.random.Generator, keep_null: bool = False) -> dict:
    draws = base.draws
    order = _order_block(base.df)
    bat = randomness.run_battery(draws, order, replicates, rng)
    X = randomness.onehot(draws)
    labels = anomalies.factor_labels(base.df, prizes)
    factors = {}
    for k in anomalies.FACTOR_DESCRIPTIONS:
        if k in labels:
            factors[k] = anomalies.factor_test(X, labels[k], permutations, rng)
    obs = dict(bat.observed)
    pv = dict(bat.pvalues)
    for k, f in factors.items():
        obs[k] = f["estatistica"]
        pv[k] = f["p"]
    null_summary = {k: {"media": float(np.nanmean(v)), "p95": float(np.nanpercentile(v, 95))}
                    for k, v in bat.null.items() if np.isfinite(v).any()}
    out = {
        "concursos": [int(base.concursos[0]), int(base.concursos[-1])], "n": int(len(draws)),
        "n_com_ordem": 0 if order is None else int(len(order)),
        "observado": obs, "p": pv, "nula": null_summary,
        "grupos_fatores": {k: f["grupos"] for k, f in factors.items()},
    }
    if keep_null:
        out["_null"] = bat.null
    return out


def power_analysis(n: int, null_h01: np.ndarray, null_h02: np.ndarray, rng: np.random.Generator,
                   reps: int = 200, weights=POWER_WEIGHTS) -> list[dict]:
    """Poder para detectar UMA bola com peso relativo w (H01 e H02, α = 5 %)."""
    c01 = np.percentile(null_h01, 95)
    c02 = np.percentile(null_h02, 95)
    E = n * config.P_NUMBER
    sd = np.sqrt(n * config.P_NUMBER * (1 - config.P_NUMBER))
    rows = []
    for w in weights:
        hit01 = hit02 = 0
        freq_ratio = []
        for _ in range(reps):
            wt = np.ones(config.N_NUMBERS)
            j = rng.integers(config.N_NUMBERS)
            wt[j] = w
            sim, _ = simulate_draws(n, rng, wt)
            counts = np.bincount(sim.ravel() - 1, minlength=config.N_NUMBERS)
            h01 = np.sum((counts - E) ** 2 / E)
            h02 = np.max(np.abs(counts - E) / sd)
            hit01 += h01 >= c01
            hit02 += h02 >= c02
            freq_ratio.append(counts[j] / E)
        rows.append({"peso": w, "freq_relativa_media": float(np.mean(freq_ratio)),
                     "poder_H01": float(hit01 / reps), "poder_H02": float(hit02 / reps)})
    return rows


def naive_patterns(base: Base, null: dict[str, np.ndarray]) -> dict:
    """Quanto "padrão" a análise ingênua acha no acaso puro, comparado ao real."""
    draws = base.draws
    n = len(draws)
    X = randomness.onehot(draws)
    counts = X.sum(axis=0)
    P = X.T @ X
    iu = np.triu_indices(config.N_NUMBERS, 1)
    real = {
        "dezena_mais_frequente": int(counts.max()), "dezena_menos_frequente": int(counts.min()),
        "par_mais_frequente": int(P[iu].max()), "trio_mais_frequente": int(randomness.statistics(draws)["H04"]),
    }
    return {"n": n, "real": real, "esperado_por_dezena": n * config.P_NUMBER,
            "esperado_por_par": n / 118.0,
            "nulo_trio_max": {"media": float(np.mean(null["H04"])),
                              "p5": float(np.percentile(null["H04"], 5)),
                              "p95": float(np.percentile(null["H04"], 95))}}


def run_family1(base: Base, prizes: pd.DataFrame | None, replicates: int = config.MC_REPLICATES,
                permutations: int = config.PERMUTATIONS, seed: int = config.SEED,
                power_reps: int = 200) -> dict:
    ss = np.random.SeedSequence(seed)
    r_disc, r_conf, r_full, r_pow = (np.random.default_rng(s) for s in ss.spawn(4))
    d0, d1 = config.DISCOVERY
    disc = analyze_period(base.between(d0, d1), prizes, replicates, permutations, r_disc)
    conf = analyze_period(base.between(config.CONFIRMATION_START, None), prizes, replicates,
                          permutations, r_conf)
    full = analyze_period(base, prizes, replicates, permutations, r_full, keep_null=True)
    null_full = full.pop("_null")

    p_disc = {k: v for k, v in disc["p"].items() if k in TEST_IDS and np.isfinite(v)}
    p_holm = holm(p_disc)
    verdicts = []
    for k in TEST_IDS:
        pd_ = disc["p"].get(k, float("nan"))
        ph = p_holm.get(k, float("nan"))
        pc = conf["p"].get(k, float("nan"))
        cand = bool(np.isfinite(ph) and ph < config.ALPHA)
        confirmed = bool(cand and np.isfinite(pc) and pc < config.ALPHA)
        if confirmed:
            v = "ANOMALIA CONFIRMADA"
        elif cand:
            v = "candidata em D, não confirmada em C"
        else:
            v = "compatível com o acaso"
        desc = randomness.DESCRIPTIONS.get(k) or anomalies.FACTOR_DESCRIPTIONS.get(k)
        verdicts.append({"id": k, "descricao": desc, "p_descoberta": pd_, "p_holm_descoberta": ph,
                         "p_confirmacao": pc, "p_completo_exploratorio": full["p"].get(k, float("nan")),
                         "veredito": v})

    scan_full = anomalies.number_scan(base.draws)
    scan_disc = anomalies.number_scan(base.between(d0, d1).draws)
    scan_conf = anomalies.number_scan(base.between(config.CONFIRMATION_START, None).draws)
    order_full = _order_block(base.df)
    return {
        "periodos": {"descoberta": disc, "confirmacao": conf, "completo": full},
        "hipoteses": verdicts,
        "dezenas": {"completo": scan_full.to_dict(orient="records"),
                    "descoberta": scan_disc.to_dict(orient="records"),
                    "confirmacao": scan_conf.to_dict(orient="records")},
        "posicao": anomalies.position_table(order_full).to_dict(orient="records") if order_full is not None else [],
        "mudanca_regime": anomalies.changepoint_detail(base.draws, base.concursos),
        "poder": power_analysis(len(base.draws), null_full["H01"], null_full["H02"], r_pow,
                                reps=power_reps),
        "padroes_ingenuos": naive_patterns(base, null_full),
        "descritiva": descriptive.summary(base.draws),
        "_null_completo": null_full,
    }
