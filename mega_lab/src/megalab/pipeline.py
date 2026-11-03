"""Etapas reprodutíveis do laboratório (usadas pela CLI e pelo relatório)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from megalab import config
from megalab.analysis import run_family1
from megalab.data import store
from megalab.evaluation import backtest as bt
from megalab.models.ml import ml_models
from megalab.models.statistical import statistical_models
from megalab.registry import ledger as lg

FAMILY1_JSON = config.REPORTS_DIR / "familia1.json"
FAMILY1_NULL = config.REPORTS_DIR / "familia1_nula.npz"
BACKTEST_JSON = config.REPORTS_DIR / "backtest.json"
BACKTEST_CSV = config.REPORTS_DIR / "backtest_por_concurso.csv.gz"
BACKTEST_STATE = config.REPORTS_DIR / "backtest_estado.json"


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(type(o))


def _clean(obj):
    """Substitui NaN/inf por None recursivamente (JSON estrito)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False, default=_default) + "\n",
                    encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze(replicates: int = config.MC_REPLICATES, permutations: int = config.PERMUTATIONS,
            power_reps: int = 200) -> dict:
    base = store.load()
    t = time.time()
    res = run_family1(base, store.load_prizes(), replicates, permutations, power_reps=power_reps)
    null = res.pop("_null_completo")
    res["meta"] = {"base_sha256": base.sha256, "ultimo_concurso": base.last, "replicas_mc": replicates,
                   "permutacoes": permutations, "semente": config.SEED,
                   "segundos": round(time.time() - t, 1)}
    write_json(FAMILY1_JSON, res)
    np.savez_compressed(FAMILY1_NULL, **{k: v for k, v in null.items()})
    return res


def run_backtest(random_players: int = 1000, progress=None) -> dict:
    base = store.load()
    t = time.time()
    res = bt.run_backtest(base.draws, base.concursos, random_players=random_players, progress=progress)
    ev = bt.evaluate(res)
    frame = res.frame()
    frame.to_csv(BACKTEST_CSV, index=False, float_format="%.6f")
    # Estado para o hedge prospectivo: log-score acumulado de cada membro até o último concurso
    from megalab.combinatorics import set_log_likelihood_ratio
    members = [m for m in res.models if m not in ("aleatorio", bt.HEDGE)]
    cum = {m: float(set_log_likelihood_ratio(res.proba[m], res.draws - 1).sum()) for m in members}
    w = bt.hedge_weights(np.array([[cum[m] for m in members]]))
    write_json(BACKTEST_STATE, {"base_sha256": base.sha256, "ultimo_concurso": base.last,
                                "loglik_acumulada_nats": cum,
                                "pesos_hedge": dict(zip(members, w.tolist()))})
    ev["meta"] = {"base_sha256": base.sha256, "ultimo_concurso": base.last,
                  "modelos": {m.name: m.description for m in bt.default_models()},
                  "jogadores_aleatorios": random_players, "segundos": round(time.time() - t, 1)}
    write_json(BACKTEST_JSON, ev)
    return ev


def ledger_predict(ledger: lg.Ledger | None = None) -> dict:
    ledger = ledger or lg.Ledger()
    base = store.load()
    weights = None
    if BACKTEST_STATE.exists():
        st = read_json(BACKTEST_STATE)
        if st.get("base_sha256") == base.sha256:
            weights = st["pesos_hedge"]
    models = statistical_models() + ml_models(config.SEED)
    return lg.register_predictions(ledger, base.draws, base.last, base.sha256, models, weights)


def ledger_score(ledger: lg.Ledger | None = None) -> list[dict]:
    ledger = ledger or lg.Ledger()
    df = store.load().df
    results = {int(r.concurso): ([int(getattr(r, f"b{i}")) for i in range(1, 7)], r.data)
               for r in df.itertuples()}
    return lg.score(ledger, results)
