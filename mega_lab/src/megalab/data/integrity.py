"""Verificações de integridade da base canônica."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from megalab.config import N_NUMBERS

B_COLS = [f"b{i}" for i in range(1, 7)]
O_COLS = [f"o{i}" for i in range(1, 7)]
WEEKDAYS = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]


def schedule_regime(dates: pd.Series, half_window_days: int = 42,
                    min_span: int = 20) -> pd.Series:
    """Classifica o regime de calendário pela taxa de sorteios por semana.

    Inferido apenas dos dados (sem datas históricas externas): para cada
    sorteio conta os sorteios numa janela centrada de ±``half_window_days``
    (truncada nas bordas da série) e divide pelo número de semanas cobertas.
    Taxa < 1,5 → ``1x/semana``; < 2,5 → ``2x/semana``; senão ``3x/semana``.
    Trechos com menos de ``min_span`` sorteios (pausas de fim de ano, feriados)
    são absorvidos pelo regime anterior.
    """
    t = pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype(np.int64)
    lo = np.maximum(t - half_window_days, t[0])
    hi = np.minimum(t + half_window_days, t[-1])
    cnt = np.searchsorted(t, hi, side="right") - np.searchsorted(t, lo, side="left")
    weeks = np.maximum(hi - lo, 7) / 7.0
    rate = cnt / weeks
    reg = np.select([rate < 1.5, rate < 2.5], ["1x/semana", "2x/semana"], "3x/semana")
    # Absorve trechos curtos
    out = reg.copy()
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and reg[j + 1] == reg[i]:
            j += 1
        if (j - i + 1) < min_span and i > 0:
            out[i:j + 1] = out[i - 1]
        i = j + 1
    return pd.Series(out, index=dates.index)


def is_year_end_special(dates: pd.Series) -> pd.Series:
    """Proxy para a Mega da Virada: sorteio em 31/12 ou 01/01."""
    d = pd.to_datetime(dates.replace("", pd.NA), errors="coerce")
    return (((d.dt.month == 12) & (d.dt.day == 31)) | ((d.dt.month == 1) & (d.dt.day == 1))).fillna(False)


def check(canon: pd.DataFrame, today: date | None = None) -> dict:
    today = today or date.today()
    issues: list[dict] = []
    c = canon["concurso"].to_numpy()

    # 1. Sequência completa
    expected = set(range(1, int(c.max()) + 1))
    missing = sorted(expected - set(c.tolist()))
    dup = canon["concurso"][canon["concurso"].duplicated()].tolist()
    for m in missing:
        issues.append({"concurso": m, "tipo": "concurso_ausente"})
    for d in dup:
        issues.append({"concurso": d, "tipo": "concurso_duplicado"})

    # 2. Dezenas válidas
    B = canon[B_COLS].to_numpy(dtype=float)
    bad_range = ~np.all((B >= 1) & (B <= N_NUMBERS), axis=1)
    not_sorted_distinct = ~np.all(np.diff(B, axis=1) > 0, axis=1)
    for cc in canon.loc[bad_range | not_sorted_distinct, "concurso"]:
        issues.append({"concurso": int(cc), "tipo": "dezenas_invalidas"})

    # 3. Ordem de extração coerente com o conjunto
    has_order = canon[O_COLS].notna().all(axis=1)
    if has_order.any():
        O = np.sort(canon.loc[has_order, O_COLS].to_numpy(dtype=int), axis=1)
        mismatch = ~np.all(O == canon.loc[has_order, B_COLS].to_numpy(dtype=int), axis=1)
        for cc in canon.loc[has_order, "concurso"][mismatch]:
            issues.append({"concurso": int(cc), "tipo": "ordem_incoerente"})

    # 4. Datas
    dts = pd.to_datetime(canon["data"].replace("", pd.NA), errors="coerce")
    no_date = canon.loc[dts.isna(), "concurso"].tolist()
    for cc in no_date:
        issues.append({"concurso": int(cc), "tipo": "data_ausente"})
    valid = dts.notna()
    s = dts[valid]
    non_increasing = canon.loc[valid, "concurso"][s.diff().dt.days.fillna(1) <= 0].tolist()
    for cc in non_increasing:
        issues.append({"concurso": int(cc), "tipo": "data_nao_crescente"})
    future = canon.loc[valid & (dts.dt.date > today), "concurso"].tolist()
    for cc in future:
        issues.append({"concurso": int(cc), "tipo": "data_futura"})

    # 5. Status de verificação
    status_counts = canon["status"].value_counts().to_dict()
    unresolved = canon.loc[~canon["resolvido"], "concurso"].tolist()
    for cc in unresolved:
        issues.append({"concurso": int(cc), "tipo": "conflito_nao_resolvido"})

    weekday = dts[valid].dt.weekday.map(lambda i: WEEKDAYS[i]).value_counts().to_dict()
    gaps = s.diff().dt.days.dropna()
    regimes = schedule_regime(canon.loc[valid, "data"])
    regime_spans = []
    prev = None
    for cc, r in zip(canon.loc[valid, "concurso"], regimes):
        if r != prev:
            regime_spans.append({"regime": r, "inicio": int(cc), "fim": int(cc)})
            prev = r
        else:
            regime_spans[-1]["fim"] = int(cc)

    return {
        "concursos": int(len(canon)),
        "primeiro": int(c.min()), "ultimo": int(c.max()),
        "primeira_data": str(s.min().date()) if len(s) else None,
        "ultima_data": str(s.max().date()) if len(s) else None,
        "ausentes": missing, "duplicados": dup,
        "status": {k: int(v) for k, v in status_counts.items()},
        "status_data": {k: int(v) for k, v in canon["status_data"].value_counts().items()},
        "status_ordem": {k: int(v) for k, v in canon["status_ordem"].value_counts().items()},
        "com_ordem": int(has_order.sum()),
        "dia_da_semana": {k: int(v) for k, v in weekday.items()},
        "intervalo_dias": {"min": int(gaps.min()), "mediana": float(gaps.median()),
                           "max": int(gaps.max())} if len(gaps) else {},
        "regimes_calendario": regime_spans,
        "sorteios_fim_de_ano": canon.loc[is_year_end_special(canon["data"]),
                                         "concurso"].astype(int).tolist(),
        "problemas": issues,
        "n_problemas": len(issues),
    }
