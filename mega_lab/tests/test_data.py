import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from megalab.config import CANONICAL_CSV, SOURCES_DIR
from megalab.data import integrity, reconcile, sources, store


def _ext(rows, order=False):
    recs = []
    for c, d, nums in rows:
        r = {"concurso": c, "data": d}
        for i, v in enumerate(sorted(nums), 1):
            r[f"b{i}"] = v
        for i in range(1, 7):
            r[f"o{i}"] = nums[i - 1] if order else pd.NA
        recs.append(r)
    df = pd.DataFrame(recs)
    for col in [f"o{i}" for i in range(1, 7)]:
        df[col] = df[col].astype("Int64")
    return df


def test_reconcile_status_levels():
    a = _ext([(1, "1996-03-11", [1, 2, 3, 4, 5, 6]), (2, "1996-03-18", [7, 8, 9, 10, 11, 12])])
    b = _ext([(1, "1996-03-11", [1, 2, 3, 4, 5, 6]), (2, "1996-03-18", [7, 8, 9, 10, 11, 12])])
    c = _ext([(1, "1996-03-11", [1, 2, 3, 4, 5, 6]), (2, "1996-03-18", [7, 8, 9, 10, 11, 13])])
    canon, conf = reconcile.reconcile({"caixa_planilha_oficial": a, "asloterias_csv": b,
                                       "megaloterias_rpkg": c})
    r1, r2 = canon.iloc[0], canon.iloc[1]
    assert r1["status"] == "A" and r1["n_fontes"] == 3
    assert r2["status"] == "X" and bool(r2["resolvido"])  # 2 x 1 → maioria estrita
    assert [r2[f"b{i}"] for i in range(1, 7)] == [7, 8, 9, 10, 11, 12]
    assert set(conf["concurso"]) == {2}


def test_reconcile_tie_is_unresolved():
    a = _ext([(1, "", [1, 2, 3, 4, 5, 6])])
    b = _ext([(1, "", [1, 2, 3, 4, 5, 7])])
    canon, _ = reconcile.reconcile({"caixa_planilha_oficial": a, "asloterias_csv": b})
    assert canon.iloc[0]["status"] == "X" and not bool(canon.iloc[0]["resolvido"])


def test_reconcile_order_agreement_and_conflict():
    a = _ext([(1, "", [6, 5, 4, 3, 2, 1])], order=True)
    b = _ext([(1, "", [6, 5, 4, 3, 2, 1])], order=True)
    canon, _ = reconcile.reconcile({"asloterias_ordem": a, "guilhermeasn_ordem": b})
    assert canon.iloc[0]["status_ordem"] == "ok"
    assert [canon.iloc[0][f"o{i}"] for i in range(1, 7)] == [6, 5, 4, 3, 2, 1]
    c = _ext([(1, "", [5, 6, 4, 3, 2, 1])], order=True)
    canon, _ = reconcile.reconcile({"asloterias_ordem": a, "guilhermeasn_ordem": c})
    assert canon.iloc[0]["status_ordem"] == "conflito"
    assert pd.isna(canon.iloc[0]["o1"])


def test_integrity_detects_problems():
    a = _ext([(1, "1996-03-11", [1, 2, 3, 4, 5, 6]), (3, "1996-03-10", [1, 2, 3, 4, 5, 6]),
              (4, "2099-01-01", [1, 2, 3, 4, 5, 6])])
    canon, _ = reconcile.reconcile({"caixa_planilha_oficial": a})
    rep = integrity.check(canon, today=date(2026, 11, 3))
    kinds = {(p["concurso"], p["tipo"]) for p in rep["problemas"]}
    assert (2, "concurso_ausente") in kinds
    assert (3, "data_nao_crescente") in kinds
    assert (4, "data_futura") in kinds


def test_schedule_regime_detects_rate_changes():
    d1 = pd.date_range("2000-01-01", periods=60, freq="7D")
    d2 = pd.date_range(d1[-1] + pd.Timedelta(days=7), periods=120, freq="3D")
    dates = pd.Series([d.strftime("%Y-%m-%d") for d in d1.append(d2)])
    reg = integrity.schedule_regime(dates)
    assert reg.iloc[10] == "1x/semana" and reg.iloc[-10] == "2x/semana"


# --- Base real versionada -----------------------------------------------------------

def test_canonical_base_is_complete_and_valid():
    b = store.load()
    d = b.draws
    assert len(d) >= 3080
    assert np.array_equal(b.concursos, np.arange(1, len(d) + 1))
    assert d.min() >= 1 and d.max() <= 60
    assert np.all(np.diff(d, axis=1) > 0)
    assert b.df.loc[0, "data"] == "1996-03-11"
    assert list(d[0]) == [4, 5, 30, 33, 41, 52]


def test_canonical_order_matches_sets():
    df = store.load().df
    o = df[[f"o{i}" for i in range(1, 7)]].dropna()
    assert len(o) >= 3066
    sorted_o = np.sort(o.to_numpy(dtype=int), axis=1)
    assert np.array_equal(sorted_o, df.loc[o.index, [f"b{i}" for i in range(1, 7)]].to_numpy())


def test_manifest_hashes_match_extracts():
    manifest = json.loads((SOURCES_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    for f in manifest["fontes"]:
        assert sources.sha256_file(sources.extract_path(f["id"])) == f["sha256_extrato"], f["id"]


def test_reconciliation_is_reproducible(tmp_path, monkeypatch):
    """Reconstruir a base a partir dos extratos gera exatamente o CSV versionado."""
    extracts = {k: v.drop(columns=["fonte"]) for k, v in sources.load_extracts().items()}
    canon, _ = reconcile.reconcile(extracts)
    out = tmp_path / "c.csv"
    canon.to_csv(out, index=False)
    assert sources.sha256_file(out) == sources.sha256_file(CANONICAL_CSV)


@pytest.mark.parametrize("sid", list(sources.SOURCES))
def test_each_extract_agrees_with_canonical(sid):
    ext = sources.load_extracts()[sid]
    can = store.load().df.set_index("concurso")
    e = ext.set_index("concurso")
    common = e.index.intersection(can.index)
    cols = [f"b{i}" for i in range(1, 7)]
    assert (e.loc[common, cols].to_numpy() == can.loc[common, cols].to_numpy()).all()
