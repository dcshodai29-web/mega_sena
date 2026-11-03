"""Atualizador da API da Caixa testado com respostas SINTÉTICAS no formato da API.

Os números abaixo são fictícios (concursos 9998/9999) e servem apenas para
testar o parser; não são resultados reais.
"""

import pandas as pd
import pytest

from megalab.data import caixa_api


def fake_response(n, data, nums, order):
    return {"numero": n, "dataApuracao": data, "listaDezenas": [f"{x:02d}" for x in nums],
            "dezenasSorteadasOrdemSorteio": [f"{x:02d}" for x in order], "acumulado": True}


def test_parse_keeps_order_and_iso_date():
    rec = caixa_api.parse(fake_response(9999, "05/11/2026", [3, 14, 15, 26, 35, 59],
                                        [35, 3, 59, 14, 26, 15]))
    assert rec["concurso"] == 9999 and rec["data"] == "2026-11-05"
    assert [rec[f"b{i}"] for i in range(1, 7)] == [3, 14, 15, 26, 35, 59]
    assert [rec[f"o{i}"] for i in range(1, 7)] == [35, 3, 59, 14, 26, 15]


def test_parse_drops_incoherent_order():
    rec = caixa_api.parse(fake_response(9999, "05/11/2026", [3, 14, 15, 26, 35, 59],
                                        [35, 3, 59, 14, 26, 16]))
    assert pd.isna(rec["o1"])


def test_update_appends_extract(tmp_path, monkeypatch):
    monkeypatch.setattr(caixa_api, "RAW_CACHE_DIR", tmp_path / "raw")
    monkeypatch.setattr(caixa_api, "extract_path", lambda sid: tmp_path / f"{sid}.csv")
    import megalab.data.sources as src
    monkeypatch.setattr(src, "extract_path", lambda sid: tmp_path / f"{sid}.csv")
    responses = {9998: fake_response(9998, "03/11/2026", [1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]),
                 9999: fake_response(9999, "05/11/2026", [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12])}

    def fetcher(n):
        return responses[9999] if n is None else responses.get(n)

    got = caixa_api.update(9998, fetcher=fetcher)
    assert got == [9998, 9999]
    df = pd.read_csv(tmp_path / "caixa_api_live.csv")
    assert list(df["concurso"]) == [9998, 9999] and (df["fonte"] == "caixa_api_live").all()
    assert (tmp_path / "raw" / "caixa_api_live" / "respostas.jsonl").exists()


def test_fetch_json_reports_connection_errors(monkeypatch):
    def boom(*a, **k):
        raise OSError("rede bloqueada")
    monkeypatch.setattr(caixa_api.urllib.request, "urlopen", boom)
    monkeypatch.setattr(caixa_api.time, "sleep", lambda s: None)
    with pytest.raises(ConnectionError):
        caixa_api.fetch_json(1, retries=2)
