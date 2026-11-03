"""Atualizador incremental pela API oficial da Caixa.

Endpoint: ``https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena/{n}``
(sem ``n`` devolve o último concurso). Campos usados: ``numero``,
``dataApuracao`` (dd/mm/aaaa), ``listaDezenas`` e
``dezenasSorteadasOrdemSorteio`` (ordem de extração).

As respostas brutas são guardadas em ``data/raw_cache/caixa_api_live/`` e os
fatos extraídos vão para o extrato ``data/sources/caixa_api_live.csv``, que
entra na reconciliação como qualquer outra fonte.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd

from megalab.config import RAW_CACHE_DIR
from megalab.data.sources import NORMALIZED_COLUMNS, extract_path, write_extract

API_BASE = "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena"
HEADERS = {"Accept": "application/json", "User-Agent": "megalab/0.1 (pesquisa estatística)"}
SOURCE_ID = "caixa_api_live"


def fetch_json(numero: int | None = None, timeout: int = 20, retries: int = 3) -> dict | None:
    url = f"{API_BASE}/{numero}" if numero else f"{API_BASE}/"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last_err = e
        except Exception as e:  # rede instável
            last_err = e
        time.sleep(2 ** attempt)
    raise ConnectionError(f"Falha ao acessar {url}: {last_err}")


def parse(js: dict) -> dict:
    """Converte uma resposta da API para uma linha normalizada."""
    nums = sorted(int(x) for x in js["listaDezenas"])
    order = js.get("dezenasSorteadasOrdemSorteio") or []
    order = [int(x) for x in order] if len(order) == 6 else None
    if order is not None and sorted(order) != nums:
        order = None  # incoerente: descarta a ordem, mantém o conjunto
    rec = {"concurso": int(js["numero"]),
           "data": datetime.strptime(js["dataApuracao"], "%d/%m/%Y").strftime("%Y-%m-%d")}
    for i, v in enumerate(nums, 1):
        rec[f"b{i}"] = v
    for i in range(1, 7):
        rec[f"o{i}"] = order[i - 1] if order else pd.NA
    return rec


def update(start: int, fetcher=fetch_json, max_new: int = 500) -> list[int]:
    """Busca concursos a partir de ``start`` até a API não ter mais resultados."""
    raw_dir = RAW_CACHE_DIR / SOURCE_ID
    raw_dir.mkdir(parents=True, exist_ok=True)
    latest = fetcher(None)
    if latest is None:
        return []
    last_available = int(latest["numero"])
    rows, got = [], []
    for n in range(start, min(last_available, start + max_new - 1) + 1):
        js = latest if n == last_available else fetcher(n)
        if js is None:
            break
        with open(raw_dir / "respostas.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(js, ensure_ascii=False) + "\n")
        rows.append(parse(js))
        got.append(n)
    if rows:
        path = extract_path(SOURCE_ID)
        old = pd.read_csv(path, dtype={"data": str}) if path.exists() else pd.DataFrame(columns=NORMALIZED_COLUMNS)
        new = pd.DataFrame(rows)
        write_extract(SOURCE_ID, pd.concat([old.drop(columns=["fonte"], errors="ignore"), new]))
    return got
