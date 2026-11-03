"""Construção e carga da base canônica."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from megalab.config import CANONICAL_CSV, CANONICAL_DIR, PRIZES_CSV, SOURCES_DIR
from megalab.data import integrity, reconcile, sources

B_COLS = [f"b{i}" for i in range(1, 7)]
O_COLS = [f"o{i}" for i in range(1, 7)]
ACCEPTED_STATUS = ("A", "B", "X")  # X só entra se resolvido por maioria estrita


@dataclass(frozen=True)
class Base:
    """Visão imutável da base usada nas análises."""

    df: pd.DataFrame
    sha256: str

    @property
    def draws(self) -> np.ndarray:
        """Matriz (n, 6) de dezenas ordenadas, em ordem cronológica."""
        return self.df[B_COLS].to_numpy(dtype=np.int64)

    @property
    def concursos(self) -> np.ndarray:
        return self.df["concurso"].to_numpy(dtype=np.int64)

    @property
    def last(self) -> int:
        return int(self.df["concurso"].max())

    def between(self, first: int, last: int | None) -> "Base":
        last = self.last if last is None else last
        sub = self.df[(self.df["concurso"] >= first) & (self.df["concurso"] <= last)]
        return Base(sub.reset_index(drop=True), self.sha256)


def build(local_copies: dict[str, Path] | None = None, today=None) -> dict:
    """Pipeline completo: fontes → extratos → reconciliação → integridade."""
    local_copies = local_copies or {}
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    extracts = {}
    for sid, src in sources.SOURCES.items():
        raw = sources.fetch(src, local_copies.get(sid))
        df = sources.extract(src, raw)
        df.to_csv(sources.extract_path(sid), index=False)
        extracts[sid] = df
        if sid == "caixa_planilha_oficial":
            sources.parse_caixa_xlsx_prizes(raw).to_csv(PRIZES_CSV, index=False)
    sources.write_manifest(extracts)
    return build_from_extracts(today=today)


def build_from_extracts(today=None) -> dict:
    extracts = sources.load_extracts()
    extracts = {k: v.drop(columns=["fonte"]) for k, v in extracts.items()}
    canon, conf = reconcile.reconcile(extracts)
    canon.to_csv(CANONICAL_CSV, index=False)
    conf.to_csv(CANONICAL_DIR / "conflitos.csv", index=False)
    report = integrity.check(canon, today=today)
    report["sha256_base"] = sources.sha256_file(CANONICAL_CSV)
    (CANONICAL_DIR / "integridade.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def load(path: Path = CANONICAL_CSV, include_single_source: bool = False) -> Base:
    """Carrega a base canônica.

    Por padrão inclui apenas concursos verificados por ≥ 2 fontes (status A/B)
    ou conflitos resolvidos por maioria estrita. ``include_single_source``
    inclui também concursos de fonte única (status C).
    """
    df = pd.read_csv(path, dtype={"data": str}, keep_default_na=False, na_values=[""])
    df["data"] = df["data"].fillna("")
    for col in O_COLS:
        df[col] = df[col].astype("Int64")
    ok = df["resolvido"].astype(bool) & df["status"].isin(
        ACCEPTED_STATUS + (("C",) if include_single_source else ()))
    df = df[ok].sort_values("concurso").reset_index(drop=True)
    # Garantia: a série usada é contígua desde o concurso 1.
    expected = np.arange(1, len(df) + 1)
    if not np.array_equal(df["concurso"].to_numpy(), expected):
        gaps = sorted(set(expected) - set(df["concurso"]))[:10]
        raise ValueError(f"Base canônica com lacunas após filtragem: {gaps}")
    return Base(df, sources.sha256_file(path))


def load_prizes(path: Path = PRIZES_CSV) -> pd.DataFrame:
    return pd.read_csv(path)
