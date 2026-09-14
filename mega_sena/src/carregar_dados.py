"""
carregar_dados.py

Responsável por ler o arquivo CSV de resultados da Mega-Sena e devolver
um DataFrame pandas com tipos e nomes de coluna padronizados.

Este módulo NÃO valida a integridade dos dados (isso é feito em
validar_dados.py). Ele apenas lê e tipa.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

COLUNAS_DEZENAS = ["n1", "n2", "n3", "n4", "n5", "n6"]
COLUNAS_OBRIGATORIAS = ["concurso", "data"] + COLUNAS_DEZENAS


def carregar_csv(caminho: str | Path) -> pd.DataFrame:
    """Lê o CSV de resultados e retorna um DataFrame bruto (sem validação).

    Parameters
    ----------
    caminho: caminho para o arquivo CSV (ex: dados/megasena.csv)

    Returns
    -------
    DataFrame com as colunas obrigatórias, na ordem em que aparecem no
    arquivo original (não reordenado, não deduplicado, não validado).

    Raises
    ------
    FileNotFoundError: se o arquivo não existir.
    ValueError: se alguma coluna obrigatória estiver ausente.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    df = pd.read_csv(caminho)

    faltantes = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Colunas obrigatórias ausentes no CSV: {faltantes}. "
            f"Colunas encontradas: {list(df.columns)}"
        )

    return df


def tipar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos: concurso -> int, data -> datetime, dezenas -> Int64 nullable.

    Não lança erro se a conversão falhar linha a linha; em vez disso,
    valores inválidos viram NaN/NaT para que validar_dados.py possa
    reportá-los explicitamente (falha silenciosa aqui seria desonesta
    com o usuário).
    """
    df = df.copy()
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce").astype("Int64")
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    for c in COLUNAS_DEZENAS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def carregar(caminho: str | Path) -> pd.DataFrame:
    """Atalho: carregar_csv + tipar_dataframe."""
    return tipar_dataframe(carregar_csv(caminho))
