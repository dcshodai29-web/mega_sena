"""
validacao_temporal.py

Validação que RESPEITA a ordem cronológica dos concursos. Nunca embaralha
os dados. Implementa:

1. Split simples treino (60%) / validação (20%) / teste final (20%),
   só para referência.
2. Validação com janela deslizante (walk-forward): para cada concurso
   t a partir de um ponto de partida mínimo, cada modelo só enxerga os
   concursos 1..t-1, gera um ranking, escolhe os 6 números de maior
   score, e o resultado (quantos acertos) é registrado contra o
   concurso t real. Isso é repetido avançando um concurso por vez até
   o fim da base.

O teste final (últimos 20%) é tratado como conjunto "intocado": não é
usado para decidir qual modelo é "o melhor" -- ele só é reportado depois,
lado a lado com o desempenho em validação, para deixar claro se o
desempenho se mantém ou não fora da amostra usada durante o
desenvolvimento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from estatisticas import matriz_dezenas
from modelos import (
    modelo_aleatorio, modelo_frequencia_acumulada, modelo_frequencia_janela_movel,
    modelo_frequencia_ponderada, modelo_tempo_desde_ultima, modelo_frequencia_pares,
    modelo_beta_binomial, modelo_combinado,
)

N_SORTEADAS = 6


def dividir_treino_validacao_teste(n: int, prop_treino=0.6, prop_val=0.2):
    """Retorna os índices de corte (por posição, não embaralhado)."""
    corte_treino = int(n * prop_treino)
    corte_val = int(n * (prop_treino + prop_val))
    return corte_treino, corte_val


def _rankear(nome_modelo: str, historico: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if nome_modelo == "aleatorio":
        return modelo_aleatorio(historico, rng)
    if nome_modelo == "frequencia_acumulada":
        return modelo_frequencia_acumulada(historico)
    if nome_modelo == "janela_movel":
        return modelo_frequencia_janela_movel(historico, janela=50)
    if nome_modelo == "frequencia_ponderada":
        return modelo_frequencia_ponderada(historico, meia_vida=100)
    if nome_modelo == "tempo_desde_ultima":
        return modelo_tempo_desde_ultima(historico)
    if nome_modelo == "frequencia_pares":
        return modelo_frequencia_pares(historico)
    if nome_modelo == "beta_binomial":
        media, _ = modelo_beta_binomial(historico)
        return media
    if nome_modelo == "combinado":
        return modelo_combinado(historico)
    raise ValueError(f"Modelo desconhecido: {nome_modelo}")


def walk_forward(
    df: pd.DataFrame,
    modelos: list[str],
    inicio_minimo: int = 60,
    passo: int = 1,
    semente: int = 42,
) -> pd.DataFrame:
    """Para cada concurso a partir de `inicio_minimo`, gera o palpite de
    cada modelo (top 6 do ranking) usando SOMENTE concursos anteriores, e
    registra os acertos contra o resultado real.

    `inicio_minimo` existe porque modelos como janela móvel e regressão
    precisam de um mínimo de histórico para produzir algo que não seja
    ruído puro -- e isso é decidido ANTES de olhar o desempenho, não
    ajustado a posteriori.
    """
    m = matriz_dezenas(df)
    n = len(m)
    rng = np.random.default_rng(semente)

    registros = []
    for t in range(inicio_minimo, n, passo):
        historico = m[:t]
        resultado_real = set(m[t].tolist())
        for nome in modelos:
            scores = _rankear(nome, historico, rng)
            palpite = set((np.argsort(scores)[-N_SORTEADAS:] + 1).tolist())
            acertos = len(palpite & resultado_real)
            registros.append({
                "concurso_index": t, "concurso": int(df.iloc[t]["concurso"]),
                "modelo": nome, "acertos": acertos,
            })
    return pd.DataFrame(registros)


def resumir_desempenho(resultados_wf: pd.DataFrame, corte_treino: int, corte_val: int) -> pd.DataFrame:
    """Resume acertos médios/medianos por modelo, separando os períodos de
    validação e teste final (baseado na posição do concurso, não em
    reamostragem)."""
    def rotulo_periodo(idx):
        if idx < corte_treino:
            return "treino"
        if idx < corte_val:
            return "validacao"
        return "teste_final"

    df = resultados_wf.copy()
    df["periodo"] = df["concurso_index"].apply(rotulo_periodo)

    resumo = (
        df.groupby(["modelo", "periodo"])["acertos"]
        .agg(media="mean", mediana="median", desvio_padrao="std", n="count",
             pct_4_ou_mais=lambda s: (s >= 4).mean() * 100,
             pct_5_ou_mais=lambda s: (s >= 5).mean() * 100,
             pct_6=lambda s: (s == 6).mean() * 100)
        .reset_index()
    )
    return resumo


def comparar_modelo_vs_aleatorio(resultados_wf: pd.DataFrame, modelo: str, periodo_mask=None) -> dict:
    """Teste de Wilcoxon pareado (mesmos concursos, dois modelos) comparando
    os acertos do modelo contra os do modelo aleatório, concurso a
    concurso. H0: não há diferença sistemática nos acertos entre os dois.
    """
    from scipy import stats as scipy_stats

    piv = resultados_wf.pivot(index="concurso_index", columns="modelo", values="acertos")
    if periodo_mask is not None:
        piv = piv[periodo_mask(piv.index)]
    if modelo not in piv.columns or "aleatorio" not in piv.columns:
        return {"erro": "modelo ou baseline aleatório ausente nos resultados"}

    a = piv[modelo].dropna()
    b = piv["aleatorio"].reindex(a.index).dropna()
    a = a.reindex(b.index)

    diffs = a - b
    if (diffs == 0).all():
        return {
            "modelo": modelo, "n_pares": len(diffs), "media_diferenca": 0.0,
            "estatistica": np.nan, "valor_p": 1.0,
            "interpretacao": "Nenhuma diferença observada entre o modelo e a linha de base aleatória.",
        }
    estat, p = scipy_stats.wilcoxon(a, b)
    return {
        "modelo": modelo,
        "n_pares": int(len(diffs)),
        "media_acertos_modelo": float(a.mean()),
        "media_acertos_aleatorio": float(b.mean()),
        "media_diferenca": float(diffs.mean()),
        "estatistica_wilcoxon": float(estat),
        "valor_p": float(p),
        "interpretacao": (
            "Valor-p baixo indicaria diferença sistemática nos acertos entre "
            "o modelo e o sorteio aleatório de referência, nesse período. "
            "Isso precisa se repetir em múltiplos períodos e no teste final "
            "intocado para ser levado a sério -- um resultado isolado, "
            "especialmente com eventos raros como 5 ou 6 acertos, pode ser "
            "puro acaso."
        ),
    }
