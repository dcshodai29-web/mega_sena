"""
simulacao.py

Simulação Monte Carlo: gera séries inteiramente aleatórias (mesmo
tamanho da série real) para servir de referência ("e se o sorteio
fosse mesmo 100% ao acaso, como series desse tamanho se comportariam?").

A pergunta que este módulo responde é: "os padrões observados nos
dados reais são incomuns frente a séries genuinamente aleatórias do
mesmo tamanho?" -- não "o sorteio real é previsível".

Implementação vetorizada em lotes (batches) para viabilizar 100.000
repetições sem estourar memória: cada lote gera um array
(lote, n_concursos, 60) de números aleatórios e usa argsort para
escolher as 6 menores posições por linha -- equivalente a sortear 6
de 60 sem reposição, para cada concurso simulado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_DEZENAS = 60
N_SORTEADAS = 6


def _simular_lote(n_concursos: int, tamanho_lote: int, rng: np.random.Generator) -> np.ndarray:
    """Retorna array (tamanho_lote, n_concursos, 6) com dezenas de 1 a 60."""
    aleatorio = rng.random((tamanho_lote, n_concursos, N_DEZENAS))
    indices = np.argpartition(aleatorio, N_SORTEADAS, axis=2)[:, :, :N_SORTEADAS]
    dezenas = indices + 1  # 1 a 60
    return np.sort(dezenas, axis=2)


def rodar_monte_carlo(
    n_concursos: int,
    n_repeticoes: int = 100_000,
    tamanho_lote: int = 500,
    semente: int = 42,
) -> dict:
    """Roda a simulação Monte Carlo e devolve estatísticas agregadas
    (não guarda todas as séries brutas, para não estourar memória).

    Parameters
    ----------
    n_concursos: tamanho de cada série simulada (deve bater com os dados reais)
    n_repeticoes: quantas séries simular
    tamanho_lote: quantas séries processar de cada vez (controla uso de memória)
    semente: semente do gerador aleatório, para reprodutibilidade
    """
    rng = np.random.default_rng(semente)

    freq_max_por_serie = []
    freq_min_por_serie = []
    soma_media_por_serie = []
    soma_todas = []
    qtd_pares_todas = []
    repeticao_media_por_serie = []
    maior_atraso_medio_por_serie = []
    pct_consecutivo_por_serie = []

    feitas = 0
    while feitas < n_repeticoes:
        lote = min(tamanho_lote, n_repeticoes - feitas)
        series = _simular_lote(n_concursos, lote, rng)  # (lote, n_concursos, 6)

        # frequência de cada dezena, por série simulada
        for s in range(lote):
            m = series[s]
            freq = np.bincount(m.flatten(), minlength=N_DEZENAS + 1)[1:]
            freq_max_por_serie.append(int(freq.max()))
            freq_min_por_serie.append(int(freq.min()))

            somas = m.sum(axis=1)
            soma_media_por_serie.append(float(somas.mean()))
            soma_todas.append(somas)

            qtd_pares = (m % 2 == 0).sum(axis=1)
            qtd_pares_todas.append(qtd_pares)

            reps = [len(set(m[i]) & set(m[i - 1])) for i in range(1, n_concursos)]
            repeticao_media_por_serie.append(float(np.mean(reps)) if reps else np.nan)

            # maior atraso médio (entre as 60 dezenas, nesta série)
            atrasos = []
            for dez in range(1, N_DEZENAS + 1):
                pos = np.where((m == dez).any(axis=1))[0]
                if len(pos) < 2:
                    atrasos.append(n_concursos)
                else:
                    atrasos.append(int(np.diff(pos).max()))
            maior_atraso_medio_por_serie.append(float(np.mean(atrasos)))

            consec = 0
            for linha in m:
                ln = np.sort(linha)
                if np.any(np.diff(ln) == 1):
                    consec += 1
            pct_consecutivo_por_serie.append(100 * consec / n_concursos)

        feitas += lote

    soma_todas = np.concatenate(soma_todas)
    qtd_pares_todas = np.concatenate(qtd_pares_todas)

    return {
        "n_repeticoes": n_repeticoes,
        "n_concursos_por_serie": n_concursos,
        "freq_max_por_serie": np.array(freq_max_por_serie),
        "freq_min_por_serie": np.array(freq_min_por_serie),
        "soma_media_por_serie": np.array(soma_media_por_serie),
        "soma_todas_simulacoes": soma_todas,
        "qtd_pares_todas_simulacoes": qtd_pares_todas,
        "repeticao_media_por_serie": np.array(repeticao_media_por_serie),
        "maior_atraso_medio_por_serie": np.array(maior_atraso_medio_por_serie),
        "pct_consecutivo_por_serie": np.array(pct_consecutivo_por_serie),
    }


def percentil_do_valor_real(distribuicao_simulada: np.ndarray, valor_real: float) -> float:
    """Em que percentil da distribuição simulada o valor real observado cai.
    Ex: 0.5 = exatamente na mediana das simulações; 0.99 = maior que 99% delas."""
    return float((distribuicao_simulada < valor_real).mean())


def comparar_real_vs_simulacao(estatisticas_reais: dict, resultado_mc: dict) -> pd.DataFrame:
    """Monta uma tabela comparando cada métrica real com a distribuição
    simulada correspondente, indicando o percentil onde o valor real cai."""
    comparacoes = [
        ("Frequência máxima de uma dezena", estatisticas_reais.get("freq_max"),
         resultado_mc["freq_max_por_serie"]),
        ("Frequência mínima de uma dezena", estatisticas_reais.get("freq_min"),
         resultado_mc["freq_min_por_serie"]),
        ("Soma média dos concursos", estatisticas_reais.get("soma_media"),
         resultado_mc["soma_media_por_serie"]),
        ("Repetição média vs. concurso anterior", estatisticas_reais.get("repeticao_media"),
         resultado_mc["repeticao_media_por_serie"]),
        ("Maior atraso médio entre dezenas", estatisticas_reais.get("maior_atraso_medio"),
         resultado_mc["maior_atraso_medio_por_serie"]),
        ("% de concursos com consecutivos", estatisticas_reais.get("pct_consecutivo"),
         resultado_mc["pct_consecutivo_por_serie"]),
    ]
    linhas = []
    for nome, valor_real, dist in comparacoes:
        if valor_real is None:
            continue
        pct = percentil_do_valor_real(dist, valor_real)
        linhas.append({
            "metrica": nome, "valor_real": valor_real,
            "media_simulada": float(np.mean(dist)), "desvio_padrao_simulado": float(np.std(dist)),
            "percentil_do_valor_real": pct,
            "incomum_5pct": bool(pct < 0.025 or pct > 0.975),
        })
    return pd.DataFrame(linhas)
