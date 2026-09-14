"""
estatisticas.py

Análise descritiva pura: contagens, médias, distribuições. Nada aqui
testa hipóteses ou afirma significância estatística -- isso é
responsabilidade de testes_aleatoriedade.py. Descrever os dados não é
o mesmo que provar (ou refutar) aleatoriedade.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pandas as pd

COLUNAS_DEZENAS = ["n1", "n2", "n3", "n4", "n5", "n6"]
N_DEZENAS = 60
N_SORTEADAS = 6


def matriz_dezenas(df: pd.DataFrame) -> np.ndarray:
    """Retorna array (n_concursos, 6) com as dezenas de cada concurso, ordenadas."""
    m = df[COLUNAS_DEZENAS].to_numpy(dtype=int)
    return np.sort(m, axis=1)


def frequencia_dezenas(df: pd.DataFrame) -> pd.DataFrame:
    """Frequência absoluta e relativa de cada dezena (1 a 60)."""
    m = matriz_dezenas(df)
    contagem = Counter(m.flatten().tolist())
    n_concursos = len(df)
    linhas = []
    for dez in range(1, N_DEZENAS + 1):
        abs_ = contagem.get(dez, 0)
        linhas.append({
            "dezena": dez,
            "frequencia_absoluta": abs_,
            "frequencia_relativa": abs_ / n_concursos if n_concursos else np.nan,
            "esperado_sob_uniformidade": n_concursos * N_SORTEADAS / N_DEZENAS,
        })
    return pd.DataFrame(linhas)


def ranking_mais_frequentes(freq_df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    return freq_df.sort_values("frequencia_absoluta", ascending=False).head(top).reset_index(drop=True)


def ranking_menos_frequentes(freq_df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    return freq_df.sort_values("frequencia_absoluta", ascending=True).head(top).reset_index(drop=True)


def atrasos(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada dezena: atraso atual (concursos desde a última vez que saiu),
    maior atraso histórico, média e desvio-padrão do intervalo entre ocorrências.
    """
    m = matriz_dezenas(df)
    n = len(df)
    linhas = []
    for dez in range(1, N_DEZENAS + 1):
        posicoes = np.where((m == dez).any(axis=1))[0]
        if len(posicoes) == 0:
            linhas.append({
                "dezena": dez, "atraso_atual": n, "maior_atraso_historico": n,
                "media_intervalo": np.nan, "desvio_padrao_intervalo": np.nan,
                "vezes_sorteada": 0,
            })
            continue
        atraso_atual = (n - 1) - posicoes[-1]
        intervalos = np.diff(posicoes)
        maior_atraso = int(intervalos.max()) if len(intervalos) else n
        maior_atraso = max(maior_atraso, int(posicoes[0]))  # intervalo antes da 1a ocorrencia
        linhas.append({
            "dezena": dez,
            "atraso_atual": int(atraso_atual),
            "maior_atraso_historico": int(maior_atraso),
            "media_intervalo": float(np.mean(intervalos)) if len(intervalos) else np.nan,
            "desvio_padrao_intervalo": float(np.std(intervalos, ddof=1)) if len(intervalos) > 1 else np.nan,
            "vezes_sorteada": int(len(posicoes)),
        })
    return pd.DataFrame(linhas)


def paridade(df: pd.DataFrame) -> pd.DataFrame:
    """Distribuição da quantidade de números pares por concurso (0 a 6)."""
    m = matriz_dezenas(df)
    n_pares = (m % 2 == 0).sum(axis=1)
    dist = pd.Series(n_pares).value_counts().sort_index()
    out = pd.DataFrame({"qtd_pares": dist.index, "qtd_concursos": dist.values})
    out["qtd_impares"] = 6 - out["qtd_pares"]
    out["proporcao"] = out["qtd_concursos"] / len(df)
    return out.reset_index(drop=True)


def soma_dezenas(df: pd.DataFrame) -> pd.Series:
    m = matriz_dezenas(df)
    return pd.Series(m.sum(axis=1), name="soma")


def estatisticas_soma(df: pd.DataFrame) -> dict:
    s = soma_dezenas(df)
    return {
        "media": float(s.mean()), "mediana": float(s.median()),
        "minimo": int(s.min()), "maximo": int(s.max()),
        "desvio_padrao": float(s.std(ddof=1)),
    }


def distribuicao_por_faixas(df: pd.DataFrame, tamanho_faixa: int = 10) -> pd.DataFrame:
    """Quantas dezenas sorteadas caem em cada faixa de tamanho `tamanho_faixa`."""
    m = matriz_dezenas(df).flatten()
    faixas = ((m - 1) // tamanho_faixa) * tamanho_faixa + 1
    faixas_label = [f"{f}-{f + tamanho_faixa - 1}" for f in faixas]
    dist = pd.Series(faixas_label).value_counts()
    ordem = [f"{i * tamanho_faixa + 1}-{(i + 1) * tamanho_faixa}" for i in range(N_DEZENAS // tamanho_faixa)]
    dist = dist.reindex(ordem, fill_value=0)
    return pd.DataFrame({"faixa": dist.index, "quantidade": dist.values})


def repeticao_entre_concursos(df: pd.DataFrame, janela: int = 1) -> pd.Series:
    """Quantidade de dezenas do concurso atual que já apareceram em algum dos
    `janela` concursos anteriores (janela=1 -> só o concurso imediatamente anterior).
    """
    m = matriz_dezenas(df)
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(janela, n):
        anteriores = set(m[max(0, i - janela):i].flatten().tolist())
        atuais = set(m[i].tolist())
        out[i] = len(atuais & anteriores)
    return pd.Series(out, name=f"repetidas_ultimos_{janela}")


def sequencias_consecutivas(df: pd.DataFrame) -> pd.DataFrame:
    """Frequência de cada par de dezenas consecutivas (ex: 10-11) que apareceu
    junto no mesmo concurso."""
    m = matriz_dezenas(df)
    contagem = Counter()
    for linha in m:
        for a, b in zip(linha[:-1], linha[1:]):
            if b == a + 1:
                contagem[(int(a), int(b))] += 1
    out = pd.DataFrame(
        [{"par": f"{a}-{b}", "frequencia": c} for (a, b), c in contagem.items()]
    ).sort_values("frequencia", ascending=False).reset_index(drop=True)
    return out


def frequencia_finais(df: pd.DataFrame) -> pd.DataFrame:
    """Frequência do dígito final (unidade) das dezenas sorteadas: 0 a 9."""
    m = matriz_dezenas(df).flatten()
    finais = m % 10
    dist = pd.Series(finais).value_counts().sort_index()
    return pd.DataFrame({"final": dist.index, "frequencia": dist.values})


def distribuicao_por_quadrante(df: pd.DataFrame) -> pd.DataFrame:
    """Divide 1-60 em 4 quadrantes de 15 e conta quantas dezenas sorteadas
    caem em cada um."""
    m = matriz_dezenas(df).flatten()
    def quad(n):
        if n <= 15: return "1-15"
        if n <= 30: return "16-30"
        if n <= 45: return "31-45"
        return "46-60"
    dist = pd.Series([quad(n) for n in m]).value_counts()
    ordem = ["1-15", "16-30", "31-45", "46-60"]
    dist = dist.reindex(ordem, fill_value=0)
    return pd.DataFrame({"quadrante": dist.index, "quantidade": dist.values})


def frequencia_pares_de_dezenas(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Frequência de cada par específico de dezenas que já saiu junto."""
    m = matriz_dezenas(df)
    contagem = Counter()
    for linha in m:
        for a, b in itertools.combinations(sorted(linha.tolist()), 2):
            contagem[(a, b)] += 1
    out = pd.DataFrame(
        [{"dezena_a": a, "dezena_b": b, "frequencia": c} for (a, b), c in contagem.items()]
    ).sort_values("frequencia", ascending=False).reset_index(drop=True)
    return out.head(top)


def frequencia_trios(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Frequência de trios específicos de dezenas.

    Aviso importante (pedido explicitamente pelo escopo do projeto):
    existem C(60,3) = 34.220 trios possíveis. Com poucas centenas de
    concursos, a esmagadora maioria dos trios nunca vai se repetir --
    isso é o comportamento ESPERADO de um processo aleatório, não uma
    anomalia. Trios que aparecem 2x não são "padrões", são o tipo de
    coincidência que qualquer amostra aleatória desse tamanho produz.
    """
    m = matriz_dezenas(df)
    contagem = Counter()
    for linha in m:
        for a, b, c in itertools.combinations(sorted(linha.tolist()), 3):
            contagem[(a, b, c)] += 1
    out = pd.DataFrame(
        [{"dezena_a": a, "dezena_b": b, "dezena_c": c, "frequencia": f}
         for (a, b, c), f in contagem.items()]
    ).sort_values("frequencia", ascending=False).reset_index(drop=True)
    return out.head(top)


def intervalos_entre_dezenas_ordenadas(df: pd.DataFrame) -> pd.DataFrame:
    """Distribuição das diferenças entre dezenas consecutivas dentro do
    mesmo concurso, já ordenadas (ex: concurso 1-10-20 -> diferenças 9, 10)."""
    m = matriz_dezenas(df)
    diffs = np.diff(m, axis=1).flatten()
    dist = pd.Series(diffs).value_counts().sort_index()
    return pd.DataFrame({"intervalo": dist.index, "frequencia": dist.values})


def baixos_e_altos(df: pd.DataFrame, corte: int = 30) -> pd.DataFrame:
    """Distribuição da quantidade de números 'baixos' (<= corte) por concurso.
    O valor de `corte` é explicitado para não deixar ambíguo o critério usado."""
    m = matriz_dezenas(df)
    qtd_baixos = (m <= corte).sum(axis=1)
    dist = pd.Series(qtd_baixos).value_counts().sort_index()
    out = pd.DataFrame({"qtd_baixos": dist.index, "qtd_concursos": dist.values})
    out["qtd_altos"] = 6 - out["qtd_baixos"]
    out["corte_usado"] = f"baixo = 1-{corte}, alto = {corte + 1}-60"
    return out


def repeticao_multi_janela(df: pd.DataFrame) -> pd.DataFrame:
    """Junta repetição vs. concurso anterior e vs. últimos 5 concursos."""
    r1 = repeticao_entre_concursos(df, janela=1)
    r5 = repeticao_entre_concursos(df, janela=5)
    return pd.DataFrame({"concurso": df["concurso"].values, "repetidas_vs_anterior": r1.values,
                          "repetidas_vs_ultimos_5": r5.values})
