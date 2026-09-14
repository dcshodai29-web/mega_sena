"""
testes_aleatoriedade.py

Testes formais de hipótese sobre aleatoriedade e independência.

Princípio central deste módulo: um teste estatístico responde uma
pergunta MUITO específica e limitada. Rejeitar H0 não prova
"previsibilidade útil"; não rejeitar H0 não prova "aleatoriedade
perfeita". Cada função devolve um dicionário com hipótese nula,
estatística, valor-p, tamanho de efeito (quando aplicável) e uma
interpretação textual que deixa essas limitações explícitas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from estatisticas import matriz_dezenas, N_DEZENAS, N_SORTEADAS


def teste_qui_quadrado_uniformidade(df: pd.DataFrame) -> dict:
    """H0: todas as 60 dezenas têm a mesma probabilidade de serem sorteadas.
    H1: pelo menos uma dezena tem probabilidade diferente das demais.
    """
    m = matriz_dezenas(df)
    n_concursos = len(df)
    observado = np.bincount(m.flatten(), minlength=N_DEZENAS + 1)[1:]
    esperado = np.full(N_DEZENAS, n_concursos * N_SORTEADAS / N_DEZENAS)
    chi2, p = stats.chisquare(observado, esperado)
    graus_liberdade = N_DEZENAS - 1
    # tamanho de efeito: Cramér's V para bondade de ajuste
    n_total = observado.sum()
    v_cramer = float(np.sqrt(chi2 / (n_total * (min(2, N_DEZENAS) - 1)))) if n_total else np.nan
    return {
        "nome": "Qui-quadrado de uniformidade (frequência das 60 dezenas)",
        "h0": "Todas as dezenas têm a mesma probabilidade de serem sorteadas (1/60 cada).",
        "h1": "Ao menos uma dezena tem probabilidade diferente de 1/60.",
        "estatistica": float(chi2),
        "graus_liberdade": int(graus_liberdade),
        "valor_p": float(p),
        "tamanho_efeito_cramer_v": v_cramer,
        "interpretacao": (
            "Valor-p baixo indicaria desvio da uniformidade nas frequências "
            "observadas. Mesmo que isso ocorra, não implica que uma dezena "
            "específica seja 'mais provável' no PRÓXIMO sorteio -- sorteios "
            "futuros continuam sendo eventos novos e independentes do "
            "histórico, a menos que exista evidência de um mecanismo físico "
            "de viés (o que este teste sozinho não pode demonstrar)."
        ),
        "limitacoes": (
            "Teste sensível ao tamanho da amostra: com poucos concursos, "
            "tem pouco poder para detectar desvios pequenos; com muitos "
            "concursos, pode rejeitar H0 por desvios triviais sem "
            "relevância prática -- por isso reportamos também o tamanho "
            "de efeito (Cramér's V), não só o valor-p."
        ),
    }


def teste_qui_quadrado_por_dezena(df: pd.DataFrame, correcao: str = "bonferroni") -> pd.DataFrame:
    """Testa CADA dezena individualmente (binomial exato: dezena sai ou não
    em cada concurso, p=6/60=0.1) e aplica correção para múltiplas
    comparações -- OBRIGATÓRIO aqui, já que 60 testes simultâneos vão
    gerar 'significâncias' falsas por puro acaso se não corrigirmos.
    """
    m = matriz_dezenas(df)
    n = len(df)
    p_esperado = N_SORTEADAS / N_DEZENAS
    linhas = []
    for dez in range(1, N_DEZENAS + 1):
        k = int((m == dez).any(axis=1).sum())
        resultado = stats.binomtest(k, n, p_esperado, alternative="two-sided")
        linhas.append({"dezena": dez, "vezes_sorteada": k, "esperado": n * p_esperado,
                        "valor_p_bruto": resultado.pvalue})
    out = pd.DataFrame(linhas)

    m_testes = len(out)
    if correcao == "bonferroni":
        out["valor_p_corrigido"] = np.minimum(out["valor_p_bruto"] * m_testes, 1.0)
    else:  # fdr (Benjamini-Hochberg)
        ordem = out["valor_p_bruto"].argsort()
        ranks = np.empty(m_testes, dtype=int)
        ranks[ordem] = np.arange(1, m_testes + 1)
        out["valor_p_corrigido"] = np.minimum(out["valor_p_bruto"] * m_testes / ranks, 1.0)

    out["significativo_apos_correcao_5pct"] = out["valor_p_corrigido"] < 0.05
    return out.sort_values("valor_p_bruto")


def teste_runs_paridade(df: pd.DataFrame) -> dict:
    """Teste de sequências (runs test) sobre a série binária: soma do
    concurso é par ou ímpar. H0: a sequência de par/ímpar é aleatória
    (sem tendência a 'grudar' em sequências longas nem alternar demais).
    """
    m = matriz_dezenas(df)
    soma = m.sum(axis=1)
    binaria = (soma % 2 == 0).astype(int)  # 1 = soma par, 0 = soma impar
    n = len(binaria)
    n1 = int(binaria.sum())
    n0 = n - n1
    runs = 1 + int(np.sum(binaria[1:] != binaria[:-1]))

    if n0 == 0 or n1 == 0:
        return {"nome": "Runs test (paridade da soma)", "aplicavel": False,
                "motivo": "Uma das categorias não ocorre na amostra."}

    media_runs = (2 * n0 * n1) / n + 1
    var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n ** 2 * (n - 1))
    z = (runs - media_runs) / np.sqrt(var_runs) if var_runs > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan

    return {
        "nome": "Runs test sobre paridade da soma dos 6 números",
        "h0": "A alternância entre concursos de soma par e soma ímpar é consistente com uma sequência aleatória.",
        "h1": "Há dependência sequencial (sequências mais longas ou mais curtas do que o esperado ao acaso).",
        "n_par": n1, "n_impar": n0, "runs_observados": runs, "runs_esperados": float(media_runs),
        "estatistica_z": float(z), "valor_p": float(p),
        "interpretacao": (
            "Este teste avalia se um concurso 'depende' do padrão de "
            "paridade dos concursos vizinhos. Não testa dezenas "
            "específicas e não implica capacidade preditiva mesmo se "
            "significativo."
        ),
    }


def autocorrelacao_soma(df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """Autocorrelação da série de somas dos concursos, para várias defasagens
    (lags). H0 implícita em cada lag: autocorrelação = 0 (sem dependência
    linear entre a soma do concurso t e do concurso t-lag).
    """
    s = matriz_dezenas(df).sum(axis=1).astype(float)
    s = s - s.mean()
    n = len(s)
    var0 = np.sum(s ** 2)
    linhas = []
    for lag in range(1, max_lag + 1):
        cov = np.sum(s[lag:] * s[:-lag])
        r = cov / var0 if var0 > 0 else np.nan
        # erro padrão aproximado sob H0 de ruído branco: 1/sqrt(n)
        se = 1 / np.sqrt(n)
        z = r / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        linhas.append({"lag": lag, "autocorrelacao": float(r), "erro_padrao_aprox": se,
                        "estatistica_z": float(z), "valor_p": float(p)})
    out = pd.DataFrame(linhas)
    m_testes = len(out)
    out["valor_p_bonferroni"] = np.minimum(out["valor_p"] * m_testes, 1.0)
    return out


def teste_independencia_repeticao(df: pd.DataFrame) -> dict:
    """Compara a distribuição observada de 'quantas dezenas se repetem
    entre concursos consecutivos' com a distribuição teórica sob
    independência (hipergeométrica: 6 escolhidas de 60, comparando com
    as 6 do concurso anterior).
    """
    m = matriz_dezenas(df)
    n = len(m)
    obs = []
    for i in range(1, n):
        obs.append(len(set(m[i].tolist()) & set(m[i - 1].tolist())))
    obs = np.array(obs)
    obs_contagem = np.bincount(obs, minlength=7)

    # distribuição teórica hipergeométrica: população 60, sucessos 6 (do concurso anterior),
    # amostra 6 (concurso atual)
    k = np.arange(0, 7)
    prob_teorica = stats.hypergeom.pmf(k, 60, 6, 6)
    esperado = prob_teorica * len(obs)

    chi2, p = stats.chisquare(obs_contagem, esperado)

    return {
        "nome": "Independência entre concursos consecutivos (repetição de dezenas)",
        "h0": "O número de dezenas repetidas entre concursos consecutivos segue a distribuição "
              "hipergeométrica esperada sob sorteios independentes.",
        "h1": "A distribuição observada de repetições difere da esperada sob independência.",
        "distribuicao_observada": obs_contagem.tolist(),
        "distribuicao_esperada": esperado.tolist(),
        "estatistica_qui2": float(chi2),
        "valor_p": float(p),
        "interpretacao": (
            "Se o valor-p for baixo, a quantidade de dezenas repetidas entre "
            "sorteios consecutivos foge do esperado sob independência -- mas "
            "isso ainda não indica QUAIS dezenas tendem a se repetir nem "
            "fornece uma regra útil de aposta."
        ),
    }


def intervalo_confianca_frequencia(df: pd.DataFrame, confianca: float = 0.95) -> pd.DataFrame:
    """Intervalo de confiança (Wilson) para a proporção de sorteios em que
    cada dezena aparece, comparado com a proporção teórica 6/60=0.10."""
    m = matriz_dezenas(df)
    n = len(df)
    p_teorico = N_SORTEADAS / N_DEZENAS
    z = stats.norm.ppf(1 - (1 - confianca) / 2)
    linhas = []
    for dez in range(1, N_DEZENAS + 1):
        k = int((m == dez).any(axis=1).sum())
        p_hat = k / n
        denom = 1 + z ** 2 / n
        centro = (p_hat + z ** 2 / (2 * n)) / denom
        margem = (z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denom
        contem_teorico = (centro - margem) <= p_teorico <= (centro + margem)
        linhas.append({
            "dezena": dez, "proporcao_observada": p_hat,
            "ic_inferior": centro - margem, "ic_superior": centro + margem,
            "contem_proporcao_teorica_0_10": bool(contem_teorico),
        })
    return pd.DataFrame(linhas)
