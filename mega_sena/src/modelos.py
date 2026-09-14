"""
modelos.py

Modelos experimentais de RANQUEAMENTO de dezenas. Nenhum modelo aqui
"prevê" o próximo sorteio -- cada um produz uma pontuação (score) para
as 60 dezenas com base apenas em informação disponível ATÉ o concurso
anterior ao que está sendo avaliado (nunca informação futura).

Todos os modelos seguem a mesma assinatura:

    modelo(historico_ate_t: np.ndarray) -> np.ndarray de shape (60,)

onde `historico_ate_t` é a matriz (m, 6) de concursos JÁ OCORRIDOS antes
do concurso a ser avaliado, e o retorno é um vetor de scores (maior =
"mais favorecido" pelo modelo) para as dezenas 1..60.

Escolher os 6 números de maior score é uma forma de gerar uma combinação
experimental -- não uma previsão validada.
"""

from __future__ import annotations

import numpy as np

N_DEZENAS = 60
N_SORTEADAS = 6


def modelo_aleatorio(historico_ate_t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ignora completamente o histórico. Serve de linha de base obrigatória:
    qualquer modelo que não supere isso, de forma consistente e fora da
    amostra, não está agregando informação real."""
    return rng.random(N_DEZENAS)


def modelo_frequencia_acumulada(historico_ate_t: np.ndarray) -> np.ndarray:
    """Score = frequência absoluta da dezena em TODO o histórico disponível.
    Hipótese (não comprovada): dezenas historicamente mais frequentes
    continuam mais frequentes. Risco: se o processo é realmente aleatório,
    isso é só ruído -- e ainda corre o risco de dar peso enorme a desvios
    de amostras pequenas no início da série."""
    if len(historico_ate_t) == 0:
        return np.ones(N_DEZENAS)
    freq = np.bincount(historico_ate_t.flatten(), minlength=N_DEZENAS + 1)[1:]
    return freq.astype(float)


def modelo_frequencia_janela_movel(historico_ate_t: np.ndarray, janela: int = 50) -> np.ndarray:
    """Score = frequência apenas nos últimos `janela` concursos.
    Hipótese: o processo pode ter mudanças de regime (equipamento, bolas,
    procedimento) e dados muito antigos "atrapalham" mais do que ajudam.
    Risco: com janela pequena, a estimativa fica ruidosa; a escolha do
    tamanho da janela é ela mesma um parâmetro que pode ser overfitado
    se for ajustado olhando o desempenho passado."""
    if len(historico_ate_t) == 0:
        return np.ones(N_DEZENAS)
    recorte = historico_ate_t[-janela:]
    freq = np.bincount(recorte.flatten(), minlength=N_DEZENAS + 1)[1:]
    return freq.astype(float)


def modelo_frequencia_ponderada(historico_ate_t: np.ndarray, meia_vida: int = 100) -> np.ndarray:
    """Score = frequência ponderada, dando peso exponencialmente maior a
    concursos recentes (meia_vida controla a velocidade do decaimento).
    Mesma hipótese e mesmos riscos do modelo de janela móvel, mas com
    transição suave em vez de corte abrupto."""
    m = len(historico_ate_t)
    if m == 0:
        return np.ones(N_DEZENAS)
    idade = np.arange(m - 1, -1, -1)  # 0 = concurso mais recente
    pesos = 0.5 ** (idade / meia_vida)
    scores = np.zeros(N_DEZENAS)
    for i in range(m):
        for dez in historico_ate_t[i]:
            scores[dez - 1] += pesos[i]
    return scores


def modelo_tempo_desde_ultima(historico_ate_t: np.ndarray) -> np.ndarray:
    """Score = quantos concursos se passaram desde a última vez que a
    dezena saiu ("atraso"). Hipótese popular e NÃO comprovada de que
    dezenas atrasadas têm mais chance de sair -- sob independência entre
    sorteios, isso é a falácia do apostador: o atraso passado não afeta
    a probabilidade do próximo sorteio."""
    m = len(historico_ate_t)
    scores = np.full(N_DEZENAS, m, dtype=float)  # nunca saiu = atraso máximo
    for dez in range(1, N_DEZENAS + 1):
        pos = np.where((historico_ate_t == dez).any(axis=1))[0] if m else np.array([])
        if len(pos):
            scores[dez - 1] = m - 1 - pos[-1]
    return scores


def modelo_frequencia_pares(historico_ate_t: np.ndarray) -> np.ndarray:
    """Score de cada dezena = soma das frequências dos pares em que ela
    participa, ponderada pela frequência do parceiro mais comum.
    Hipótese: dezenas que fazem parte de pares frequentes podem ser
    "preferidas" -- hipótese fraca, já que pares frequentes tendem a
    simplesmente refletir dezenas individualmente frequentes."""
    m = len(historico_ate_t)
    if m == 0:
        return np.ones(N_DEZENAS)
    pair_score = np.zeros(N_DEZENAS)
    for linha in historico_ate_t:
        for i in range(6):
            for j in range(6):
                if i != j:
                    pair_score[linha[i] - 1] += 1  # conta co-ocorrências
    return pair_score


def modelo_beta_binomial(
    historico_ate_t: np.ndarray, alpha_prior: float = 1.0, beta_prior: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Modelo bayesiano Beta-Binomial para a taxa de ocorrência de cada dezena.

    - Prior: Beta(alpha_prior, beta_prior). Com alpha=beta=1, é a Beta
      uniforme (não-informativa) -- não presumimos, antes de ver dados,
      que nenhuma dezena é mais provável que outra.
    - Verossimilhança: cada concurso é (aproximadamente, tratando as 6
      dezenas como 6 tentativas independentes, o que é uma simplificação
      já que na prática a extração é sem reposição dentro do concurso)
      uma Binomial(1, p_dezena) por concurso.
    - Posterior: Beta(alpha_prior + sucessos, beta_prior + fracassos).
    - Retorna a MÉDIA posterior de cada dezena e o intervalo de
      credibilidade de 95% (usando a função quantil da Beta).

    Importante: mesmo a média posterior mais alta não significa que essa
    dezena "vai sair" -- é apenas a melhor estimativa (sob este modelo
    simplificado) da taxa histórica de longo prazo dessa dezena,
    suavizada pelo prior.
    """
    from scipy import stats as scipy_stats

    m = len(historico_ate_t)
    sucessos = np.bincount(historico_ate_t.flatten(), minlength=N_DEZENAS + 1)[1:] if m else np.zeros(N_DEZENAS)
    fracassos = m - sucessos
    alpha_post = alpha_prior + sucessos
    beta_post = beta_prior + fracassos
    media_posterior = alpha_post / (alpha_post + beta_post)
    ic_inferior = scipy_stats.beta.ppf(0.025, alpha_post, beta_post)
    ic_superior = scipy_stats.beta.ppf(0.975, alpha_post, beta_post)
    return media_posterior, np.stack([ic_inferior, ic_superior], axis=1)


def modelo_combinado(historico_ate_t: np.ndarray, pesos: dict | None = None) -> np.ndarray:
    """Combina frequência acumulada, atraso e frequência em janela móvel
    com pesos FIXADOS ANTES da avaliação (não ajustados olhando o
    resultado) -- exatamente para evitar que o ato de "combinar modelos"
    vire uma forma disfarçada de overfitting.
    """
    pesos = pesos or {"frequencia_acumulada": 0.4, "janela_movel": 0.4, "atraso": 0.2}
    freq = modelo_frequencia_acumulada(historico_ate_t)
    freq_n = freq / freq.max() if freq.max() > 0 else freq
    janela = modelo_frequencia_janela_movel(historico_ate_t, janela=50)
    janela_n = janela / janela.max() if janela.max() > 0 else janela
    atraso = modelo_tempo_desde_ultima(historico_ate_t)
    atraso_n = atraso / atraso.max() if atraso.max() > 0 else atraso
    return (pesos["frequencia_acumulada"] * freq_n
            + pesos["janela_movel"] * janela_n
            + pesos["atraso"] * atraso_n)


def modelo_regressao_logistica(historico_ate_t: np.ndarray) -> np.ndarray:
    """Regressão logística simples: para cada dezena, treina um modelo que
    prevê P(dezena sai no próximo concurso) usando como features a
    frequência acumulada até aquele ponto e o atraso atual, calculadas em
    cada passo do próprio histórico (sem olhar o futuro). É um modelo
    experimental simples, não uma prova de causalidade.
    """
    from sklearn.linear_model import LogisticRegression

    m = len(historico_ate_t)
    if m < 20:  # dados insuficientes para treinar algo minimamente estável
        return np.ones(N_DEZENAS)

    scores = np.zeros(N_DEZENAS)
    for dez in range(1, N_DEZENAS + 1):
        freqs, atrasos_, alvo = [], [], []
        atraso_corrente = 0
        freq_corrente = 0
        for i in range(m):
            freqs.append(freq_corrente)
            atrasos_.append(atraso_corrente)
            saiu = dez in historico_ate_t[i]
            alvo.append(int(saiu))
            if saiu:
                freq_corrente += 1
                atraso_corrente = 0
            else:
                atraso_corrente += 1
        X = np.column_stack([freqs, atrasos_])
        y = np.array(alvo)
        if y.sum() == 0 or y.sum() == len(y):
            scores[dez - 1] = y.mean()
            continue
        try:
            clf = LogisticRegression(max_iter=200)
            clf.fit(X, y)
            x_novo = np.array([[freq_corrente, atraso_corrente]])
            scores[dez - 1] = clf.predict_proba(x_novo)[0, 1]
        except Exception:
            scores[dez - 1] = y.mean()
    return scores


REGISTRO_MODELOS = {
    "aleatorio": "Linha de base: ignora o histórico.",
    "frequencia_acumulada": "Frequência histórica total de cada dezena.",
    "janela_movel": "Frequência apenas nos últimos N concursos.",
    "frequencia_ponderada": "Frequência com peso exponencialmente maior para concursos recentes.",
    "tempo_desde_ultima": "Atraso: concursos desde a última ocorrência (falácia do apostador testável).",
    "frequencia_pares": "Pontuação baseada em co-ocorrência com outras dezenas.",
    "beta_binomial": "Modelo bayesiano com prior não-informativo.",
    "combinado": "Combinação linear de 3 modelos, com pesos fixados a priori.",
    "regressao_logistica": "Regressão logística simples por dezena (frequência + atraso como features).",
}
