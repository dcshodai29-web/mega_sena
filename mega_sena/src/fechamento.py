"""
fechamento.py

Sistema de fechamento (wheel) simples: fechamento COMPLETO, ou seja,
dado um conjunto de k dezenas escolhidas pelo usuário, gera TODAS as
apostas de 6 dezenas contidas nesse conjunto (C(k,6) apostas).

Este é o único tipo de fechamento com garantia matemática simples e
verificável: se as 6 dezenas sorteadas estiverem TODAS dentro das k
dezenas escolhidas, uma das apostas geradas acerta a sena. Fechamentos
"reduzidos" (menos que C(k,6) apostas, com garantia parcial) existem na
prática, mas exigem estruturas combinatórias específicas (sistemas de
cobertura) que estão fora do escopo desta primeira versão -- isso é
declarado explicitamente para não fingir uma garantia que o código não
implementa.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

N_SORTEADAS = 6
TOTAL_COMBINACOES_60_6 = math.comb(60, 6)


@dataclass
class ResultadoFechamento:
    dezenas_escolhidas: list[int]
    quantidade_dezenas: int
    quantidade_apostas: int
    apostas: list[tuple[int, ...]]
    custo_total: float
    probabilidade_das_6_sorteadas_estarem_no_conjunto: float
    condicoes_de_garantia: str
    o_que_nao_e_garantido: str
    risco_divisao_premio: str


def gerar_fechamento_completo(dezenas_escolhidas: list[int], preco_aposta: float) -> ResultadoFechamento:
    """Gera o fechamento completo (todas as combinações de 6 dentro do
    conjunto escolhido)."""
    dezenas_escolhidas = sorted(set(dezenas_escolhidas))
    k = len(dezenas_escolhidas)
    if k < N_SORTEADAS:
        raise ValueError(f"É preciso escolher pelo menos {N_SORTEADAS} dezenas (recebido: {k}).")
    if any(not (1 <= d <= 60) for d in dezenas_escolhidas):
        raise ValueError("Todas as dezenas devem estar entre 1 e 60.")

    apostas = list(itertools.combinations(dezenas_escolhidas, N_SORTEADAS))
    quantidade_apostas = len(apostas)
    custo_total = quantidade_apostas * preco_aposta

    # probabilidade de que as 6 dezenas realmente sorteadas estejam TODAS
    # dentro do conjunto de k escolhidas: hipergeométrica P(X=6) com
    # população 60, sucessos k, amostra 6
    from scipy.stats import hypergeom
    prob_todas_dentro = float(hypergeom.pmf(N_SORTEADAS, 60, k, N_SORTEADAS))

    condicoes = (
        f"GARANTIA CONDICIONAL: se as 6 dezenas sorteadas pela Mega-Sena estiverem "
        f"TODAS entre as {k} dezenas que você escolheu, então, necessariamente, "
        f"UMA das {quantidade_apostas} apostas geradas acerta a sena (6 pontos). "
        f"Esta é uma garantia matemática do fechamento completo, não uma "
        f"estimativa."
    )
    nao_garantido = (
        "NÃO é garantido que as 6 dezenas sorteadas estarão dentro do seu "
        f"conjunto de {k} escolhidas -- isso continua sendo um evento aleatório, "
        f"com probabilidade {prob_todas_dentro:.8f} "
        f"(aproximadamente 1 em {round(1/prob_todas_dentro):,}) se as dezenas "
        "forem escolhidas sem nenhuma informação preditiva real. O fechamento "
        "não aumenta essa probabilidade -- ele garante apenas que, SE isso "
        "acontecer, você não vai deixar de ter uma aposta premiada com 6 "
        "pontos dentro do conjunto coberto."
    )
    risco_divisao = (
        "Se você e outras pessoas acertarem a sena no mesmo concurso, o prêmio "
        "é dividido entre todos os ganhadores -- o fechamento não protege "
        "contra isso, e conjuntos de dezenas populares (ex: datas de "
        "aniversário, 1 a 31) tendem a ser escolhidos por mais apostadores."
    )

    return ResultadoFechamento(
        dezenas_escolhidas=dezenas_escolhidas,
        quantidade_dezenas=k,
        quantidade_apostas=quantidade_apostas,
        apostas=apostas,
        custo_total=custo_total,
        probabilidade_das_6_sorteadas_estarem_no_conjunto=prob_todas_dentro,
        condicoes_de_garantia=condicoes,
        o_que_nao_e_garantido=nao_garantido,
        risco_divisao_premio=risco_divisao,
    )
