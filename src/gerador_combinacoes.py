"""
gerador_combinacoes.py

Gerador de combinações para fins de SIMULAÇÃO e organização de apostas.
Nada aqui é uma "previsão": mesmo a opção baseada em ranking experimental
apenas prioriza a ORDEM em que as dezenas são incluídas na combinação,
sem qualquer garantia de que isso melhore a chance de acerto.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

N_DEZENAS = 60
N_SORTEADAS = 6


def total_combinacoes(n=N_DEZENAS, k=N_SORTEADAS) -> int:
    return math.comb(n, k)


def probabilidade_com_n_apostas(n_apostas: int, n=N_DEZENAS, k=N_SORTEADAS) -> float:
    """P(acertar a sena) com n_apostas combinações DISTINTAS: n_apostas / C(n,k).
    Aumenta a cobertura proporcionalmente, mas não informa quais números
    sairão nem muda a probabilidade de cada combinação individual."""
    return n_apostas / total_combinacoes(n, k)


@dataclass
class ResultadoGeracao:
    combinacoes: list[tuple[int, ...]]
    quantidade: int
    custo_total: float
    probabilidade_total: float
    combinacoes_excluidas_por_filtro: int
    aviso: str


def gerar_combinacoes(
    quantidade_desejada: int,
    preco_aposta: float,
    orcamento_maximo: float | None = None,
    modo: str = "aleatorio",
    scores_ranking: np.ndarray | None = None,
    filtro_soma: tuple[int, int] | None = None,
    filtro_paridade: tuple[int, int] | None = None,
    semente: int = 42,
) -> ResultadoGeracao:
    """Gera combinações distintas de 6 dezenas.

    Parameters
    ----------
    quantidade_desejada: quantas combinações tentar gerar
    preco_aposta: preço de uma aposta simples (parâmetro do usuário --
        este módulo não assume nenhum valor "oficial" de preço, já que
        isso muda ao longo do tempo)
    orcamento_maximo: se informado, limita a quantidade final pelo custo
    modo: "aleatorio" ou "ranking" (usa scores_ranking para priorizar
        dezenas de maior score, mas ainda com aleatoriedade -- não é uma
        seleção determinística "dos 6 melhores", para não sugerir uma
        falsa certeza)
    filtro_soma: (min, max) -- exclui combinações fora dessa faixa de soma
    filtro_paridade: (min_pares, max_pares) -- exclui combinações fora
        dessa faixa de quantidade de números pares
    """
    rng = np.random.default_rng(semente)
    combinacoes: set[tuple[int, ...]] = set()
    excluidas = 0
    tentativas_max = quantidade_desejada * 200 + 1000

    if modo == "ranking" and scores_ranking is not None:
        pesos = np.clip(scores_ranking, 1e-6, None)
        pesos = pesos / pesos.sum()
    else:
        pesos = None

    tentativas = 0
    universo = np.arange(1, N_DEZENAS + 1)
    while len(combinacoes) < quantidade_desejada and tentativas < tentativas_max:
        tentativas += 1
        if pesos is not None:
            escolha = rng.choice(universo, size=N_SORTEADAS, replace=False, p=pesos)
        else:
            escolha = rng.choice(universo, size=N_SORTEADAS, replace=False)
        combo = tuple(sorted(int(x) for x in escolha))

        if filtro_soma is not None:
            soma = sum(combo)
            if not (filtro_soma[0] <= soma <= filtro_soma[1]):
                excluidas += 1
                continue
        if filtro_paridade is not None:
            pares = sum(1 for x in combo if x % 2 == 0)
            if not (filtro_paridade[0] <= pares <= filtro_paridade[1]):
                excluidas += 1
                continue

        combinacoes.add(combo)

    lista = list(combinacoes)

    if orcamento_maximo is not None:
        max_por_orcamento = int(orcamento_maximo // preco_aposta)
        lista = lista[:max_por_orcamento]

    custo_total = len(lista) * preco_aposta
    prob_total = probabilidade_com_n_apostas(len(lista))

    aviso = (
        "Estas combinações são para fins de simulação/organização de apostas. "
        "Um ranking experimental usado no modo 'ranking' influencia apenas a "
        "PRIORIDADE de inclusão de dezenas -- não garante nem aumenta a "
        "probabilidade real de acerto de cada combinação, que permanece "
        f"1/{total_combinacoes()} para qualquer combinação, gerada por "
        "qualquer método."
    )
    if filtro_soma is not None or filtro_paridade is not None:
        aviso += (
            f" Foram excluídas {excluidas} combinações candidatas pelos filtros "
            "aplicados. Excluir combinações fora de um padrão comum NÃO aumenta "
            "a probabilidade das combinações restantes -- apenas reduz quantas "
            "combinações 'atípicas' você joga."
        )

    return ResultadoGeracao(
        combinacoes=lista,
        quantidade=len(lista),
        custo_total=custo_total,
        probabilidade_total=prob_total,
        combinacoes_excluidas_por_filtro=excluidas,
        aviso=aviso,
    )
