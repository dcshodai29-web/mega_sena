"""
Testes automatizados básicos.

Cobrem: fórmulas de probabilidade, validação de dados (casos com erro
propositalmente inseridos) e sanidade dos módulos de estatística e
gerador de combinações. Não é uma suíte exaustiva, mas cobre os pontos
mais fáceis de quebrar silenciosamente.

Execução: a partir da pasta mega_sena/, rode `pytest testes/ -v`.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import validar_dados
import carregar_dados
import estatisticas as est
import gerador_combinacoes as gc
import fechamento as fch

tipar_dataframe = carregar_dados.tipar_dataframe


def _df_valido(n=30, semente=1):
    rng = np.random.default_rng(semente)
    linhas = []
    for i in range(1, n + 1):
        dezenas = sorted(rng.choice(np.arange(1, 61), size=6, replace=False).tolist())
        linhas.append({
            "concurso": i, "data": pd.Timestamp("2023-01-01") + pd.Timedelta(days=3 * i),
            "n1": dezenas[0], "n2": dezenas[1], "n3": dezenas[2],
            "n4": dezenas[3], "n5": dezenas[4], "n6": dezenas[5],
        })
    return pd.DataFrame(linhas)


# ---------- Fórmulas de probabilidade ----------

def test_combinacoes_60_6():
    assert gc.total_combinacoes() == 50_063_860


def test_probabilidade_aposta_simples():
    p = 1 / gc.total_combinacoes()
    assert math.isclose(p, gc.probabilidade_com_n_apostas(1), rel_tol=1e-12)


def test_probabilidade_escala_linear_com_apostas():
    p10 = gc.probabilidade_com_n_apostas(10)
    p1 = gc.probabilidade_com_n_apostas(1)
    assert math.isclose(p10, 10 * p1, rel_tol=1e-9)


def test_soma_probabilidades_paridade_igual_a_1():
    from scipy.stats import hypergeom
    total = sum(hypergeom.pmf(k, 60, 30, 6) for k in range(0, 7))
    assert math.isclose(total, 1.0, rel_tol=1e-9)


# ---------- Validação de dados ----------

def test_dados_bons_passam_sem_problemas():
    df = tipar_dataframe(_df_valido())
    rel = validar_dados.validar(df)
    assert rel.ok
    assert rel.linhas_com_dezena_fora_do_intervalo == 0
    assert rel.linhas_com_dezena_repetida == 0


def test_detecta_dezena_fora_do_intervalo():
    df = _df_valido()
    df.loc[0, "n1"] = 61  # fora de 1-60
    df = tipar_dataframe(df)
    rel = validar_dados.validar(df)
    assert not rel.ok
    assert rel.linhas_com_dezena_fora_do_intervalo == 1


def test_detecta_dezena_repetida_no_mesmo_concurso():
    df = _df_valido()
    df.loc[0, "n2"] = df.loc[0, "n1"]
    df = tipar_dataframe(df)
    rel = validar_dados.validar(df)
    assert not rel.ok
    assert rel.linhas_com_dezena_repetida == 1


def test_detecta_concurso_duplicado():
    df = _df_valido()
    nova_linha = df.iloc[0].copy()
    nova_linha["data"] = df.iloc[1]["data"]  # mesmo concurso, data diferente
    df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
    df = tipar_dataframe(df)
    rel = validar_dados.validar(df)
    assert not rel.ok
    assert int(df.iloc[0]["concurso"]) in rel.concursos_duplicados


def test_detecta_concurso_faltante():
    df = _df_valido(n=10)
    df = df[df["concurso"] != 5]  # remove o concurso 5
    df = tipar_dataframe(df)
    rel = validar_dados.validar(df)
    assert 5 in rel.concursos_faltantes


def test_limpar_remove_duplicata_exata():
    df = _df_valido(n=5)
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df_dup = tipar_dataframe(df_dup)
    limpo = validar_dados.limpar_com_seguranca(df_dup)
    assert len(limpo) == 5


# ---------- Estatísticas ----------

def test_frequencia_soma_correta():
    df = tipar_dataframe(_df_valido(n=50))
    freq = est.frequencia_dezenas(df)
    # cada concurso contribui 6 ocorrências; soma de todas as frequências = n*6
    assert freq["frequencia_absoluta"].sum() == 50 * 6


def test_matriz_dezenas_shape():
    df = tipar_dataframe(_df_valido(n=20))
    m = est.matriz_dezenas(df)
    assert m.shape == (20, 6)


def test_soma_dezenas_dentro_dos_limites_teoricos():
    df = tipar_dataframe(_df_valido(n=40))
    s = est.soma_dezenas(df)
    assert s.min() >= 1 + 2 + 3 + 4 + 5 + 6  # menor soma teórica possível
    assert s.max() <= 55 + 56 + 57 + 58 + 59 + 60  # maior soma teórica possível


# ---------- Gerador de combinações ----------

def test_gerador_nao_repete_combinacoes():
    resultado = gc.gerar_combinacoes(quantidade_desejada=20, preco_aposta=5.0, semente=7)
    assert len(resultado.combinacoes) == len(set(resultado.combinacoes))


def test_gerador_respeita_orcamento():
    resultado = gc.gerar_combinacoes(
        quantidade_desejada=100, preco_aposta=5.0, orcamento_maximo=30.0, semente=3
    )
    assert resultado.custo_total <= 30.0 + 1e-9
    assert resultado.quantidade <= 6


# ---------- Fechamento ----------

def test_fechamento_quantidade_apostas():
    resultado = fch.gerar_fechamento_completo(list(range(1, 8)), preco_aposta=5.0)
    assert resultado.quantidade_apostas == math.comb(7, 6)


def test_fechamento_exige_minimo_6_dezenas():
    with pytest.raises(ValueError):
        fch.gerar_fechamento_completo([1, 2, 3], preco_aposta=5.0)


def test_fechamento_garante_acerto_se_todas_dezenas_sorteadas_estao_no_conjunto():
    conjunto = list(range(1, 13))  # 12 dezenas
    resultado = fch.gerar_fechamento_completo(conjunto, preco_aposta=5.0)
    sorteio_simulado = (1, 2, 3, 4, 5, 6)  # hipotético, todas dentro do conjunto
    assert sorteio_simulado in resultado.apostas
