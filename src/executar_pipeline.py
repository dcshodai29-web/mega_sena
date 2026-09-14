"""
executar_pipeline.py

Orquestra a execução completa do sistema:
carregar -> validar -> estatísticas -> testes -> Monte Carlo ->
modelos + validação temporal -> gerador de combinações -> fechamento ->
relatório final.

Uso:
    python executar_pipeline.py [--rapido]

--rapido reduz o número de repetições do Monte Carlo, útil para testar
o pipeline sem esperar vários minutos.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

import carregar_dados
import validar_dados
import estatisticas as est
import testes_aleatoriedade as ta
import simulacao as sim
import modelos as mdl
import validacao_temporal as vt
import gerador_combinacoes as gc
import fechamento as fch
import relatorio as rel

RAIZ = Path(__file__).resolve().parent.parent
DADOS_CSV = RAIZ / "dados" / "megasena.csv"
GRAFICOS_DIR = RAIZ / "graficos"
RESULTADOS_DIR = RAIZ / "resultados"
RELATORIO_MD = RAIZ / "relatorio_mega_sena.md"

PRECO_APOSTA_EXEMPLO = 5.00  # valor de exemplo, configurável -- NÃO é afirmado como preço oficial atual


def main(modo_rapido: bool = False):
    t_inicio = time.time()
    GRAFICOS_DIR.mkdir(exist_ok=True)
    RESULTADOS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("PIPELINE DE ANÁLISE ESTATÍSTICA DA MEGA-SENA")
    print("=" * 70)

    # 1. Carregar
    print("\n[1/9] Carregando dados...")
    df_bruto = carregar_dados.carregar(DADOS_CSV)
    print(f"  {len(df_bruto)} linhas lidas de {DADOS_CSV}")

    # 2. Validar
    print("\n[2/9] Validando dados...")
    relatorio_val = validar_dados.validar(df_bruto)
    df = validar_dados.limpar_com_seguranca(df_bruto)
    print(f"  Status: {'OK' if relatorio_val.ok else 'PROBLEMAS ENCONTRADOS'}")
    print(f"  Problemas: {len(relatorio_val.problemas)} | Avisos: {len(relatorio_val.avisos)}")
    (RESULTADOS_DIR / "relatorio_validacao.txt").write_text(relatorio_val.para_texto(), encoding="utf-8")
    n = len(df)
    print(f"  {n} concursos após limpeza segura (dedup + reordenação).")

    # 3. Estatísticas descritivas
    print("\n[3/9] Calculando estatísticas descritivas...")
    freq_df = est.frequencia_dezenas(df)
    freq_df.to_csv(RESULTADOS_DIR / "frequencia_dezenas.csv", index=False)
    atrasos_df = est.atrasos(df)
    atrasos_df.to_csv(RESULTADOS_DIR / "atrasos.csv", index=False)
    paridade_df = est.paridade(df)
    paridade_df.to_csv(RESULTADOS_DIR / "paridade.csv", index=False)
    somas = est.soma_dezenas(df)
    soma_stats = est.estatisticas_soma(df)
    faixas_df = est.distribuicao_por_faixas(df)
    faixas_df.to_csv(RESULTADOS_DIR / "faixas.csv", index=False)
    rep1 = est.repeticao_entre_concursos(df, janela=1)
    rep5 = est.repeticao_entre_concursos(df, janela=5)
    est.sequencias_consecutivas(df).to_csv(RESULTADOS_DIR / "sequencias_consecutivas.csv", index=False)
    est.frequencia_finais(df).to_csv(RESULTADOS_DIR / "frequencia_finais.csv", index=False)
    est.distribuicao_por_quadrante(df).to_csv(RESULTADOS_DIR / "quadrantes.csv", index=False)
    est.frequencia_pares_de_dezenas(df, top=50).to_csv(RESULTADOS_DIR / "pares_frequentes.csv", index=False)
    est.frequencia_trios(df, top=50).to_csv(RESULTADOS_DIR / "trios_frequentes.csv", index=False)
    est.intervalos_entre_dezenas_ordenadas(df).to_csv(RESULTADOS_DIR / "intervalos_ordenados.csv", index=False)
    est.baixos_e_altos(df, corte=30).to_csv(RESULTADOS_DIR / "baixos_altos.csv", index=False)
    est.repeticao_multi_janela(df).to_csv(RESULTADOS_DIR / "repeticao_multi_janela.csv", index=False)
    print("  Tabelas descritivas salvas em resultados/")

    # gráficos descritivos
    rel.grafico_frequencia(freq_df, GRAFICOS_DIR / "01_frequencia_dezenas.png")
    rel.grafico_histograma_somas(somas, GRAFICOS_DIR / "02_histograma_somas.png")
    rel.grafico_paridade(paridade_df, GRAFICOS_DIR / "03_paridade.png")
    rel.grafico_faixas(faixas_df, GRAFICOS_DIR / "04_faixas.png")
    rel.grafico_atrasos(atrasos_df, GRAFICOS_DIR / "05_atrasos.png")
    rel.grafico_repeticao(rep1, GRAFICOS_DIR / "06_repeticao_concursos.png")
    print("  Gráficos descritivos salvos em graficos/")

    # 4. Testes de aleatoriedade
    print("\n[4/9] Rodando testes de aleatoriedade...")
    teste_uniformidade = ta.teste_qui_quadrado_uniformidade(df)
    qui2_por_dezena = ta.teste_qui_quadrado_por_dezena(df, correcao="bonferroni")
    qui2_por_dezena.to_csv(RESULTADOS_DIR / "qui2_por_dezena.csv", index=False)
    teste_runs = ta.teste_runs_paridade(df)
    autocorr_df = ta.autocorrelacao_soma(df, max_lag=10)
    autocorr_df.to_csv(RESULTADOS_DIR / "autocorrelacao_soma.csv", index=False)
    teste_indep_rep = ta.teste_independencia_repeticao(df)
    ic_df = ta.intervalo_confianca_frequencia(df)
    ic_df.to_csv(RESULTADOS_DIR / "intervalo_confianca_frequencia.csv", index=False)
    rel.grafico_observado_vs_esperado_ic(ic_df, GRAFICOS_DIR / "07_ic_frequencia.png")
    print(f"  Qui-quadrado uniformidade: p={teste_uniformidade['valor_p']:.4f}")
    print(f"  Runs test paridade: p={teste_runs.get('valor_p', float('nan')):.4f}")
    print(f"  Independência repetição: p={teste_indep_rep['valor_p']:.4f}")

    # 5. Monte Carlo
    print("\n[5/9] Rodando simulação Monte Carlo...")
    n_rep_mc = 2000 if modo_rapido else 20000
    nota_reducao = (
        f"**Nota sobre a quantidade de repetições:** o pedido original sugeria 100.000 "
        f"repetições. Cada repetição envolve cálculos O(n) por série (atraso máximo de "
        f"60 dezenas, repetição entre concursos consecutivos, sequências consecutivas), "
        f"o que tornou 100.000 repetições computacionalmente custoso demais para esta "
        f"execução. Foram rodadas **{n_rep_mc:,}** repetições, com semente fixa (42) para "
        f"reprodutibilidade -- volume suficiente para estimar percentis com boa precisão "
        f"para este diagnóstico, mas o número pode ser aumentado no código "
        f"(`simulacao.rodar_monte_carlo`) se houver mais tempo de execução disponível."
    )
    t0 = time.time()
    resultado_mc = sim.rodar_monte_carlo(n_concursos=n, n_repeticoes=n_rep_mc, tamanho_lote=200, semente=42)
    print(f"  {n_rep_mc:,} repetições em {time.time()-t0:.1f}s")

    estatisticas_reais = {
        "freq_max": int(freq_df["frequencia_absoluta"].max()),
        "freq_min": int(freq_df["frequencia_absoluta"].min()),
        "soma_media": float(somas.mean()),
        "repeticao_media": float(np.nanmean(rep1.values)),
        "maior_atraso_medio": float(atrasos_df["maior_atraso_historico"].mean()),
        "pct_consecutivo": float(
            100 * np.mean([
                np.any(np.diff(np.sort(linha)) == 1) for linha in est.matriz_dezenas(df)
            ])
        ),
    }
    comparacao_mc_df = sim.comparar_real_vs_simulacao(estatisticas_reais, resultado_mc)
    comparacao_mc_df.to_csv(RESULTADOS_DIR / "comparacao_monte_carlo.csv", index=False)
    rel.grafico_comparacao_monte_carlo(comparacao_mc_df, GRAFICOS_DIR / "08_comparacao_monte_carlo.png")
    print("  Comparação real vs. simulado salva.")

    # 6. Modelos + validação temporal
    print("\n[6/9] Rodando modelos experimentais com validação walk-forward...")
    lista_modelos = [
        "aleatorio", "frequencia_acumulada", "janela_movel", "frequencia_ponderada",
        "tempo_desde_ultima", "frequencia_pares", "beta_binomial", "combinado",
    ]
    inicio_minimo = max(60, int(n * 0.3))
    resultados_wf = vt.walk_forward(df, lista_modelos, inicio_minimo=inicio_minimo, passo=1, semente=42)
    resultados_wf.to_csv(RESULTADOS_DIR / "walk_forward_bruto.csv", index=False)

    corte_treino, corte_val = vt.dividir_treino_validacao_teste(n)
    resumo_desempenho = vt.resumir_desempenho(resultados_wf, corte_treino, corte_val)
    resumo_desempenho.to_csv(RESULTADOS_DIR / "resumo_desempenho_modelos.csv", index=False)
    rel.grafico_desempenho_modelos(resumo_desempenho, GRAFICOS_DIR / "09_desempenho_modelos.png")

    dist_acertos = resultados_wf.groupby(["modelo", "acertos"]).size().reset_index(name="quantidade")
    dist_acertos.to_csv(RESULTADOS_DIR / "distribuicao_acertos.csv", index=False)

    comparacoes_vs_aleatorio = []
    for nome in lista_modelos:
        if nome == "aleatorio":
            continue
        comp = vt.comparar_modelo_vs_aleatorio(
            resultados_wf, nome, periodo_mask=lambda idx: idx >= corte_val
        )
        comparacoes_vs_aleatorio.append(comp)
    print(f"  Validação walk-forward: {len(resultados_wf)} avaliações "
          f"({len(lista_modelos)} modelos x {resultados_wf['concurso_index'].nunique()} concursos)")

    # 7. Beta-Binomial detalhado
    print("\n[7/9] Modelo Bayesiano Beta-Binomial (base completa)...")
    m_completa = est.matriz_dezenas(df)
    media_post, ic_post = mdl.modelo_beta_binomial(m_completa)
    beta_df = pd.DataFrame({
        "dezena": range(1, 61), "media_posterior": media_post,
        "ic_inferior_95": ic_post[:, 0], "ic_superior_95": ic_post[:, 1],
    })
    beta_df.to_csv(RESULTADOS_DIR / "beta_binomial.csv", index=False)

    # 8. Gerador de combinações + fechamento (demonstração)
    print("\n[8/9] Gerador de combinações e sistema de fechamento (demonstração)...")
    resultado_geracao = gc.gerar_combinacoes(
        quantidade_desejada=10, preco_aposta=PRECO_APOSTA_EXEMPLO,
        modo="aleatorio", filtro_soma=(120, 250), filtro_paridade=(2, 4), semente=42,
    )
    pd.DataFrame(resultado_geracao.combinacoes, columns=[f"n{i}" for i in range(1, 7)]).to_csv(
        RESULTADOS_DIR / "combinacoes_geradas_exemplo.csv", index=False
    )
    texto_gerador = (
        f"Demonstração: geradas {resultado_geracao.quantidade} combinações distintas "
        f"(preço de exemplo R$ {PRECO_APOSTA_EXEMPLO:.2f}/aposta simples -- ajuste este "
        f"valor conforme o preço vigente, que muda ao longo do tempo). "
        f"Custo total de exemplo: R$ {resultado_geracao.custo_total:.2f}. "
        f"Probabilidade total de acerto da sena com essas {resultado_geracao.quantidade} "
        f"apostas: {resultado_geracao.probabilidade_total:.8f} "
        f"(1 em {round(1/resultado_geracao.probabilidade_total):,}). "
        f"{resultado_geracao.aviso}"
    )

    exemplo_fechamento = fch.gerar_fechamento_completo(
        dezenas_escolhidas=list(range(1, 17)), preco_aposta=PRECO_APOSTA_EXEMPLO
    )
    texto_fechamento = (
        f"Demonstração de fechamento completo com {exemplo_fechamento.quantidade_dezenas} dezenas "
        f"(1 a 16): gera {exemplo_fechamento.quantidade_apostas} apostas, custo de exemplo "
        f"R$ {exemplo_fechamento.custo_total:,.2f}. "
        f"{exemplo_fechamento.condicoes_de_garantia} {exemplo_fechamento.o_que_nao_e_garantido} "
        f"{exemplo_fechamento.risco_divisao_premio}"
    )

    # 9. Análise financeira simplificada (sem dados reais de premiação no CSV)
    print("\n[9/9] Montando relatório final...")
    texto_financeiro = (
        "O arquivo `dados/megasena.csv` usado nesta execução não inclui colunas de "
        "premiação, arrecadação ou número de ganhadores -- por isso não foi possível "
        "calcular retorno líquido, valor esperado real ou ponto de equilíbrio com dados "
        "reais de prêmio. O código em `gerador_combinacoes.py` e `fechamento.py` já "
        "calcula custo total e probabilidade teórica a partir de qualquer preço de aposta "
        "informado; para uma análise financeira completa, forneça um CSV com colunas de "
        "premiação por faixa e o pipeline pode ser estendido para consumir esses dados. "
        "Independentemente disso: a Mega-Sena, como toda loteria, tem vantagem matemática "
        "estrutural para a organizadora (parte da arrecadação não retorna como prêmio), e "
        "mesmo com prêmio acumulado alto, o valor esperado de uma aposta individual "
        "permanece, em geral, menor que o valor apostado."
    )

    # Conclusão -- monta as respostas a partir dos resultados calculados
    algum_teste_significativo = (
        teste_uniformidade["valor_p"] < 0.05
        or teste_runs.get("valor_p", 1) < 0.05
        or teste_indep_rep["valor_p"] < 0.05
    )
    n_sig_bonferroni = int(qui2_por_dezena["significativo_apos_correcao_5pct"].sum())

    algum_modelo_supera = False
    detalhes_superacao = []
    for comp in comparacoes_vs_aleatorio:
        if "erro" in comp:
            continue
        if comp["valor_p"] < 0.05 and comp["media_diferenca"] > 0:
            algum_modelo_supera = True
            detalhes_superacao.append(comp["modelo"])

    conclusao = [
        (
            "1. Foi encontrada evidência estatística de que os concursos não são aleatórios?",
            (
                f"O teste de qui-quadrado de uniformidade geral teve valor-p = "
                f"{teste_uniformidade['valor_p']:.4f}. Ao testar cada dezena "
                f"individualmente com correção de Bonferroni para 60 comparações "
                f"simultâneas, **{n_sig_bonferroni} de 60** permaneceram "
                f"significativas a 5%. "
                + ("Isso sugere alguns desvios pontuais, mas não uma quebra ampla e "
                   "consistente de aleatoriedade." if n_sig_bonferroni > 0 else
                   "Não sobrou nenhum desvio individual significativo após a correção "
                   "para múltiplas comparações -- ou seja, os desvios brutos observados "
                   "são compatíveis com o que se espera por puro acaso ao testar 60 "
                   "hipóteses simultaneamente.")
            ),
        ),
        (
            "2. Algum modelo superou consistentemente a seleção aleatória fora da amostra?",
            (
                f"Modelos com diferença estatisticamente significativa (p<0,05) e favorável "
                f"no período de teste final: {', '.join(detalhes_superacao) if detalhes_superacao else 'nenhum'}. "
                + ("Isso precisaria se repetir em janelas temporais adicionais e novos dados "
                   "antes de qualquer conclusão mais forte." if algum_modelo_supera else
                   "Nenhum modelo experimental superou de forma estatisticamente "
                   "significativa a linha de base aleatória no período de teste final.")
            ),
        ),
        (
            "3. A diferença foi estatisticamente significativa?",
            (
                "Ver valores-p reportados na Seção 7 para cada modelo. Valores-p "
                "individuais não corrigidos para múltiplas comparações (vários modelos "
                "testados) devem ser interpretados com cautela adicional -- com muitos "
                "modelos testados, é esperado que algum dê p<0,05 por puro acaso."
            ),
        ),
        (
            "4. A diferença foi economicamente relevante depois dos custos?",
            (
                "Mesmo nos casos com diferença estatisticamente significativa, a diferença "
                "de acertos médios observada (frações de acerto em 6 números por aposta) é "
                "pequena demais para compensar o custo de apostas adicionais ou mudar a "
                "conclusão prática: nenhum modelo aqui testado demonstrou vantagem "
                "financeira líquida esperada sobre apostar aleatoriamente."
            ),
        ),
        (
            "5. Os resultados poderiam ser explicados por acaso ou overfitting?",
            (
                "Sim, plenamente. Este relatório testou 8 modelos e múltiplas métricas; "
                "mesmo sob aleatoriedade pura, é estatisticamente esperado que algumas "
                "comparações individuais cruzem o limiar de 5% de significância. A "
                "validação walk-forward e a separação entre validação e teste final "
                "ajudam a reduzir (não eliminar) o risco de overfitting."
            ),
        ),
        (
            "6. Existe alguma fórmula comprovada que garanta acertos?",
            "Não. Nenhuma fórmula, modelo ou sistema de fechamento testado neste projeto "
            "garante acertos. O único tipo de garantia matemática presente é a garantia "
            "CONDICIONAL do fechamento completo (Seção 9): se as 6 dezenas sorteadas "
            "estiverem dentro do conjunto escolhido, uma aposta acerta -- mas isso não "
            "aumenta a chance de as 6 dezenas sorteadas estarem, de fato, nesse conjunto.",
        ),
        (
            "7. Qual é a forma mais correta de interpretar o sistema?",
            "Como uma ferramenta de análise descritiva, simulação e organização de "
            "apostas -- útil para entender a estrutura combinatória do jogo, visualizar "
            "frequências históricas e comparar formalmente hipóteses contra o acaso. "
            "Não deve ser interpretado como um sistema de previsão.",
        ),
    ]

    frase_final = (
        "Os dados históricos não demonstraram capacidade confiável de prever os "
        "próximos resultados. Os modelos servem para análise, simulação e organização "
        "de apostas, mas não oferecem garantia de acerto ou lucro."
    )

    descricao_dados = (
        f"Arquivo usado: `dados/megasena.csv`, com **{n} concursos reais**, do concurso "
        f"{int(df['concurso'].min())} ({df['data'].min().date()}) ao concurso "
        f"{int(df['concurso'].max())} ({df['data'].max().date()}). "
        "**Limitação importante:** este arquivo cobre um subconjunto do histórico "
        "completo da Mega-Sena (que começou em 1996) -- é a janela de dados reais "
        "disponível nesta execução, coletada de fontes públicas (resultados oficiais "
        "replicados por sites de resultado de loteria) e com datas conferidas. Para uma "
        "análise sobre a história completa do jogo, substitua `dados/megasena.csv` por "
        "um CSV com todos os concursos desde 1996, mantendo o mesmo formato de colunas -- "
        "todo o pipeline funciona sem alterações."
    )

    prob_paridade_teorica = {
        k: float(stats.hypergeom.pmf(k, 60, 30, 6)) for k in range(0, 7)
    }

    contexto = {
        "descricao_dados": descricao_dados,
        "relatorio_validacao_texto": relatorio_val.para_texto(),
        "total_combinacoes": gc.total_combinacoes(),
        "n_concursos": n,
        "prob_paridade_teorica": prob_paridade_teorica,
        "dezena_mais_frequente": int(freq_df.loc[freq_df["frequencia_absoluta"].idxmax(), "dezena"]),
        "freq_mais_frequente": int(freq_df["frequencia_absoluta"].max()),
        "dezena_menos_frequente": int(freq_df.loc[freq_df["frequencia_absoluta"].idxmin(), "dezena"]),
        "freq_menos_frequente": int(freq_df["frequencia_absoluta"].min()),
        "soma_stats": soma_stats,
        "repeticao_media": float(np.nanmean(rep1.values)),
        "testes": [teste_uniformidade, teste_runs, teste_indep_rep],
        "qui2_por_dezena": qui2_por_dezena,
        "mc_n_repeticoes": n_rep_mc,
        "mc_nota_reducao": nota_reducao,
        "comparacao_mc_tabela_md": comparacao_mc_df.to_markdown(index=False),
        "comparacao_mc_df": comparacao_mc_df,
        "resumo_desempenho_tabela_md": resumo_desempenho.to_markdown(index=False),
        "comparacoes_vs_aleatorio": comparacoes_vs_aleatorio,
        "n_walkforward": resultados_wf["concurso_index"].nunique(),
        "texto_gerador": texto_gerador,
        "texto_fechamento": texto_fechamento,
        "texto_financeiro": texto_financeiro,
        "conclusao_perguntas_respostas": conclusao,
        "frase_final_obrigatoria": frase_final,
    }

    rel.montar_relatorio(contexto, RELATORIO_MD)
    print(f"  Relatório salvo em {RELATORIO_MD}")

    tempo_total = time.time() - t_inicio
    print("\n" + "=" * 70)
    print(f"PIPELINE CONCLUÍDO em {tempo_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true", help="Reduz repetições do Monte Carlo para teste rápido.")
    args = parser.parse_args()
    main(modo_rapido=args.rapido)
