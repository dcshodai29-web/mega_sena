"""
relatorio.py

Gera os gráficos (matplotlib) e monta o relatório final em Markdown,
juntando os resultados de todos os outros módulos. A seção final
"Conclusão sobre previsibilidade" é OBRIGATÓRIA e é escrita a partir
dos resultados calculados, não de um texto fixo otimista.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _salvar(fig, caminho: Path):
    fig.tight_layout()
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


def grafico_frequencia(freq_df: pd.DataFrame, caminho: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    cores = ["#2ecc71" if v >= freq_df["esperado_sob_uniformidade"].iloc[0] else "#95a5a6"
             for v in freq_df["frequencia_absoluta"]]
    ax.bar(freq_df["dezena"], freq_df["frequencia_absoluta"], color=cores)
    ax.axhline(freq_df["esperado_sob_uniformidade"].iloc[0], color="#e74c3c", linestyle="--",
               label="Esperado sob uniformidade")
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Frequência absoluta")
    ax.set_title("Frequência de cada dezena (1-60)")
    ax.legend()
    _salvar(fig, caminho)


def grafico_histograma_somas(somas: pd.Series, caminho: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(somas, bins=30, color="#3498db", edgecolor="white")
    ax.axvline(somas.mean(), color="#e74c3c", linestyle="--", label=f"Média = {somas.mean():.1f}")
    ax.set_xlabel("Soma das 6 dezenas")
    ax.set_ylabel("Número de concursos")
    ax.set_title("Distribuição da soma das dezenas sorteadas")
    ax.legend()
    _salvar(fig, caminho)


def grafico_paridade(paridade_df: pd.DataFrame, caminho: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(paridade_df["qtd_pares"], paridade_df["qtd_concursos"], color="#9b59b6")
    ax.set_xlabel("Quantidade de números pares no concurso")
    ax.set_ylabel("Número de concursos")
    ax.set_title("Distribuição de paridade (pares x ímpares)")
    _salvar(fig, caminho)


def grafico_faixas(faixas_df: pd.DataFrame, caminho: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(faixas_df["faixa"], faixas_df["quantidade"], color="#f39c12")
    ax.set_xlabel("Faixa de dezenas")
    ax.set_ylabel("Quantidade sorteada")
    ax.set_title("Distribuição das dezenas sorteadas por faixa de 10")
    _salvar(fig, caminho)


def grafico_atrasos(atrasos_df: pd.DataFrame, caminho: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ordenado = atrasos_df.sort_values("atraso_atual", ascending=False)
    ax.bar(ordenado["dezena"].astype(str), ordenado["atraso_atual"], color="#1abc9c")
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Concursos desde a última ocorrência")
    ax.set_title("Atraso atual de cada dezena")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    _salvar(fig, caminho)


def grafico_repeticao(rep_series: pd.Series, caminho: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rep_series.values, color="#e67e22", linewidth=0.8)
    media = np.nanmean(rep_series.values)
    ax.axhline(media, color="#2c3e50", linestyle="--", label=f"Média = {media:.2f}")
    ax.set_xlabel("Índice do concurso (cronológico)")
    ax.set_ylabel("Dezenas repetidas vs. concurso anterior")
    ax.set_title("Repetição de dezenas entre concursos consecutivos")
    ax.legend()
    _salvar(fig, caminho)


def grafico_observado_vs_esperado_ic(ic_df: pd.DataFrame, caminho: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(
        ic_df["dezena"], ic_df["proporcao_observada"],
        yerr=[ic_df["proporcao_observada"] - ic_df["ic_inferior"],
              ic_df["ic_superior"] - ic_df["proporcao_observada"]],
        fmt="o", markersize=3, color="#2980b9", ecolor="#aed6f1", elinewidth=1, capsize=2,
    )
    ax.axhline(6 / 60, color="#e74c3c", linestyle="--", label="Proporção teórica (0,10)")
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Proporção observada (com IC 95%)")
    ax.set_title("Frequência observada de cada dezena com intervalo de confiança")
    ax.legend()
    _salvar(fig, caminho)


def grafico_comparacao_monte_carlo(comparacao_df: pd.DataFrame, caminho: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(comparacao_df["metrica"], comparacao_df["percentil_do_valor_real"] * 100, color="#8e44ad")
    ax.axvline(50, color="#7f8c8d", linestyle=":")
    ax.axvline(2.5, color="#e74c3c", linestyle="--", linewidth=0.8)
    ax.axvline(97.5, color="#e74c3c", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Percentil do valor REAL na distribuição simulada (Monte Carlo)")
    ax.set_title("Real vs. simulações puramente aleatórias\n(fora de 2,5%-97,5% = incomum)")
    _salvar(fig, caminho)


def grafico_desempenho_modelos(resumo_df: pd.DataFrame, caminho: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = resumo_df.pivot(index="modelo", columns="periodo", values="media")
    pivot = pivot.reindex(columns=[c for c in ["treino", "validacao", "teste_final"] if c in pivot.columns])
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(6 * 6 / 60, color="#e74c3c", linestyle="--", linewidth=1,
               label="Esperado ao acaso (0,6 acertos)")
    ax.set_ylabel("Média de acertos por aposta (0 a 6)")
    ax.set_title("Desempenho médio de cada modelo, por período")
    ax.legend()
    _salvar(fig, caminho)


def montar_relatorio(contexto: dict, caminho_saida: Path) -> str:
    """Monta o relatório final em Markdown a partir de um dicionário com
    todos os resultados já calculados (`contexto`). Ver executar_pipeline.py
    para o formato esperado das chaves de `contexto`."""

    linhas = []
    linhas.append("# Relatório Técnico — Análise Estatística e Simulação da Mega-Sena")
    linhas.append("")
    linhas.append(
        "> **Aviso obrigatório de honestidade estatística:** este relatório NÃO promete "
        "acertos, lucro ou sucesso garantido. A Mega-Sena é, até prova em contrário, um "
        "processo de sorteio independente e aleatório. Nada aqui constitui aconselhamento "
        "financeiro ou uma fórmula para vencer a loteria."
    )
    linhas.append("")

    linhas.append("## 1. Fonte e período dos dados")
    linhas.append("")
    linhas.append(contexto["descricao_dados"])
    linhas.append("")

    linhas.append("## 2. Validação dos dados")
    linhas.append("")
    linhas.append("```")
    linhas.append(contexto["relatorio_validacao_texto"])
    linhas.append("```")
    linhas.append("")

    linhas.append("## 3. Fundamentos matemáticos")
    linhas.append("")
    linhas.append(f"- Combinações possíveis (C(60,6)): **{contexto['total_combinacoes']:,}**")
    linhas.append(f"- Probabilidade de uma aposta simples acertar a sena: **1 / {contexto['total_combinacoes']:,}** "
                   f"≈ {1/contexto['total_combinacoes']:.10f}")
    linhas.append(f"- Probabilidade de uma dezena específica sair em um concurso: 6/60 = 0,10")
    linhas.append(f"- Frequência esperada de uma dezena em N={contexto['n_concursos']} concursos: "
                   f"{contexto['n_concursos'] * 0.10:.1f} (desvio-padrão ≈ "
                   f"{(contexto['n_concursos'] * 0.10 * 0.90) ** 0.5:.2f})")
    linhas.append("")
    linhas.append("**Probabilidade de sair exatamente k números pares (30 pares e 30 ímpares em 60):**")
    linhas.append("")
    linhas.append("| k pares | P(K=k) |")
    linhas.append("|---|---|")
    for k, p in contexto["prob_paridade_teorica"].items():
        linhas.append(f"| {k} | {p:.4%} |")
    linhas.append("")
    linhas.append(
        "Note que certas contagens de paridade são mais frequentes simplesmente porque "
        "existem mais combinações possíveis com aquela estrutura (ex: 3 pares e 3 ímpares "
        "é a mais numerosa), e não porque o sorteio 'prefere' esse padrão."
    )
    linhas.append("")

    linhas.append("## 4. Análise descritiva")
    linhas.append("")
    linhas.append("Gráficos gerados na pasta `graficos/` e tabelas completas em `resultados/`. Destaques:")
    linhas.append("")
    linhas.append(f"- Dezena mais sorteada: **{contexto['dezena_mais_frequente']}** "
                   f"({contexto['freq_mais_frequente']} vezes)")
    linhas.append(f"- Dezena menos sorteada: **{contexto['dezena_menos_frequente']}** "
                   f"({contexto['freq_menos_frequente']} vezes)")
    linhas.append(f"- Soma média das 6 dezenas: {contexto['soma_stats']['media']:.1f} "
                   f"(mediana {contexto['soma_stats']['mediana']:.0f}, "
                   f"mín {contexto['soma_stats']['minimo']}, máx {contexto['soma_stats']['maximo']})")
    linhas.append(f"- Repetição média de dezenas vs. concurso anterior: {contexto['repeticao_media']:.2f} "
                   f"(de 6 possíveis)")
    linhas.append("")

    linhas.append("## 5. Testes de aleatoriedade e independência")
    linhas.append("")
    for teste in contexto["testes"]:
        linhas.append(f"### {teste['nome']}")
        linhas.append(f"- H0: {teste.get('h0', '—')}")
        linhas.append(f"- H1: {teste.get('h1', '—')}")
        if "estatistica" in teste:
            linhas.append(f"- Estatística: {teste['estatistica']:.4f}")
        if "valor_p" in teste:
            linhas.append(f"- Valor-p: {teste['valor_p']:.4f}")
        if "tamanho_efeito_cramer_v" in teste:
            linhas.append(f"- Tamanho de efeito (Cramér's V): {teste['tamanho_efeito_cramer_v']:.4f}")
        linhas.append(f"- Interpretação: {teste.get('interpretacao', '—')}")
        if "limitacoes" in teste:
            linhas.append(f"- Limitações: {teste['limitacoes']}")
        linhas.append("")

    n_sig_bonferroni = int(contexto["qui2_por_dezena"]["significativo_apos_correcao_5pct"].sum())
    linhas.append(
        f"**Teste por dezena com correção de Bonferroni (60 testes simultâneos):** "
        f"{n_sig_bonferroni} de 60 dezenas permaneceram estatisticamente significativas "
        f"após a correção (tabela completa em `resultados/qui2_por_dezena.csv`)."
    )
    linhas.append("")

    linhas.append("## 6. Simulação Monte Carlo")
    linhas.append("")
    linhas.append(
        f"Foram simuladas **{contexto['mc_n_repeticoes']:,} séries** inteiramente aleatórias "
        f"de {contexto['n_concursos']} concursos cada (mesmo tamanho da base real), para "
        "servir de referência sobre o que é 'normal' esperar de puro acaso."
    )
    linhas.append("")
    linhas.append(contexto["mc_nota_reducao"])
    linhas.append("")
    linhas.append(contexto["comparacao_mc_tabela_md"])
    linhas.append("")
    n_incomuns = int(contexto["comparacao_mc_df"]["incomum_5pct"].sum())
    linhas.append(
        f"**{n_incomuns} de {len(contexto['comparacao_mc_df'])}** métricas comparadas caíram fora "
        "da faixa considerada comum (percentil 2,5%-97,5%) frente às simulações aleatórias."
    )
    linhas.append("")

    linhas.append("## 7. Modelos experimentais e validação temporal")
    linhas.append("")
    linhas.append(
        "Todos os modelos foram avaliados com validação **walk-forward**: em cada passo, o "
        "modelo só usa concursos anteriores ao que está sendo avaliado. Nenhuma informação "
        "futura foi usada em nenhum momento."
    )
    linhas.append("")
    linhas.append(contexto["resumo_desempenho_tabela_md"])
    linhas.append("")
    linhas.append("### Comparação de cada modelo contra a linha de base aleatória (teste final)")
    linhas.append("")
    for comp in contexto["comparacoes_vs_aleatorio"]:
        if "erro" in comp:
            continue
        linhas.append(
            f"- **{comp['modelo']}**: média {comp['media_acertos_modelo']:.3f} vs. "
            f"aleatório {comp['media_acertos_aleatorio']:.3f} "
            f"(diferença {comp['media_diferenca']:+.3f}, valor-p Wilcoxon = {comp['valor_p']:.4f}, "
            f"n={comp['n_pares']})"
        )
    linhas.append("")
    linhas.append(
        "**Sobre eventos raros:** acertos de 5 ou 6 números são extremamente raros mesmo em "
        f"{contexto['n_walkforward']} avaliações walk-forward; a tabela de distribuição de "
        "acertos (`resultados/distribuicao_acertos.csv`) mostra a contagem exata, e "
        "conclusões sobre essas faixas específicas devem ser tratadas com cautela adicional "
        "por causa do tamanho pequeno da amostra nessas faixas."
    )
    linhas.append("")

    linhas.append("## 8. Modelo bayesiano Beta-Binomial")
    linhas.append("")
    linhas.append(
        "Prior utilizado: Beta(1,1) — uniforme, não-informativo (nenhuma dezena é presumida "
        "mais provável que outra antes de olhar os dados). A tabela completa com média "
        "posterior e intervalo de credibilidade de 95% para cada dezena está em "
        "`resultados/beta_binomial.csv`."
    )
    linhas.append("")
    linhas.append(
        "Uma média posterior mais alta reflete apenas uma taxa histórica de longo prazo "
        "suavizada — não uma previsão de que aquela dezena sairá no próximo concurso."
    )
    linhas.append("")

    linhas.append("## 9. Gerador de combinações e sistemas de fechamento")
    linhas.append("")
    linhas.append(contexto["texto_gerador"])
    linhas.append("")
    linhas.append(contexto["texto_fechamento"])
    linhas.append("")

    linhas.append("## 10. Análise financeira")
    linhas.append("")
    linhas.append(contexto["texto_financeiro"])
    linhas.append("")

    linhas.append("## 11. Conclusão sobre previsibilidade")
    linhas.append("")
    for pergunta, resposta in contexto["conclusao_perguntas_respostas"]:
        linhas.append(f"**{pergunta}**")
        linhas.append("")
        linhas.append(resposta)
        linhas.append("")
    linhas.append("---")
    linhas.append("")
    linhas.append(
        "> " + contexto["frase_final_obrigatoria"]
    )
    linhas.append("")

    texto = "\n".join(linhas)
    caminho_saida.write_text(texto, encoding="utf-8")
    return texto
