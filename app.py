"""
app.py — Interface web (Streamlit) do sistema de análise estatística da Mega-Sena.

Este arquivo é o ponto de entrada para publicar o projeto gratuitamente
no Streamlit Community Cloud (https://share.streamlit.io). Ele reaproveita
os módulos já existentes em src/ -- nenhuma lógica estatística foi
duplicada ou reescrita aqui, só a camada de interface.

Rodar localmente (opcional, para testar antes de publicar):
    streamlit run app.py
"""

from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

import carregar_dados
import validar_dados
import estatisticas as est
import testes_aleatoriedade as ta
import simulacao as sim
import modelos as mdl
import validacao_temporal as vt
import gerador_combinacoes as gc
import fechamento as fch

st.set_page_config(page_title="Mega-Sena — Análise Estatística", page_icon="🎲", layout="wide")

AVISO = (
    "⚠️ **Isto é uma ferramenta de análise, simulação e organização de apostas — não um "
    "previsor.** Nenhuma combinação gerada aqui tem probabilidade maior de ser sorteada do "
    "que qualquer outra. A Mega-Sena é um processo de sorteio independente e aleatório."
)


def renderizar_bolas(numeros: list[int], cor_inicio="#2ecc71", cor_fim="#1a9e53", cor_texto="#06210f") -> str:
    """Monta HTML de bolinhas coloridas (estilo volante) para exibir uma combinação."""
    bolas = "".join(
        f'<div style="width:38px;height:38px;border-radius:50%;'
        f'background:linear-gradient(135deg,{cor_inicio},{cor_fim});color:{cor_texto};'
        f'display:flex;align-items:center;justify-content:center;font-weight:700;'
        f'font-size:14px;font-family:-apple-system,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,.3);">'
        f'{n:02d}</div>'
        for n in numeros
    )
    return f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:10px 0;">{bolas}</div>'


def link_whatsapp(texto: str) -> str:
    return "https://wa.me/?text=" + urllib.parse.quote(texto)


def texto_whatsapp_jogos(lista_jogos: list[list[int]], ultimo_concurso: int, ultima_data, total_sorteios: int) -> str:
    linhas = ["🎲 *Meus jogos da Mega-Sena*", ""]
    for i, jogo in enumerate(lista_jogos, start=1):
        prefixo = f"Jogo {i}: " if len(lista_jogos) > 1 else ""
        linhas.append(f"{prefixo}{'-'.join(f'{n:02d}' for n in jogo)}")
    linhas.append("")
    linhas.append(f"Base: {total_sorteios} sorteios reais, até o concurso {ultimo_concurso} ({ultima_data}).")
    linhas.append("Toda combinação tem a mesma probabilidade — isso é só uma forma de organizar os números 🍀")
    return "\n".join(linhas)


@st.cache_data
def carregar_base(caminho_ou_bytes, usar_upload: bool) -> pd.DataFrame:
    if usar_upload:
        df_bruto = pd.read_csv(caminho_ou_bytes)
        faltantes = [c for c in carregar_dados.COLUNAS_OBRIGATORIAS if c not in df_bruto.columns]
        if faltantes:
            raise ValueError(f"Colunas obrigatórias ausentes: {faltantes}")
        df_bruto = carregar_dados.tipar_dataframe(df_bruto)
    else:
        df_bruto = carregar_dados.carregar(caminho_ou_bytes)
    return df_bruto


@st.cache_data
def analise_completa(df_hash: str, df: pd.DataFrame) -> dict:
    """Roda a análise descritiva + testes (rápido, cacheado pelo conteúdo dos dados)."""
    freq_df = est.frequencia_dezenas(df)
    atrasos_df = est.atrasos(df)
    paridade_df = est.paridade(df)
    somas = est.soma_dezenas(df)
    soma_stats = est.estatisticas_soma(df)
    faixas_df = est.distribuicao_por_faixas(df)
    rep1 = est.repeticao_entre_concursos(df, janela=1)
    teste_uniformidade = ta.teste_qui_quadrado_uniformidade(df)
    qui2_por_dezena = ta.teste_qui_quadrado_por_dezena(df)
    teste_runs = ta.teste_runs_paridade(df)
    teste_indep = ta.teste_independencia_repeticao(df)
    ic_df = ta.intervalo_confianca_frequencia(df)
    pares_df = est.frequencia_pares_de_dezenas(df, top=20)
    return dict(
        freq_df=freq_df, atrasos_df=atrasos_df, paridade_df=paridade_df, somas=somas,
        soma_stats=soma_stats, faixas_df=faixas_df, rep1=rep1,
        teste_uniformidade=teste_uniformidade, qui2_por_dezena=qui2_por_dezena,
        teste_runs=teste_runs, teste_indep=teste_indep, ic_df=ic_df, pares_df=pares_df,
    )


def hash_df(df: pd.DataFrame) -> str:
    return pd.util.hash_pandas_object(df).sum().__str__()


# ---------------------------------------------------------------- Sidebar
st.sidebar.title("🎲 Mega-Sena")
st.sidebar.caption("Sistema educacional de análise estatística")

fonte = st.sidebar.radio(
    "Base de dados",
    ["Usar base incluída no projeto", "Enviar meu próprio CSV"],
    help="O CSV precisa ter as colunas: concurso,data,n1,n2,n3,n4,n5,n6",
)

if fonte == "Enviar meu próprio CSV":
    arquivo = st.sidebar.file_uploader("Arquivo CSV", type=["csv"])
    if arquivo is None:
        st.sidebar.info("Envie um CSV para continuar, ou volte para a base incluída.")
        st.stop()
    try:
        df_bruto = carregar_base(arquivo, usar_upload=True)
    except Exception as e:
        st.sidebar.error(f"Erro ao ler o CSV: {e}")
        st.stop()
else:
    df_bruto = carregar_base(RAIZ / "dados" / "megasena.csv", usar_upload=False)

relatorio_val = validar_dados.validar(df_bruto)
df = validar_dados.limpar_com_seguranca(df_bruto)
n = len(df)

st.sidebar.metric("Concursos na base", n)
st.sidebar.metric("Período", f"{df['data'].min().date()} → {df['data'].max().date()}")
if not relatorio_val.ok:
    st.sidebar.warning(f"{len(relatorio_val.problemas)} problema(s) de validação encontrados — veja a aba 'Validação'.")

st.title("Mega-Sena — Painel de Análise Estatística")
st.info(AVISO)

abas = st.tabs([
    "Validação", "Frequência", "Perfil dos sorteios", "Testes de aleatoriedade",
    "Simulação Monte Carlo", "Modelos + validação temporal", "Gerador", "Fechamento",
])

dados = analise_completa(hash_df(df), df)

# ---------------------------------------------------------------- Validação
with abas[0]:
    st.subheader("Relatório de validação dos dados")
    st.text(relatorio_val.para_texto())

# ---------------------------------------------------------------- Frequência
with abas[1]:
    st.subheader("Frequência de cada dezena")
    freq_df = dados["freq_df"]
    fig, ax = plt.subplots(figsize=(12, 4))
    media = freq_df["esperado_sob_uniformidade"].iloc[0]
    cores = ["#2ecc71" if v >= media else "#95a5a6" for v in freq_df["frequencia_absoluta"]]
    ax.bar(freq_df["dezena"], freq_df["frequencia_absoluta"], color=cores)
    ax.axhline(media, color="#e74c3c", linestyle="--", label="Esperado sob uniformidade")
    ax.set_xlabel("Dezena"); ax.set_ylabel("Frequência"); ax.legend()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 mais frequentes**")
        st.dataframe(freq_df.sort_values("frequencia_absoluta", ascending=False).head(10), hide_index=True)
    with col2:
        st.markdown("**Top 10 menos frequentes**")
        st.dataframe(freq_df.sort_values("frequencia_absoluta").head(10), hide_index=True)

    st.markdown("**Atrasos (concursos desde a última ocorrência)**")
    st.dataframe(dados["atrasos_df"].sort_values("atraso_atual", ascending=False).head(15), hide_index=True)

    st.markdown("**Pares de dezenas mais frequentes**")
    st.dataframe(dados["pares_df"], hide_index=True)

# ---------------------------------------------------------------- Perfil
with abas[2]:
    st.subheader("Perfil estatístico dos sorteios")
    c1, c2, c3, c4 = st.columns(4)
    ss = dados["soma_stats"]
    c1.metric("Soma média", f"{ss['media']:.1f}")
    c2.metric("Soma mínima", ss["minimo"])
    c3.metric("Soma máxima", ss["maximo"])
    c4.metric("Repetição média vs. anterior", f"{np.nanmean(dados['rep1'].values):.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dados["somas"], bins=25, color="#3498db", edgecolor="white")
        ax.set_title("Distribuição da soma"); st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        pd_ = dados["paridade_df"]
        ax.bar(pd_["qtd_pares"], pd_["qtd_concursos"], color="#9b59b6")
        ax.set_title("Paridade (qtd. de números pares)"); st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(dados["faixas_df"]["faixa"], dados["faixas_df"]["quantidade"], color="#f39c12")
    ax.set_title("Distribuição por faixa de 10 dezenas"); st.pyplot(fig)

# ---------------------------------------------------------------- Testes
with abas[3]:
    st.subheader("Testes formais de aleatoriedade")

    for teste in [dados["teste_uniformidade"], dados["teste_runs"], dados["teste_indep"]]:
        with st.expander(teste["nome"]):
            st.write(f"**H0:** {teste.get('h0','—')}")
            st.write(f"**H1:** {teste.get('h1','—')}")
            if "valor_p" in teste:
                st.write(f"**Valor-p:** {teste['valor_p']:.4f}")
            st.write(teste.get("interpretacao", ""))

    st.markdown("**Teste por dezena com correção de Bonferroni (60 comparações simultâneas)**")
    qui2 = dados["qui2_por_dezena"]
    n_sig = int(qui2["significativo_apos_correcao_5pct"].sum())
    st.write(f"{n_sig} de 60 dezenas permaneceram significativas após a correção.")
    st.dataframe(qui2, hide_index=True)

    st.markdown("**Intervalo de confiança (95%) da frequência de cada dezena**")
    st.dataframe(dados["ic_df"], hide_index=True)

# ---------------------------------------------------------------- Monte Carlo
with abas[4]:
    st.subheader("Simulação Monte Carlo")
    st.caption(
        "Compara os dados reais com séries inteiramente aleatórias do mesmo tamanho. "
        "Reduzido para caber no tempo de resposta de um app web -- ajuste no código para "
        "rodar mais repetições localmente."
    )
    n_rep = st.slider("Número de repetições", 200, 5000, 1000, step=200)
    if st.button("Rodar simulação"):
        with st.spinner(f"Simulando {n_rep:,} séries aleatórias..."):
            resultado_mc = sim.rodar_monte_carlo(n_concursos=n, n_repeticoes=n_rep, tamanho_lote=100, semente=42)
            estatisticas_reais = {
                "freq_max": int(dados["freq_df"]["frequencia_absoluta"].max()),
                "freq_min": int(dados["freq_df"]["frequencia_absoluta"].min()),
                "soma_media": float(dados["somas"].mean()),
                "repeticao_media": float(np.nanmean(dados["rep1"].values)),
                "maior_atraso_medio": float(dados["atrasos_df"]["maior_atraso_historico"].mean()),
                "pct_consecutivo": float(100 * np.mean([
                    np.any(np.diff(np.sort(linha)) == 1) for linha in est.matriz_dezenas(df)
                ])),
            }
            comparacao = sim.comparar_real_vs_simulacao(estatisticas_reais, resultado_mc)
        st.dataframe(comparacao, hide_index=True)
        n_incomuns = int(comparacao["incomum_5pct"].sum())
        st.write(f"**{n_incomuns} de {len(comparacao)}** métricas caíram fora da faixa comum (percentil 2,5%-97,5%).")

# ---------------------------------------------------------------- Modelos
with abas[5]:
    st.subheader("Modelos experimentais + validação temporal (walk-forward)")
    st.caption("Cada modelo só usa concursos anteriores ao que está sendo avaliado — nunca dados futuros.")
    modelos_disponiveis = list(mdl.REGISTRO_MODELOS.keys())
    modelos_disponiveis.remove("regressao_logistica")  # custoso demais para rodar ao vivo
    escolhidos = st.multiselect("Modelos a comparar", modelos_disponiveis,
                                 default=["aleatorio", "frequencia_acumulada", "janela_movel", "tempo_desde_ultima"])
    inicio_minimo = st.slider("Início mínimo do walk-forward (concursos)", 20, max(21, n - 1), min(60, n - 1))
    if st.button("Rodar validação temporal") and escolhidos:
        with st.spinner("Rodando walk-forward..."):
            resultados_wf = vt.walk_forward(df, escolhidos, inicio_minimo=inicio_minimo, passo=1, semente=42)
            corte_treino, corte_val = vt.dividir_treino_validacao_teste(n)
            resumo = vt.resumir_desempenho(resultados_wf, corte_treino, corte_val)
        st.dataframe(resumo, hide_index=True)

        if "aleatorio" in escolhidos:
            st.markdown("**Comparação contra o modelo aleatório (Wilcoxon pareado)**")
            for nome in escolhidos:
                if nome == "aleatorio":
                    continue
                comp = vt.comparar_modelo_vs_aleatorio(resultados_wf, nome, periodo_mask=lambda idx: idx >= corte_val)
                if "erro" in comp:
                    continue
                st.write(
                    f"- **{nome}**: média {comp['media_acertos_modelo']:.3f} vs. aleatório "
                    f"{comp['media_acertos_aleatorio']:.3f} (p={comp['valor_p']:.4f}, n={comp['n_pares']})"
                )
        st.warning(
            "Lembrete: mesmo diferenças com p<0,05 podem ser explicadas por múltiplas "
            "comparações (vários modelos testados) ou por acaso -- não é evidência de que "
            "o modelo 'funciona' para apostas futuras."
        )

# ---------------------------------------------------------------- Gerador
with abas[6]:
    st.subheader("Gerador de combinações (simulação)")
    col1, col2, col3 = st.columns(3)
    qtd = col1.number_input("Quantidade de combinações", 1, 500, 5)
    preco = col2.number_input("Preço da aposta simples (R$)", 0.5, 100.0, 5.0, step=0.5)
    orcamento = col3.number_input("Orçamento máximo (R$, opcional)", 0.0, 100000.0, 0.0, step=10.0)
    usar_filtro_soma = st.checkbox("Filtrar por faixa de soma (120 a 250)")
    usar_filtro_paridade = st.checkbox("Filtrar por paridade (2 a 4 números pares)")

    if st.button("Gerar combinações"):
        resultado = gc.gerar_combinacoes(
            quantidade_desejada=int(qtd), preco_aposta=float(preco),
            orcamento_maximo=float(orcamento) if orcamento > 0 else None,
            filtro_soma=(120, 250) if usar_filtro_soma else None,
            filtro_paridade=(2, 4) if usar_filtro_paridade else None,
            semente=int(np.random.randint(0, 1_000_000)),
        )
        st.write(f"**{resultado.quantidade}** combinações geradas · custo total **R$ {resultado.custo_total:.2f}** · "
                 f"probabilidade total **{resultado.probabilidade_total:.8f}** "
                 f"(1 em {round(1/resultado.probabilidade_total):,})")

        ld = df.iloc[-1]
        for i, jogo in enumerate(resultado.combinacoes, start=1):
            jogo_ordenado = sorted(jogo)
            c_bolas, c_btn = st.columns([5, 2])
            with c_bolas:
                st.markdown(f"**Jogo {i}**" if resultado.quantidade > 1 else "**Jogo gerado**")
                st.markdown(renderizar_bolas(jogo_ordenado), unsafe_allow_html=True)
            with c_btn:
                texto = texto_whatsapp_jogos([jogo_ordenado], int(ld["concurso"]), ld["data"].date(), n)
                st.link_button("📲 Compartilhar", link_whatsapp(texto), use_container_width=True)

        if resultado.quantidade > 1:
            texto_todos = texto_whatsapp_jogos(
                [sorted(j) for j in resultado.combinacoes], int(ld["concurso"]), ld["data"].date(), n
            )
            st.link_button("📲 Compartilhar todos os jogos no WhatsApp", link_whatsapp(texto_todos))

        st.warning(resultado.aviso)
        with st.expander("Ver como tabela"):
            st.dataframe(pd.DataFrame(resultado.combinacoes, columns=[f"n{i}" for i in range(1, 7)]), hide_index=True)

    st.divider()
    st.markdown("**📲 Já tem um jogo escolhido? Compartilhe direto:**")
    manual = st.text_input("Digite os 6 números (separados por espaço ou vírgula)", key="manual_mega")
    if manual:
        nums = sorted(set(int(x) for x in "".join(c if c.isdigit() else " " for c in manual).split()))
        nums = [n for n in nums if 1 <= n <= 60]
        if len(nums) >= 6:
            st.markdown(renderizar_bolas(nums), unsafe_allow_html=True)
            ld = df.iloc[-1]
            texto = texto_whatsapp_jogos([nums], int(ld["concurso"]), ld["data"].date(), n)
            st.link_button("📲 Compartilhar no WhatsApp", link_whatsapp(texto))
        else:
            st.caption("Digite pelo menos 6 números válidos entre 1 e 60.")

# ---------------------------------------------------------------- Fechamento
with abas[7]:
    st.subheader("Sistema de fechamento completo (wheel)")
    dezenas_txt = st.text_input("Dezenas escolhidas (mín. 6, separadas por vírgula)", "1,2,3,4,5,6,7,8")
    preco_f = st.number_input("Preço da aposta simples (R$)", 0.5, 100.0, 5.0, step=0.5, key="preco_fechamento")
    if st.button("Gerar fechamento"):
        try:
            dezenas = [int(x.strip()) for x in dezenas_txt.split(",") if x.strip()]
            resultado = fch.gerar_fechamento_completo(dezenas, preco_aposta=float(preco_f))
            st.write(f"**{resultado.quantidade_apostas}** apostas geradas a partir de "
                     f"{resultado.quantidade_dezenas} dezenas · custo total **R$ {resultado.custo_total:,.2f}**")
            st.info(resultado.condicoes_de_garantia)
            st.warning(resultado.o_que_nao_e_garantido)
            st.warning(resultado.risco_divisao_premio)
            if resultado.quantidade_apostas <= 500:
                st.dataframe(pd.DataFrame(resultado.apostas, columns=[f"n{i}" for i in range(1, 7)]), hide_index=True)
            else:
                st.caption(f"{resultado.quantidade_apostas} apostas — mostrando apenas as 500 primeiras.")
                st.dataframe(pd.DataFrame(resultado.apostas[:500], columns=[f"n{i}" for i in range(1, 7)]), hide_index=True)
        except ValueError as e:
            st.error(str(e))

st.divider()
st.caption(
    "Código-fonte e relatório técnico completo disponíveis no repositório do projeto. "
    "Esta ferramenta não garante acertos, lucro ou qualquer vantagem sobre apostas aleatórias."
)
