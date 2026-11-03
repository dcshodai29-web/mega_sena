# 01 — Auditoria dos repositórios de referência

Data da auditoria: 2026-11-03. Todos os repositórios foram lidos a partir dos
forks em `dcshodai29-web/*`, idênticos aos originais no momento do fork.

Objetivo: decidir **o que aproveitar** (ideias, dados, código) e **o que não
aproveitar** para o MEGA LAB, com base em qualidade de código, licença,
fontes de dados e limitações. Nenhum código destes projetos foi copiado para o
MEGA LAB: o código do laboratório foi escrito do zero. Dos repositórios
aproveitamos apenas **dados factuais** (números sorteados e datas), sempre
cruzados com outras fontes (ver `04_fontes_de_dados.md`).

## Resumo executivo

| Repositório (original) | Linguagem | Licença | Dados | Último concurso | Testes | Veredito |
|---|---|---|---|---|---|---|
| `OsJunnior/mega_sena` | Python | **nenhuma** | CSV, 506 concursos (2551–3056) | 3056 (10/09/2026) | 18 (pytest, passam) | Boa intenção metodológica; base pequena demais; 2 defeitos nos modelos |
| `salles97/MegaSena` | Python + HTML | **nenhuma** | JSON da API da Caixa, ordenado | 3016 (09/06/2026) | nenhum | Extrator útil como referência de esquema da API; análise apenas descritiva |
| `luizvalle/mega_sena_analysis` | Python (notebook) | MIT | Planilha oficial da Caixa + CSV asloterias + premiação | 2964 (24/01/2026) | nenhum | Melhor fonte de dados (planilha oficial com premiação); análise GLRT correta mas limitada |
| `McGuyverBR/loteria` (fork de `damarals/loteria`) | R | MIT | CSV raspado de megaloterias.com.br | 2458 (26/02/2022) | testthat | Desatualizado desde 2022; fonte independente útil para conferência |
| `RoninXK/megadata-megasena` *(5º repositório anexado à sessão)* | Python + Streamlit | **nenhuma** | CSV com **ordem de extração** (asloterias) | 3066 (03/10/2026) | nenhum | Única fonte local com ordem de sorteio; backtest sem vazamento, mas sem teste de significância |

## 1. `OsJunnior/mega_sena`

**Pontos fortes**
- Postura estatística honesta: README e relatório declaram explicitamente que o
  sistema não é um previsor; linha de base aleatória obrigatória; split
  cronológico treino/validação/teste; walk-forward que só usa concursos
  anteriores (`src/validacao_temporal.py`).
- Testes automatizados (18, todos passam em ~1 s).
- Testes clássicos implementados: qui-quadrado global e por dezena com
  Bonferroni, runs test de paridade, autocorrelação da soma, Monte Carlo.

**Defeitos encontrados**
1. **`modelo_frequencia_pares` é idêntico a 5 × frequência simples.** O laço
   `for i in range(6): for j in range(6): if i != j: pair_score[linha[i]-1] += 1`
   soma 5 para cada dezena sorteada, independentemente do parceiro. O modelo
   rotulado "pares" é na verdade o modelo de frequência acumulada — o
   comparativo de modelos conta o mesmo modelo duas vezes.
2. **Viés no desempate.** O palpite é `np.argsort(scores)[-6:]`. Quando há
   empates (comuns em janelas curtas, ex.: janela de 50 concursos), o
   `argsort` estável favorece sistematicamente as **dezenas mais altas**. Não
   cria vantagem preditiva, mas distorce o que o modelo "escolhe".
3. **Base muito pequena:** 506 concursos (2023–2026), ~16 % do histórico. O
   poder estatístico para detectar vieses de bola é baixo (ver análise de poder
   no relatório do MEGA LAB).
4. Pasta `mega_sena/` duplicada dentro do repositório (upload repetido;
   conteúdo idêntico ao da raiz).
5. Wilcoxon pareado sobre acertos (variável discreta 0–6 com muitos empates) é
   um teste fraco aqui; a distribuição nula dos acertos é conhecida
   exatamente (hipergeométrica) e deveria ser usada.
6. Sem licença: o código não pode ser reutilizado legalmente sem autorização.

## 2. `salles97/MegaSena`

**Pontos fortes**
- `megasena_extractor.py` consome a API oficial
  `servicebus2.caixa.gov.br/portaldeloterias/api/megasena/{n}` com retentativas
  e atualização incremental em SQLite. Serviu de referência para o **esquema**
  da API (campos `numero`, `dataApuracao`, `listaDezenas`,
  `dezenasSorteadasOrdemSorteio`, `listaRateioPremio`, `acumulado`).

**Defeitos / limitações**
1. **Descarta a ordem de extração:** lê `dezenasSorteadasOrdemSorteio` e em
   seguida ordena (`sorted(dezenas)`), perdendo a única informação que permite
   estudar viés de posição/física das bolas.
2. **Campo `acumulado_val` errado:** usa `valorArrecadado` (arrecadação) como
   se fosse valor acumulado.
3. Análise apenas descritiva (frequências, duplas, trios), sem nenhum teste
   contra o acaso — "duplas mais frequentes" sem correção para ~1.770
   comparações múltiplas é ruído esperado.
4. Arquivo com nome `megasena_export_json.py.py`; sem `requirements.txt`; sem
   testes; sem licença.

## 3. `luizvalle/mega_sena_analysis`

**Pontos fortes**
- `data/Mega-Sena.xlsx` é a **planilha oficial exportada do site da Caixa**
  (comentário no notebook: "Downloaded from loterias.caixa.gov.br"), com 20
  colunas: ganhadores e rateio por faixa, acumulado, arrecadação, estimativa,
  cidade/UF dos ganhadores. Concursos 1–2964, sem lacunas.
- Modelo de probabilidade correto (p = C(59,5)/C(60,6) = 0,1) e GLRT por
  dezena com discussão de testes múltiplos.
- Raspador assíncrono bem escrito (retentativas, tipagem, validação de
  consistência ganhadores × prêmio). Licença MIT.

**Limitações**
- Dados em ordem crescente (sem ordem de extração).
- Notebook depende de download dinâmico (POST em asloterias.com.br): não é
  reprodutível sem rede e sem o site estar no ar.
- `requirements.txt` com versões fixas incluindo pacotes de macOS (`appnope`).
- Não há backtesting nem modelo preditivo.

## 4. `McGuyverBR/loteria` (fork de `damarals/loteria`)

**Pontos fortes**
- Pacote R bem estruturado (roxygen, testthat, CI, licença MIT), com
  atualização diária via GitHub Actions — **parada desde 28/02/2022**.

**Limitações**
- Raspa HTML de um site terceiro (`megaloterias.com.br`) por XPath: frágil e
  não oficial.
- Dados ordenados; concursos 1–2458 apenas.
- Útil para o MEGA LAB somente como **fonte independente de conferência**
  dos concursos antigos.

## 5. `RoninXK/megadata-megasena` (anexado à sessão, não listado no pedido)

**Pontos fortes**
- `data/processed/mega_sena_tratada.csv` preserva a **ordem de extração**
  das bolas (concursos 1–3066). Conferido contra uma segunda fonte
  independente (`guilhermeasn/loteria.json`): **2.797 de 2.797 concursos com
  ordem idêntica**.
- Backtest sem vazamento (`historico = df.iloc[:indice]`).

**Limitações**
- Arquivo bruto (`data/raw/*.xlsx`) está no `.gitignore` → o tratamento não é
  reprodutível a partir do repositório.
- Backtest compara médias de acertos (0,595 × 0,595 × 0,602) sem nenhum teste
  de significância; a diferença observada é compatível com o acaso.
- Sem testes, sem licença.

## O que o MEGA LAB aproveita e o que corrige

| Aproveitado | Como |
|---|---|
| Planilha oficial da Caixa (1–2964) | Fonte primária de números, datas e premiação |
| Ordem de extração (megadata + guilhermeasn) | Testes de viés por posição/bola |
| Esquema da API da Caixa (salles97) | Atualizador próprio, testado com fixture |
| Ideia de walk-forward e linha de base aleatória (OsJunnior) | Reimplementado com nula exata, e-valores e testes anti-vazamento |

| Problema encontrado | Correção no MEGA LAB |
|---|---|
| Bases parciais e de fonte única | Base reconciliada por votação entre 9 extratos, com status por concurso |
| Ordem de extração descartada | Mantida (`o1..o6`) e conferida entre duas fontes |
| Modelos duplicados / desempate enviesado | Interface única de modelo; desempate aleatório com semente |
| Comparações sem teste ou com teste fraco | Nula exata hipergeométrica, Monte Carlo condicional, Holm/BH, e-valores |
| Padrões "descobertos" no mesmo dado em que foram procurados | Hipóteses pré-registradas, conjunto de descoberta × holdout selado × registro prospectivo |
