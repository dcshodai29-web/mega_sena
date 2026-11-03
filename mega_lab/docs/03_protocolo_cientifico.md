# 03 — Protocolo científico (pré-registro)

Este documento foi escrito **antes** de rodar qualquer teste sobre os dados e
fixa hipóteses, estatísticas, períodos, parâmetros e regras de decisão.
Mudanças posteriores devem ser feitas em commit separado, com justificativa,
e nunca podem reclassificar um resultado já observado.

## Regra fundamental

> Nenhuma suposta descoberta é aceita sem demonstrar que funciona em
> concursos que não participaram da sua formulação, e sem superar uma linha de
> base aleatória com significância estatística.

## Hipótese nula (H0)

Cada concurso é uma amostra uniforme, sem reposição, de 6 dezenas entre 60,
independente de todos os outros concursos e de qualquer fator externo
(data, dia da semana, prêmio acumulado, local, equipamento). Em particular, a
ordem de extração é uma permutação uniforme das 6 dezenas.

## Períodos

| Uso | Concursos | Observação |
|---|---|---|
| Descoberta (D) | 1 – 2000 | onde as hipóteses são avaliadas primeiro |
| Confirmação (C) | 2001 – último verificado | usado **uma vez** para confirmar o que passou em D |
| Backtest: aquecimento | 1 – 100 | modelos só acumulam histórico |
| Backtest: desenvolvimento | 101 – 2000 | |
| Backtest: validação | 2001 – 2600 | |
| Backtest: holdout selado | 2601 – último verificado | período de decisão dos modelos |
| Prospectivo | concursos posteriores ao primeiro registro no livro-razão | únicos dados realmente desconhecidos por qualquer pessoa ou IA no momento da previsão |

Nota: um modelo de linguagem pode ter visto resultados antigos durante o
treinamento. Por isso o **teste definitivo é o prospectivo**: previsões
registradas e publicadas (commit) antes do sorteio.

Nível de significância: α = 0,05 em todos os testes.

## Família 1 — Testes de aleatoriedade e anomalias (H01–H15)

Todas as estatísticas têm p-valor por **Monte Carlo** sob H0 (B = 1.000
históricos simulados do mesmo tamanho, `p = (1 + #{T_sim ≥ T_obs}) / (B + 1)`),
o que dispensa aproximações assintóticas e incorpora a dependência causada
pela extração sem reposição. Testes por fator usam **permutação** dos rótulos
(B = 1.000).

| ID | Hipótese alternativa | Estatística |
|---|---|---|
| H01 | Dezenas não uniformes | χ² de Pearson das 60 frequências |
| H02 | Alguma dezena individual anômala (bola "viciada") | max_i \|z_i\| |
| H03 | Pares com co-ocorrência anômala | χ² das 1.770 contagens de pares |
| H04 | Trios com co-ocorrência anômala | contagem máxima entre 34.220 trios |
| H05 | Repetição excessiva de combinações | nº de pares de concursos que compartilham ≥ 4 dezenas (milhões de comparações) |
| H06 | Dependência do concurso anterior | χ² da distribuição de dezenas repetidas do concurso anterior (0, 1, 2, ≥3) |
| H07 | Transições dezena→dezena entre concursos consecutivos | χ² da matriz 60×60 de transições |
| H08 | Intervalos (atrasos) não geométricos | χ² dos intervalos agrupados (1..30, >30) contra Geom(0,1) |
| H09 | Memória na soma das dezenas | Ljung-Box Q, defasagens 1–10 |
| H10a | Paridade anômala | χ² do nº de pares por concurso (0..6) |
| H10b | Baixas/altas anômalas | χ² do nº de dezenas 1–30 por concurso |
| H10c | Faixas (décadas) anômalas | χ² das contagens por faixa 1–10, …, 51–60 |
| H10d | Dígito final anômalo | χ² das contagens por dígito final 0–9 |
| H11 | Viés de posição na ordem de extração (física das bolas) | χ² da tabela dezena × posição (60×6) |
| H12 | Excesso/falta de números consecutivos | χ² do nº de pares consecutivos por concurso |
| H13 | Mudança de regime (equipamento/procedimento) | máximo, sobre pontos de corte, do χ² de homogeneidade antes × depois |
| H14 | Deriva lenta | variância dos χ² em janelas de 200 concursos |
| H15a | Dia da semana influencia | χ² de homogeneidade dia × dezena (permutação) |
| H15b | Mega da Virada difere | idem, especial × regular |
| H15c | Prêmio acumulado influencia | idem, concurso acumulado × não acumulado |
| H15d | Frequência semanal de sorteios (regime de calendário) influencia | idem, regime × dezena |

**Regra de decisão.** Correção de Holm sobre as 21 hipóteses no período D.
Uma hipótese só vira **anomalia confirmada** se: (1) p_Holm < 0,05 em D **e**
(2) p < 0,05 em C com a mesma estatística. Qualquer outro caso é reportado
como "não confirmado", com os p-valores.

**Calibração e poder.** O relatório inclui (a) quantos "padrões" a análise
ingênua encontra em históricos perfeitamente justos e (b) a curva de poder
para detectar uma única bola com peso relativo w ∈ {1,05; 1,10; 1,15; 1,20; 1,30}
com o tamanho real da base — isto é, o tamanho mínimo de viés físico que a base
seria capaz de revelar.

## Família 2 — Previsibilidade (H16)

Modelos pré-fixados (nenhum hiperparâmetro é ajustado olhando desempenho):

| ID | Modelo | Ideia |
|---|---|---|
| M00 | aleatorio | linha de base: uniforme |
| M01 | freq_total | frequência acumulada |
| M02 | freq_janela100 | frequência nos últimos 100 |
| M03 | ewma_hl50 | frequência exponencial, meia-vida 50 |
| M04 | quentes20 | frequência nos últimos 20 |
| M05 | frios | inverso da frequência acumulada |
| M06 | atrasados | proporcional ao atraso atual |
| M07 | dirichlet | média a posteriori Dirichlet-multinomial, prior 50 por dezena |
| M08 | markov1 | transições dezena→dezena do concurso anterior |
| M09 | afinidade | co-ocorrência histórica com as dezenas do concurso anterior |
| M10 | logistica | regressão logística sobre features causais, re-treino a cada 100 |
| M11 | gboost | gradient boosting sobre as mesmas features, re-treino a cada 100 |
| M12 | hedge | mistura dos modelos M01–M11 com pesos exponenciais pelo log-score passado ("aprende com os erros") |

Probabilidades dos modelos heurísticos (M01–M09): `p ∝ score`, normalizado
para somar 6, encolhido 50 % em direção a 0,1. Bilhete = 6 maiores `p`.

**Métricas:** acertos do bilhete; log-score de conjunto em bits contra o
uniforme; e-valor com mistura 50 % uniforme.

**Regra de decisão.** Um modelo só é declarado "melhor que o acaso" se:
(1) no holdout selado, p_Holm < 0,05 (família de 24 testes: 12 modelos ×
{acertos, bits}, unilateral); **e** (2) prospectivamente, seu e-valor
acumulado atingir 12/0,05 = 240 (correção de Bonferroni para e-valores
com 12 modelos). A condição (2) é válida a qualquer momento (desigualdade de
Ville), então o livro-razão pode ser consultado após cada concurso sem
inflar o erro tipo I.

## O que contaria como descoberta real

Uma anomalia confirmada (Família 1) **e** um modelo que a explore com
desempenho superior ao acaso no holdout e no prospectivo (Família 2). Uma
anomalia estatística sem ganho preditivo é registrada como curiosidade, não
como padrão preditivo.

## Adendo 1 (2026-11-03) — nova hipótese gerada pelos dados, só para teste prospectivo

**Contexto.** Rodada a Família 1 (1.000 réplicas): nenhuma hipótese passou
na correção de Holm no período de descoberta (D = 1–2000), logo **nenhuma
anomalia foi confirmada**. Porém, no período de confirmação (C = 2001–3080),
que pelo protocolo só serviria para confirmar candidatas de D, apareceram
p-valores baixos em H01 (uniformidade, p = 0,006), H03 (pares, p = 0,008) e
H07 (transições, p = 0,012) — três estatísticas que reagem à mesma causa:
frequências desiguais entre dezenas. Com 21 testes em C, isso não é
significativo após correção (Holm: p ≈ 0,13) e não pode ser declarado
descoberta. É, no entanto, uma hipótese legítima **gerada pelos dados**.

**H17 (pré-registrada agora, antes de ver o resultado do backtest).**
"As dezenas não têm a mesma probabilidade de sair; a frequência acumulada
carrega informação sobre o próximo sorteio."

- Dados de teste: **apenas concursos ≥ 3081** (ainda não sorteados neste
  registro), via livro-razão.
- Teste primário: e-valor prospectivo do modelo `freq_total` (escolhido por
  ser a codificação mais direta da hipótese), limiar e ≥ 20 (α = 0,05,
  hipótese única, válido a qualquer momento).
- Teste secundário, horizonte fixo: χ² de uniformidade (H01, Monte Carlo)
  nos primeiros 500 concursos prospectivos (3081–3580).
- Nenhuma outra variante (janela, modelo, dezena específica) será usada para
  decidir H17.
