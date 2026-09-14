# Relatório Técnico — Análise Estatística e Simulação da Mega-Sena

> **Aviso obrigatório de honestidade estatística:** este relatório NÃO promete acertos, lucro ou sucesso garantido. A Mega-Sena é, até prova em contrário, um processo de sorteio independente e aleatório. Nada aqui constitui aconselhamento financeiro ou uma fórmula para vencer a loteria.

## 1. Fonte e período dos dados

Arquivo usado: `dados/megasena.csv`, com **506 concursos reais**, do concurso 2551 (2023-01-04) ao concurso 3056 (2026-09-10). **Limitação importante:** este arquivo cobre um subconjunto do histórico completo da Mega-Sena (que começou em 1996) -- é a janela de dados reais disponível nesta execução, coletada de fontes públicas (resultados oficiais replicados por sites de resultado de loteria) e com datas conferidas. Para uma análise sobre a história completa do jogo, substitua `dados/megasena.csv` por um CSV com todos os concursos desde 1996, mantendo o mesmo formato de colunas -- todo o pipeline funciona sem alterações.

## 2. Validação dos dados

```
# Relatório de validação dos dados

Total de linhas lidas: 506
Status geral: OK (sem problemas bloqueantes)

## Detalhes
- Linhas duplicadas (idênticas): 0
- Concursos duplicados (mesmo número, dados diferentes ou não): []
- Concursos faltantes na sequência: 0 (exemplos: [])
- Valores ausentes por coluna: {}
- Linhas com dezena fora de 1-60: 0
- Linhas com dezena repetida no mesmo concurso: 0
- Linhas com data inválida: 0
- Fora de ordem cronológica (precisou reordenar): False
```

## 3. Fundamentos matemáticos

- Combinações possíveis (C(60,6)): **50,063,860**
- Probabilidade de uma aposta simples acertar a sena: **1 / 50,063,860** ≈ 0.0000000200
- Probabilidade de uma dezena específica sair em um concurso: 6/60 = 0,10
- Frequência esperada de uma dezena em N=506 concursos: 50.6 (desvio-padrão ≈ 6.75)

**Probabilidade de sair exatamente k números pares (30 pares e 30 ímpares em 60):**

| k pares | P(K=k) |
|---|---|
| 0 | 1.1860% |
| 1 | 8.5395% |
| 2 | 23.8119% |
| 3 | 32.9251% |
| 4 | 23.8119% |
| 5 | 8.5395% |
| 6 | 1.1860% |

Note que certas contagens de paridade são mais frequentes simplesmente porque existem mais combinações possíveis com aquela estrutura (ex: 3 pares e 3 ímpares é a mais numerosa), e não porque o sorteio 'prefere' esse padrão.

## 4. Análise descritiva

Gráficos gerados na pasta `graficos/` e tabelas completas em `resultados/`. Destaques:

- Dezena mais sorteada: **46** (65 vezes)
- Dezena menos sorteada: **29** (36 vezes)
- Soma média das 6 dezenas: 183.6 (mediana 185, mín 63, máx 311)
- Repetição média de dezenas vs. concurso anterior: 0.59 (de 6 possíveis)

## 5. Testes de aleatoriedade e independência

### Qui-quadrado de uniformidade (frequência das 60 dezenas)
- H0: Todas as dezenas têm a mesma probabilidade de serem sorteadas (1/60 cada).
- H1: Ao menos uma dezena tem probabilidade diferente de 1/60.
- Estatística: 49.2569
- Valor-p: 0.8131
- Tamanho de efeito (Cramér's V): 0.1274
- Interpretação: Valor-p baixo indicaria desvio da uniformidade nas frequências observadas. Mesmo que isso ocorra, não implica que uma dezena específica seja 'mais provável' no PRÓXIMO sorteio -- sorteios futuros continuam sendo eventos novos e independentes do histórico, a menos que exista evidência de um mecanismo físico de viés (o que este teste sozinho não pode demonstrar).
- Limitações: Teste sensível ao tamanho da amostra: com poucos concursos, tem pouco poder para detectar desvios pequenos; com muitos concursos, pode rejeitar H0 por desvios triviais sem relevância prática -- por isso reportamos também o tamanho de efeito (Cramér's V), não só o valor-p.

### Runs test sobre paridade da soma dos 6 números
- H0: A alternância entre concursos de soma par e soma ímpar é consistente com uma sequência aleatória.
- H1: Há dependência sequencial (sequências mais longas ou mais curtas do que o esperado ao acaso).
- Valor-p: 0.8187
- Interpretação: Este teste avalia se um concurso 'depende' do padrão de paridade dos concursos vizinhos. Não testa dezenas específicas e não implica capacidade preditiva mesmo se significativo.

### Independência entre concursos consecutivos (repetição de dezenas)
- H0: O número de dezenas repetidas entre concursos consecutivos segue a distribuição hipergeométrica esperada sob sorteios independentes.
- H1: A distribuição observada de repetições difere da esperada sob independência.
- Valor-p: 0.9988
- Interpretação: Se o valor-p for baixo, a quantidade de dezenas repetidas entre sorteios consecutivos foge do esperado sob independência -- mas isso ainda não indica QUAIS dezenas tendem a se repetir nem fornece uma regra útil de aposta.

**Teste por dezena com correção de Bonferroni (60 testes simultâneos):** 0 de 60 dezenas permaneceram estatisticamente significativas após a correção (tabela completa em `resultados/qui2_por_dezena.csv`).

## 6. Simulação Monte Carlo

Foram simuladas **20,000 séries** inteiramente aleatórias de 506 concursos cada (mesmo tamanho da base real), para servir de referência sobre o que é 'normal' esperar de puro acaso.

**Nota sobre a quantidade de repetições:** o pedido original sugeria 100.000 repetições. Cada repetição envolve cálculos O(n) por série (atraso máximo de 60 dezenas, repetição entre concursos consecutivos, sequências consecutivas), o que tornou 100.000 repetições computacionalmente custoso demais para esta execução. Foram rodadas **20,000** repetições, com semente fixa (42) para reprodutibilidade -- volume suficiente para estimar percentis com boa precisão para este diagnóstico, mas o número pode ser aumentado no código (`simulacao.rodar_monte_carlo`) se houver mais tempo de execução disponível.

| metrica                               |   valor_real |   media_simulada |   desvio_padrao_simulado |   percentil_do_valor_real | incomum_5pct   |
|:--------------------------------------|-------------:|-----------------:|-------------------------:|--------------------------:|:---------------|
| Frequência máxima de uma dezena       |    65        |        67.0096   |                3.29714   |                   0.22945 | False          |
| Frequência mínima de uma dezena       |    36        |        35.512    |                2.68104   |                   0.4553  | False          |
| Soma média dos concursos              |   183.591    |       183.016    |                1.8058    |                   0.6261  | False          |
| Repetição média vs. concurso anterior |     0.590099 |         0.600316 |                0.0312259 |                   0.3602  | False          |
| Maior atraso médio entre dezenas      |    42.7333   |        42.8751   |                1.35972   |                   0.468   | False          |
| % de concursos com consecutivos       |    39.5257   |        42.1255   |                2.18698   |                   0.11015 | False          |

**0 de 6** métricas comparadas caíram fora da faixa considerada comum (percentil 2,5%-97,5%) frente às simulações aleatórias.

## 7. Modelos experimentais e validação temporal

Todos os modelos foram avaliados com validação **walk-forward**: em cada passo, o modelo só usa concursos anteriores ao que está sendo avaliado. Nenhuma informação futura foi usada em nenhum momento.

| modelo               | periodo     |    media |   mediana |   desvio_padrao |   n |   pct_4_ou_mais |   pct_5_ou_mais |   pct_6 |
|:---------------------|:------------|---------:|----------:|----------------:|----:|----------------:|----------------:|--------:|
| aleatorio            | teste_final | 0.5      |       0   |        0.685782 | 102 |        0.980392 |               0 |       0 |
| aleatorio            | treino      | 0.539474 |       0   |        0.717627 | 152 |        0.657895 |               0 |       0 |
| aleatorio            | validacao   | 0.613861 |       1   |        0.720698 | 101 |        0        |               0 |       0 |
| beta_binomial        | teste_final | 0.637255 |       1   |        0.656863 | 102 |        0        |               0 |       0 |
| beta_binomial        | treino      | 0.677632 |       1   |        0.742307 | 152 |        0        |               0 |       0 |
| beta_binomial        | validacao   | 0.554455 |       0   |        0.624103 | 101 |        0        |               0 |       0 |
| combinado            | teste_final | 0.637255 |       1   |        0.700625 | 102 |        0        |               0 |       0 |
| combinado            | treino      | 0.631579 |       0.5 |        0.74304  | 152 |        0        |               0 |       0 |
| combinado            | validacao   | 0.50495  |       0   |        0.687368 | 101 |        0        |               0 |       0 |
| frequencia_acumulada | teste_final | 0.637255 |       1   |        0.656863 | 102 |        0        |               0 |       0 |
| frequencia_acumulada | treino      | 0.677632 |       1   |        0.742307 | 152 |        0        |               0 |       0 |
| frequencia_acumulada | validacao   | 0.554455 |       0   |        0.624103 | 101 |        0        |               0 |       0 |
| frequencia_pares     | teste_final | 0.637255 |       1   |        0.656863 | 102 |        0        |               0 |       0 |
| frequencia_pares     | treino      | 0.677632 |       1   |        0.742307 | 152 |        0        |               0 |       0 |
| frequencia_pares     | validacao   | 0.554455 |       0   |        0.624103 | 101 |        0        |               0 |       0 |
| frequencia_ponderada | teste_final | 0.598039 |       0.5 |        0.693103 | 102 |        0        |               0 |       0 |
| frequencia_ponderada | treino      | 0.638158 |       0   |        0.750711 | 152 |        0        |               0 |       0 |
| frequencia_ponderada | validacao   | 0.455446 |       0   |        0.608683 | 101 |        0        |               0 |       0 |
| janela_movel         | teste_final | 0.705882 |       1   |        0.683443 | 102 |        0        |               0 |       0 |
| janela_movel         | treino      | 0.638158 |       1   |        0.732856 | 152 |        0        |               0 |       0 |
| janela_movel         | validacao   | 0.514851 |       0   |        0.642088 | 101 |        0        |               0 |       0 |
| tempo_desde_ultima   | teste_final | 0.676471 |       1   |        0.733387 | 102 |        0        |               0 |       0 |
| tempo_desde_ultima   | treino      | 0.585526 |       0   |        0.722799 | 152 |        0        |               0 |       0 |
| tempo_desde_ultima   | validacao   | 0.564356 |       0   |        0.712963 | 101 |        0        |               0 |       0 |

### Comparação de cada modelo contra a linha de base aleatória (teste final)

- **frequencia_acumulada**: média 0.637 vs. aleatório 0.500 (diferença +0.137, valor-p Wilcoxon = 0.1301, n=102)
- **janela_movel**: média 0.706 vs. aleatório 0.500 (diferença +0.206, valor-p Wilcoxon = 0.0260, n=102)
- **frequencia_ponderada**: média 0.598 vs. aleatório 0.500 (diferença +0.098, valor-p Wilcoxon = 0.3278, n=102)
- **tempo_desde_ultima**: média 0.676 vs. aleatório 0.500 (diferença +0.176, valor-p Wilcoxon = 0.0564, n=102)
- **frequencia_pares**: média 0.637 vs. aleatório 0.500 (diferença +0.137, valor-p Wilcoxon = 0.1301, n=102)
- **beta_binomial**: média 0.637 vs. aleatório 0.500 (diferença +0.137, valor-p Wilcoxon = 0.1301, n=102)
- **combinado**: média 0.637 vs. aleatório 0.500 (diferença +0.137, valor-p Wilcoxon = 0.1314, n=102)

**Sobre eventos raros:** acertos de 5 ou 6 números são extremamente raros mesmo em 355 avaliações walk-forward; a tabela de distribuição de acertos (`resultados/distribuicao_acertos.csv`) mostra a contagem exata, e conclusões sobre essas faixas específicas devem ser tratadas com cautela adicional por causa do tamanho pequeno da amostra nessas faixas.

## 8. Modelo bayesiano Beta-Binomial

Prior utilizado: Beta(1,1) — uniforme, não-informativo (nenhuma dezena é presumida mais provável que outra antes de olhar os dados). A tabela completa com média posterior e intervalo de credibilidade de 95% para cada dezena está em `resultados/beta_binomial.csv`.

Uma média posterior mais alta reflete apenas uma taxa histórica de longo prazo suavizada — não uma previsão de que aquela dezena sairá no próximo concurso.

## 9. Gerador de combinações e sistemas de fechamento

Demonstração: geradas 10 combinações distintas (preço de exemplo R$ 5.00/aposta simples -- ajuste este valor conforme o preço vigente, que muda ao longo do tempo). Custo total de exemplo: R$ 50.00. Probabilidade total de acerto da sena com essas 10 apostas: 0.00000020 (1 em 5,006,386). Estas combinações são para fins de simulação/organização de apostas. Um ranking experimental usado no modo 'ranking' influencia apenas a PRIORIDADE de inclusão de dezenas -- não garante nem aumenta a probabilidade real de acerto de cada combinação, que permanece 1/50063860 para qualquer combinação, gerada por qualquer método. Foram excluídas 1 combinações candidatas pelos filtros aplicados. Excluir combinações fora de um padrão comum NÃO aumenta a probabilidade das combinações restantes -- apenas reduz quantas combinações 'atípicas' você joga.

Demonstração de fechamento completo com 16 dezenas (1 a 16): gera 8008 apostas, custo de exemplo R$ 40,040.00. GARANTIA CONDICIONAL: se as 6 dezenas sorteadas pela Mega-Sena estiverem TODAS entre as 16 dezenas que você escolheu, então, necessariamente, UMA das 8008 apostas geradas acerta a sena (6 pontos). Esta é uma garantia matemática do fechamento completo, não uma estimativa. NÃO é garantido que as 6 dezenas sorteadas estarão dentro do seu conjunto de 16 escolhidas -- isso continua sendo um evento aleatório, com probabilidade 0.00015996 (aproximadamente 1 em 6,252) se as dezenas forem escolhidas sem nenhuma informação preditiva real. O fechamento não aumenta essa probabilidade -- ele garante apenas que, SE isso acontecer, você não vai deixar de ter uma aposta premiada com 6 pontos dentro do conjunto coberto. Se você e outras pessoas acertarem a sena no mesmo concurso, o prêmio é dividido entre todos os ganhadores -- o fechamento não protege contra isso, e conjuntos de dezenas populares (ex: datas de aniversário, 1 a 31) tendem a ser escolhidos por mais apostadores.

## 10. Análise financeira

O arquivo `dados/megasena.csv` usado nesta execução não inclui colunas de premiação, arrecadação ou número de ganhadores -- por isso não foi possível calcular retorno líquido, valor esperado real ou ponto de equilíbrio com dados reais de prêmio. O código em `gerador_combinacoes.py` e `fechamento.py` já calcula custo total e probabilidade teórica a partir de qualquer preço de aposta informado; para uma análise financeira completa, forneça um CSV com colunas de premiação por faixa e o pipeline pode ser estendido para consumir esses dados. Independentemente disso: a Mega-Sena, como toda loteria, tem vantagem matemática estrutural para a organizadora (parte da arrecadação não retorna como prêmio), e mesmo com prêmio acumulado alto, o valor esperado de uma aposta individual permanece, em geral, menor que o valor apostado.

## 11. Conclusão sobre previsibilidade

**1. Foi encontrada evidência estatística de que os concursos não são aleatórios?**

O teste de qui-quadrado de uniformidade geral teve valor-p = 0.8131. Ao testar cada dezena individualmente com correção de Bonferroni para 60 comparações simultâneas, **0 de 60** permaneceram significativas a 5%. Não sobrou nenhum desvio individual significativo após a correção para múltiplas comparações -- ou seja, os desvios brutos observados são compatíveis com o que se espera por puro acaso ao testar 60 hipóteses simultaneamente.

**2. Algum modelo superou consistentemente a seleção aleatória fora da amostra?**

Modelos com diferença estatisticamente significativa (p<0,05) e favorável no período de teste final: janela_movel. Isso precisaria se repetir em janelas temporais adicionais e novos dados antes de qualquer conclusão mais forte.

**3. A diferença foi estatisticamente significativa?**

Ver valores-p reportados na Seção 7 para cada modelo. Valores-p individuais não corrigidos para múltiplas comparações (vários modelos testados) devem ser interpretados com cautela adicional -- com muitos modelos testados, é esperado que algum dê p<0,05 por puro acaso.

**4. A diferença foi economicamente relevante depois dos custos?**

Mesmo nos casos com diferença estatisticamente significativa, a diferença de acertos médios observada (frações de acerto em 6 números por aposta) é pequena demais para compensar o custo de apostas adicionais ou mudar a conclusão prática: nenhum modelo aqui testado demonstrou vantagem financeira líquida esperada sobre apostar aleatoriamente.

**5. Os resultados poderiam ser explicados por acaso ou overfitting?**

Sim, plenamente. Este relatório testou 8 modelos e múltiplas métricas; mesmo sob aleatoriedade pura, é estatisticamente esperado que algumas comparações individuais cruzem o limiar de 5% de significância. A validação walk-forward e a separação entre validação e teste final ajudam a reduzir (não eliminar) o risco de overfitting.

**6. Existe alguma fórmula comprovada que garanta acertos?**

Não. Nenhuma fórmula, modelo ou sistema de fechamento testado neste projeto garante acertos. O único tipo de garantia matemática presente é a garantia CONDICIONAL do fechamento completo (Seção 9): se as 6 dezenas sorteadas estiverem dentro do conjunto escolhido, uma aposta acerta -- mas isso não aumenta a chance de as 6 dezenas sorteadas estarem, de fato, nesse conjunto.

**7. Qual é a forma mais correta de interpretar o sistema?**

Como uma ferramenta de análise descritiva, simulação e organização de apostas -- útil para entender a estrutura combinatória do jogo, visualizar frequências históricas e comparar formalmente hipóteses contra o acaso. Não deve ser interpretado como um sistema de previsão.

---

> Os dados históricos não demonstraram capacidade confiável de prever os próximos resultados. Os modelos servem para análise, simulação e organização de apostas, mas não oferecem garantia de acerto ou lucro.
