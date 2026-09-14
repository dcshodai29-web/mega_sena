# Mega-Sena — Sistema Educacional de Análise Estatística e Simulação

Sistema para análise descritiva, testes de aleatoriedade, simulação
Monte Carlo, modelos experimentais de ranqueamento (com validação
temporal honesta) e organização de apostas para a Mega-Sena.

## O que este sistema NÃO é

Não é um previsor. Não garante acertos, lucro ou qualquer vantagem
sobre apostar aleatoriamente. Todas as regras de honestidade
estatística que motivaram este projeto estão detalhadas em
`relatorio_mega_sena.md`, especialmente na seção final "Conclusão
sobre previsibilidade".

## Dados usados

`dados/megasena.csv` contém **506 concursos reais** da Mega-Sena, do
concurso 2551 (04/01/2023) ao concurso 3056 (10/09/2026), coletados de
fontes públicas de resultado de loteria e com datas conferidas.

**Isto é um subconjunto do histórico completo** (a Mega-Sena começou em
1996). Para rodar a análise sobre o histórico completo, substitua este
arquivo por um CSV com todos os concursos desde 1996, mantendo as
colunas: `concurso,data,n1,n2,n3,n4,n5,n6` (data em formato `AAAA-MM-DD`).
Nenhum outro arquivo precisa mudar.

## Instalação

```bash
cd mega_sena
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

Pipeline completo (leva alguns minutos por causa da simulação Monte Carlo):

```bash
cd src
python3 executar_pipeline.py
```

Modo rápido, para testar se está tudo funcionando (reduz a simulação
Monte Carlo para 2.000 repetições em vez de 20.000):

```bash
python3 executar_pipeline.py --rapido
```

Ao final, confira:

- `relatorio_mega_sena.md` — relatório técnico completo
- `graficos/` — todos os gráficos gerados (PNG)
- `resultados/` — todas as tabelas em CSV (frequências, testes, modelos, etc.)

## Testes automatizados

```bash
cd mega_sena
python3 -m pytest testes/ -v
```

## Estrutura

```
mega_sena/
├── dados/
│   └── megasena.csv              # dados reais (506 concursos, ver acima)
├── src/
│   ├── carregar_dados.py         # leitura e tipagem do CSV
│   ├── validar_dados.py          # validação completa + limpeza segura
│   ├── estatisticas.py           # análise descritiva (20+ métricas)
│   ├── testes_aleatoriedade.py   # qui-quadrado, runs test, autocorrelação, IC, Bonferroni
│   ├── simulacao.py              # Monte Carlo vetorizado em lotes
│   ├── modelos.py                # 9 modelos de ranqueamento (inclui Beta-Binomial e regressão logística)
│   ├── validacao_temporal.py     # split treino/validação/teste + walk-forward
│   ├── gerador_combinacoes.py    # gerador com orçamento, filtros e probabilidade
│   ├── fechamento.py             # sistema de fechamento completo (wheel)
│   ├── relatorio.py              # gráficos + montagem do relatório Markdown
│   └── executar_pipeline.py      # orquestra tudo, do início ao fim
├── testes/
│   └── test_basico.py            # 18 testes: fórmulas, validação, estatística, gerador, fechamento
├── graficos/                      # gerado pela execução
├── resultados/                    # gerado pela execução
├── requirements.txt
├── README.md
└── relatorio_mega_sena.md         # gerado pela execução
```

## Limitações conhecidas (declaradas de propósito)

1. **Volume de dados**: 506 concursos é uma fração do histórico
   completo da Mega-Sena. Resultados podem mudar com mais dados.
2. **Monte Carlo reduzido**: rodamos 20.000 repetições em vez das
   100.000 inicialmente propostas, por custo computacional das métricas
   calculadas por série (ver nota no próprio relatório). O código
   suporta aumentar isso (`n_repeticoes` em `simulacao.rodar_monte_carlo`).
3. **Sem dados de premiação**: o CSV usado não tem colunas de prêmio,
   arrecadação ou ganhadores, então a análise financeira (Seção 10 do
   relatório) é qualitativa, não numérica. O código já está pronto para
   consumir essas colunas se um CSV mais completo for fornecido.
4. **Sistemas de fechamento reduzidos** (com garantia parcial, cobrindo
   menos que todas as combinações) não foram implementados nesta
   versão — só o fechamento completo, que tem garantia matemática mais
   simples de verificar.
5. **Modelo de regressão logística** está implementado em `modelos.py`
   mas não foi incluído no comparativo principal do walk-forward por
   custo computacional (treina 60 regressões por concurso avaliado);
   pode ser habilitado adicionando `"regressao_logistica"` à lista de
   modelos em `executar_pipeline.py`.
