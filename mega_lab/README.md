# MEGA LAB — Engenharia reversa da Mega-Sena

Laboratório computacional para investigar, com rigor estatístico, se existe
**alguma** informação nos resultados passados da Mega-Sena capaz de melhorar
previsões em relação ao acaso.

> **Regra fundamental:** nenhuma suposta descoberta é aceita sem evidência
> fora da amostra e comparação com uma linha de base aleatória.

**Resultado atual (base até o concurso 3080):** nenhum padrão preditivo
encontrado. Ver [`reports/RELATORIO_DESCOBERTAS.md`](reports/RELATORIO_DESCOBERTAS.md).
Este projeto **não** é recomendação de aposta.

| | Módulo | O que faz |
|---|---|---|
| 01 | Banco histórico | 3.080 concursos (1996–2026) reconciliados entre 9 fontes, com hash SHA-256, status de verificação por concurso e ordem de extração |
| 02 | Estatística | frequências, atrasos, somas, paridade, faixas, pares, probabilidades exatas |
| 03 | Anomalias | 21 hipóteses pré-registradas (uniformidade, pares, trios, memória, posição de extração, mudança de regime, fatores conhecidos) |
| 04 | Machine learning | regressão logística e gradient boosting sobre features causais, re-treino periódico |
| 05 | Monte Carlo | nula por simulação (1.000 históricos justos por teste), análise de poder com bolas viciadas simuladas |
| 06 | Backtesting | walk-forward sem vazamento, nula exata, e-valores, Holm, 1.000 jogadores aleatórios de controle |
| — | Registro prospectivo | livro-razão append-only encadeado por hash, previsões gravadas antes de cada sorteio |

## Instalação

```bash
cd mega_lab
pip install -e .[dev]
pytest            # 65 testes: dados, anti-vazamento, calibração das nulas, livro-razão
```

## Uso

```bash
megalab data check         # reconstrói a base a partir dos extratos versionados e confere integridade
megalab data update        # busca concursos novos (API da Caixa + espelhos) e reconcilia
megalab data add-result --concurso 3081 --data 2026-11-03 --dezenas 1 2 3 4 5 6   # inclusão manual

megalab analyze            # Família 1: aleatoriedade e anomalias (≈ 3 min)
megalab backtest           # Família 2: walk-forward dos 13 modelos (≈ 1 min)
megalab report             # gera reports/RELATORIO_DESCOBERTAS.md

megalab ledger score       # avalia previsões cujo resultado já está na base
megalab ledger predict     # registra previsões para o próximo concurso
megalab ledger verify      # confere a cadeia de hashes
megalab ledger status      # desempenho prospectivo acumulado
```

Ciclo após cada sorteio: `megalab data update && megalab ledger score && megalab backtest && megalab ledger predict && megalab report`,
seguido de commit e push **antes** do próximo sorteio (o commit é o carimbo de
tempo público). O workflow `.github/workflows/mega_lab_ledger.yml` automatiza
isso quando o GitHub Actions estiver habilitado no repositório.

## Documentação

- [`docs/01_auditoria_repositorios.md`](docs/01_auditoria_repositorios.md) — auditoria dos repositórios de referência
- [`docs/02_arquitetura.md`](docs/02_arquitetura.md) — arquitetura, interfaces, métricas
- [`docs/03_protocolo_cientifico.md`](docs/03_protocolo_cientifico.md) — hipóteses pré-registradas, períodos, regras de decisão, adendo H17
- [`docs/04_fontes_de_dados.md`](docs/04_fontes_de_dados.md) — fontes, licenças, reconciliação

## Garantias metodológicas (verificadas por testes)

- Modelos recebem cópia **somente-leitura** do passado; tentar escrever falha.
- Alterar concursos futuros não altera nenhuma feature nem previsão passada.
- A base versionada é exatamente a reconstruída a partir dos extratos.
- p-valores de acertos usam a distribuição exata (soma de hipergeométricas),
  válida para qualquer estratégia; o simulador de Monte Carlo é testado contra
  as probabilidades exatas.
- E-valores são supermartingais sob H0 (desigualdade de Ville verificada por simulação).
- Qualquer alteração no livro-razão quebra a cadeia de hashes.

## Licença

Código: MIT. Dados: resultados oficiais da Caixa (fatos públicos), extraídos
das fontes listadas em `docs/04_fontes_de_dados.md`.
