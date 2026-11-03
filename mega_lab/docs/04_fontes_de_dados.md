# 04 — Fontes de dados e base histórica verificável

## Fontes usadas

| ID | Descrição | Origem dos dados | Concursos | Ordem de extração | Licença |
|---|---|---|---|---|---|
| caixa_planilha_oficial | Planilha `Mega-Sena.xlsx` exportada de loterias.caixa.gov.br (via luizvalle/mega_sena_analysis) | Caixa | 1 – 2964 | não | dados públicos da Caixa; repositório MIT |
| asloterias_csv | CSV baixado de asloterias.com.br (luizvalle/mega_sena_analysis) | asloterias | 1 – 2964 | não | repositório MIT |
| megaloterias_rpkg | Pacote R loteria (damarals → McGuyverBR), raspagem de megaloterias.com.br | megaloterias | 1 – 2458 | não | MIT |
| caixa_api_salles | Exportação JSON da API oficial da Caixa (salles97/MegaSena) | Caixa | 1 – 3016 | não | sem licença |
| osjunnior_csv | CSV de 506 concursos (OsJunnior/mega_sena) | desconhecida | 2551 – 3056 | não | sem licença |
| asloterias_ordem | CSV com ordem de extração (RoninXK/megadata-megasena) | asloterias | 1 – 3066 | **sim** | sem licença |
| guilhermeasn_ordem | JSON com ordem de extração (guilhermeasn/loteria.json) | desconhecida | 1 – 2797 | **sim** | não informada |
| lotolab_api | JSON coletado da API da Caixa (faelw/lotolab-megasena) | Caixa | 1 – 3080 | não | sem licença |
| rust_spider | JSON de histórico (edxgar48/rust_spider_dados) | desconhecida | 1 – 3080 | não | não informada |

URLs (fixadas em commit quando possível) e hashes SHA-256 de cada arquivo
bruto estão em `src/megalab/data/sources.py` e em
`data/sources/MANIFEST.json`. De cada fonte foram extraídos **somente fatos**
(número do concurso, data e dezenas) para `data/sources/<id>.csv`; nenhum
código das fontes foi reutilizado. Para fontes sem licença declarada isso se
limita a dados factuais públicos (resultados oficiais de loteria).

**Fonte oficial bloqueada nesta sessão:** a API
`servicebus2.caixa.gov.br` e os sites asloterias/megaloterias foram
bloqueados pela política de rede do ambiente em que o laboratório foi
construído. O atualizador `megalab data update` usa a API oficial e funciona
em qualquer máquina com acesso a ela.

## Resultado da reconciliação (2026-11-03)

- **3.080 concursos**, do 1 (11/03/1996) ao 3080 (01/11/2026), **sem
  lacunas e sem duplicatas**.
- **Nenhuma divergência de dezenas** entre as 9 fontes em nenhum concurso.
- Status: **3.066 concursos "A"** (≥ 3 fontes concordantes) e **14
  concursos "B"** (3067–3080, 2 fontes concordantes: lotolab_api e
  rust_spider). Nenhum concurso depende de fonte única.
- **1 divergência de data**: concurso 2955 (Mega da Virada 2025) —
  5 fontes (incluindo a planilha oficial e a API da Caixa) dizem
  2026-01-01; 2 fontes dizem 2025-12-31. Prevalece a maioria/fonte oficial.
- **Ordem de extração** disponível para 3.066 concursos; nos 2.797 em que há
  duas fontes de ordem, **as duas concordam em 100 %**.
- Regimes de calendário inferidos dos dados: 1x/semana (concursos 1–280,
  até 07/2001), 2x/semana (281–2619, até 08/2023), 3x/semana (2620–3080).
- Sorteios de fim de ano (31/12 ou 01/01): 23 concursos.

Limitação importante: concordância entre fontes protege contra erros de
transcrição e raspagem, **não** contra um erro na origem (a Caixa). Várias
fontes derivam da mesma origem (coluna "Origem dos dados").

## Como reconstruir

```bash
megalab data build      # baixa (ou usa cache), confere SHA-256, extrai, reconcilia
pytest tests/test_data.py
```

O teste `test_reconciliation_is_reproducible` garante que a base versionada
é exatamente a reconstruída a partir dos extratos.

## Mudanças de equipamento e procedimento

Não foi possível obter, nesta sessão, registros oficiais datados de troca de
globos, bolas ou procedimentos. Por isso:

1. o arquivo `data/eventos_conhecidos.csv` existe para registrar eventos
   documentados **com fonte**; ele começa vazio — não inventamos datas;
2. a investigação de mudanças usa métodos que não dependem dessas datas:
   varredura de ponto de mudança (H13), deriva em janelas (H14) e regimes de
   calendário inferidos (H15d).
