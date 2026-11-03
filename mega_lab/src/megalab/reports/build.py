"""Monta ``reports/RELATORIO_DESCOBERTAS.md`` a partir dos resultados salvos."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from megalab import config
from megalab.combinatorics import TOTAL_COMBINATIONS, prize_odds
from megalab.data import store
from megalab.pipeline import (BACKTEST_CSV, BACKTEST_JSON, FAMILY1_JSON, FAMILY1_NULL, read_json,
                              write_json)
from megalab.registry.ledger import Ledger, prospective_summary
from megalab.reports import figures
from megalab.stats import randomness

REPORT = config.REPORTS_DIR / "RELATORIO_DESCOBERTAS.md"
FIG = config.FIGURES_DIR


def _p(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return "< 0,001" if x < 0.001 else f"{x:.3f}".replace(".", ",")


def _f(x, nd=3, sign=False) -> str:
    if x is None:
        return "—"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",")


def _int(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def build_report() -> str:
    base = store.load()
    integ = read_json(config.CANONICAL_DIR / "integridade.json")
    f1 = read_json(FAMILY1_JSON)
    btj = read_json(BACKTEST_JSON)
    frame = pd.read_csv(BACKTEST_CSV)
    null = dict(np.load(FAMILY1_NULL))
    led = Ledger()
    ok_ledger, _ = led.verify()
    prosp = prospective_summary(led)
    preds = led.predictions()

    # ---------------- figuras ----------------
    full = f1["periodos"]["completo"]
    figures.frequencies(f1["dezenas"]["completo"], full["n"], FIG / "01_frequencias.png")
    figures.hypothesis_pvalues(f1["hipoteses"], FIG / "02_hipoteses_pvalores.png")
    keys = ["H01", "H02", "H03", "H05", "H07", "H13"]
    figures.null_distributions(null, full["observado"], {k: randomness.DESCRIPTIONS[k] for k in keys}, keys,
                               FIG / "03_monte_carlo_nula.png")
    figures.power_curve(f1["poder"], full["n"], FIG / "04_poder.png")
    figures.backtest_hits(btj["holdout"], FIG / "05_backtest_holdout.png",
                          f"Holdout selado (concursos {btj['holdout']['concursos'][0]}–{btj['holdout']['concursos'][1]})")
    figures.evalue_paths(frame, FIG / "06_evalores.png")
    cp = f1["mudanca_regime"]
    cuts = np.arange(randomness.CHANGEPOINT_MARGIN, full["n"] - randomness.CHANGEPOINT_MARGIN + 1,
                     randomness.CHANGEPOINT_STEP) + 1
    figures.changepoint(cp["perfil"], cuts, full["nula"]["H13"]["p95"], cp["concurso_corte"],
                        FIG / "07_mudanca_regime.png")

    # ---------------- números-chave ----------------
    hyps = f1["hipoteses"]
    confirmed = [h for h in hyps if h["veredito"] == "ANOMALIA CONFIRMADA"]
    candidates = [h for h in hyps if h["veredito"].startswith("candidata")]
    raw_sig = [h for h in hyps if (h["p_descoberta"] or 1) < config.ALPHA]
    full_sig = [h for h in hyps if (h["p_completo_exploratorio"] or 1) < config.ALPHA]
    hold = btj["holdout"]
    winners = hold.get("modelos_significativos", [])
    scan = pd.DataFrame(f1["dezenas"]["completo"])
    top = scan.reindex(scan["z"].abs().sort_values(ascending=False).index).head(5)
    power = f1["poder"]
    detect80 = next((r["peso"] for r in power if r["poder_H02"] >= 0.8), None)
    detect80_freq = next((r["freq_relativa_media"] for r in power if r["poder_H02"] >= 0.8), None)
    h17 = hold["modelos"].get("freq_total", {})
    h17_rate = (h17.get("log10_evalor_final") or 0) / max(hold["n"], 1)
    h17_needed = (np.log10(20) / h17_rate) if h17_rate > 0 else None
    min_pc = min((h["p_confirmacao"] for h in hyps if h["p_confirmacao"] is not None), default=1.0)
    exploratory = [h for h in hyps if (h["p_completo_exploratorio"] or 1) < config.ALPHA
                   or (h["p_confirmacao"] or 1) < config.ALPHA]
    odds = prize_odds()
    naive = f1["padroes_ingenuos"]
    desc = f1["descritiva"]

    L: list[str] = []
    w = L.append
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w("# MEGA LAB — Relatório de descobertas")
    w("")
    w(f"*Gerado em {now} por `megalab report`. Base: {_int(base.last)} concursos "
      f"(1 a {base.last}), SHA-256 `{base.sha256[:16]}…`. Sementes fixas; "
      f"{f1['meta']['replicas_mc']} réplicas de Monte Carlo por teste.*")
    w("")
    w("> Hipóteses, evidências e resultados dos experimentos. Regra fundamental: nenhuma "
      "descoberta é aceita sem evidência fora da amostra e comparação com uma linha de base aleatória.")
    w("")
    w("## Sumário executivo")
    w("")
    if not confirmed and not winners:
        w("**Não encontramos nenhuma informação nos resultados passados que melhore a previsão "
          "da Mega-Sena em relação ao acaso.**")
    elif confirmed and not winners:
        w("**Encontramos anomalia(s) estatística(s) confirmada(s) fora da amostra, mas nenhum modelo "
          "conseguiu transformá-las em previsões melhores que o acaso.**")
    else:
        w("**Atenção: há resultado(s) que passaram nos critérios do holdout — ver seção 06; a "
          "confirmação definitiva depende do registro prospectivo.**")
    w("")
    w(f"- **Base:** {_int(integ['concursos'])} concursos de {integ['primeira_data']} a {integ['ultima_data']}, "
      f"sem lacunas; 9 fontes independentes concordam em 100 % das dezenas "
      f"({integ['status'].get('A', 0)} concursos com ≥ 3 fontes, {integ['status'].get('B', 0)} com 2).")
    w(f"- **Aleatoriedade (21 hipóteses pré-registradas):** {len(raw_sig)} com p < 0,05 no período de "
      f"descoberta sem correção (esperado por acaso: ~1); **{len(candidates) + len(confirmed)} sobreviveram "
      f"à correção de Holm**; **{len(confirmed)} confirmada(s)** no período de confirmação.")
    if exploratory:
        w(f"- **Sinal exploratório (não é descoberta):** fora da sequência pré-registrada, "
          f"{len(exploratory)} estatísticas ficam abaixo de 0,05 no período 2001–3080 ou na base inteira "
          f"({', '.join(h['id'] for h in exploratory)}). Quase todas reagem à mesma coisa — frequências um "
          f"pouco desiguais entre dezenas (ex.: dezena {int(top.iloc[0]['dezena']):02d} saiu "
          f"{int(top.iloc[0]['frequencia'])} vezes contra {_f(top.iloc[0]['esperado'], 0)} esperadas). Como não "
          "apareceu no período de descoberta, não pode ser declarado padrão; foi registrado como **H17** "
          "para teste exclusivamente em concursos futuros.")
    w(f"- **Bolas viciadas:** a base teria {('≥ 80 % de chance de detectar uma bola com peso relativo ' + _f(detect80, 2)) if detect80 else 'pouco poder'} "
      f"(que sairia ≈ {_f((detect80_freq - 1) * 100 if detect80_freq else 0, 0)} % mais vezes que o esperado). "
      "Vieses físicos menores que isso não podem ser descartados — mas também não poderiam ser explorados de forma útil.")
    w(f"- **Previsão (13 modelos, walk-forward):** no holdout selado ({_int(hold['n'])} concursos), "
      f"{'nenhum modelo' if not winners else ', '.join(winners)} superou o acaso após correção de Holm. "
      f"Acertos médios entre {_f(min(r['acertos_media'] for r in hold['modelos'].values()))} e "
      f"{_f(max(r['acertos_media'] for r in hold['modelos'].values()))} por bilhete (acaso: 0,600); "
      + ("o melhor resultado em acertos foi da própria linha de base aleatória (`aleatorio`)."
         if max(hold['modelos'], key=lambda m: hold['modelos'][m]['acertos_media']) == "aleatorio"
         else "a linha de base aleatória ficou dentro do mesmo intervalo."))
    w(f"- **Registro prospectivo:** livro-razão com {len(preds)} rodada(s) de previsões registradas "
      f"antes do sorteio; {prosp['concursos_avaliados']} avaliada(s). Cadeia de hashes "
      f"{'íntegra' if ok_ledger else 'COM PROBLEMAS'}.")
    w("")

    # ---------------- 01 ----------------
    w("## 01 — Banco histórico")
    w("")
    w("| Item | Valor |")
    w("|---|---|")
    w(f"| Concursos | {_int(integ['concursos'])} (1 a {integ['ultimo']}) |")
    w(f"| Período | {integ['primeira_data']} a {integ['ultima_data']} |")
    w(f"| Concursos ausentes / duplicados | {len(integ['ausentes'])} / {len(integ['duplicados'])} |")
    w(f"| Status A (≥ 3 fontes) / B (2 fontes) / C / X | {integ['status'].get('A', 0)} / {integ['status'].get('B', 0)} / "
      f"{integ['status'].get('C', 0)} / {integ['status'].get('X', 0)} |")
    w(f"| Divergências de data | {integ['status_data'].get('conflito', 0)} (concurso 2955, resolvida pela maioria e pela planilha oficial) |")
    w(f"| Concursos com ordem de extração | {_int(integ['com_ordem'])} ({integ['status_ordem'].get('ok', 0)} conferidos em 2 fontes, 100 % de concordância) |")
    regs = "; ".join(f"{r['regime']}: {r['inicio']}–{r['fim']}" for r in integ["regimes_calendario"])
    w(f"| Regimes de calendário (inferidos) | {regs} |")
    w(f"| Sorteios de fim de ano | {len(integ['sorteios_fim_de_ano'])} |")
    w("")
    w("Detalhes: `docs/04_fontes_de_dados.md`, `data/canonical/integridade.json`, `data/sources/MANIFEST.json`.")
    w("")

    # ---------------- 02 ----------------
    w("## 02 — Estatística descritiva")
    w("")
    w(f"![Frequências](figures/01_frequencias.png)")
    w("")
    w(f"- Probabilidades exatas de uma aposta simples: sena 1 em {_int(TOTAL_COMBINATIONS)}; "
      f"quina 1 em {_int(round(1 / odds['quina']))}; quadra 1 em {_int(round(1 / odds['quadra']))}.")
    w(f"- Soma média das dezenas: {_f(desc['soma']['media'], 2)} (teórica {_f(desc['soma']['teorica_media'], 1)}).")
    w(f"- Combinações de 6 repetidas na história: {desc['combinacoes_repetidas_6']} "
      f"(esperado ≈ {_f(base.last * (base.last - 1) / 2 / TOTAL_COMBINATIONS, 2)}).")
    rep = desc["repeticao_anterior"]
    w("- Dezenas repetidas do concurso anterior (observado × esperado): " + "; ".join(
        f"{k}: {rep['observado'].get(k, 0)} × {_f(float(rep['esperado'][k]), 0)}" for k in ["0", "1", "2", "3"]) + ".")
    w("")
    w("Dezenas mais afastadas do esperado (base completa, exploratório):")
    w("")
    w("| Dezena | Frequência | Esperado | z | p bilateral | q (BH, 60 testes) |")
    w("|---|---|---|---|---|---|")
    for r in top.itertuples():
        w(f"| {r.dezena:02d} | {r.frequencia} | {_f(r.esperado, 0)} | {_f(r.z, 2, True)} | {_p(r.p_bilateral)} | {_p(r.q_bh)} |")
    w("")

    # ---------------- 03 ----------------
    w("## 03 — Anomalias e testes de aleatoriedade")
    w("")
    w("![p-valores](figures/02_hipoteses_pvalores.png)")
    w("")
    w("| ID | Hipótese | p descoberta | p Holm | p confirmação | p base completa (exploratório) | Veredito |")
    w("|---|---|---|---|---|---|---|")
    for h in hyps:
        w(f"| {h['id']} | {h['descricao']} | {_p(h['p_descoberta'])} | {_p(h['p_holm_descoberta'])} | "
          f"{_p(h['p_confirmacao'])} | {_p(h['p_completo_exploratorio'])} | {h['veredito']} |")
    w("")
    w(f"Mudança de regime (H13): o maior contraste antes × depois ocorre no concurso "
      f"{cp['concurso_corte']} (χ² = {_f(cp['chi2_max'], 1)}; 95 % do máximo sob o acaso = "
      f"{_f(full['nula']['H13']['p95'], 1)}).")
    w("")
    w("![Mudança de regime](figures/07_mudanca_regime.png)")
    w("")
    if exploratory:
        w("### Sinal exploratório e hipótese H17")
        w("")
        w("Pela regra pré-registrada, nada foi confirmado: nenhuma hipótese passou na correção de Holm "
          "no período de descoberta (1–2000). Mas seria desonesto esconder o que aparece fora dessa regra:")
        w("")
        for h in exploratory:
            w(f"- **{h['id']}** ({h['descricao']}): p confirmação = {_p(h['p_confirmacao'])}; "
              f"p base completa = {_p(h['p_completo_exploratorio'])}.")
        w("")
        w("Leitura: H01, H03, H05 e H07 são estatísticas correlacionadas — todas aumentam quando "
          "algumas dezenas saem mais que outras. O conjunto aponta para **uma** possível causa "
          "(frequências levemente desiguais, concentradas em 2001–3080), não para várias descobertas. "
          f"Na base completa, a dezena {int(top.iloc[0]['dezena']):02d} tem z = {_f(top.iloc[0]['z'], 2, True)} "
          f"(q de Benjamini-Hochberg = {_p(top.iloc[0]['q_bh'])} entre 60 dezenas). "
          "Contra isso: (1) o período 1–2000 não mostrou o efeito (H01 p = "
          f"{_p(next(h for h in hyps if h['id'] == 'H01')['p_descoberta'])}); (2) com 21 testes em C, "
          f"o menor p-valor ({_p(min_pc)}) vira {_p(min(1.0, min_pc * len(hyps)))} após Holm; "
          "(3) nenhum modelo transformou isso em mais acertos no holdout.")
        w("")
        w("Por isso a hipótese foi registrada como **H17** (adendo 1 de `docs/03_protocolo_cientifico.md`, "
          "commit anterior à execução do backtest) e será julgada **apenas em concursos ≥ 3081** pelo "
          "e-valor prospectivo do modelo `freq_total` (limiar 20).")
        if h17_needed:
            w(f"No holdout, `freq_total` acumulou log10 e = {_f(h17.get('log10_evalor_final'), 2, True)} em "
              f"{hold['n']} concursos. Se esse ritmo se mantiver, H17 precisaria de ≈ {_int(h17_needed)} "
              f"sorteios (≈ {_f(h17_needed / 156, 0)} anos a 3 por semana) para atingir e = 20 — ou seja, "
              "mesmo que o efeito seja real, ele é pequeno demais para ter valor prático.")
        w("")

    # ---------------- 04 ----------------
    w("## 04 — Machine learning")
    w("")
    w("Dois modelos (regressão logística e gradient boosting) receberam 10 features causais por dezena "
      "(frequências em janelas de 10/50/200/total, atraso, presença nos 2 concursos anteriores, EWMA, "
      "afinidade e transição com o concurso anterior) e foram re-treinados a cada 100 concursos usando "
      "apenas o passado. Os testes `tests/test_no_leakage.py` provam que alterar o futuro não altera "
      "nenhuma feature nem previsão.")
    w("")
    w("| Modelo | Período | Acertos médios | p (acertos) | bits/concurso | p (bits) |")
    w("|---|---|---|---|---|---|")
    for m in ("logistica", "gboost"):
        for per in ("desenvolvimento", "validacao", "holdout"):
            r = btj[per]["modelos"][m]
            w(f"| {m} | {per} | {_f(r['acertos_media'])} | {_p(r['p_unilateral'])} | "
              f"{_f(r['bits_media'], 4, True)} | {_p(r['bits_p_unilateral'])} |")
    w("")
    lv = btj["validacao"]["modelos"]["logistica"]
    lh = btj["holdout"]["modelos"]["logistica"]
    w(f"Exemplo didático do risco de sobreajuste: a regressão logística parecia promissora na validação "
      f"(acertos {_f(lv['acertos_media'])}, p = {_p(lv['p_unilateral'])}; bits p = {_p(lv['bits_p_unilateral'])}) "
      f"e caiu no holdout (acertos {_f(lh['acertos_media'])}, p = {_p(lh['p_unilateral'])}; "
      f"bits {_f(lh['bits_media'], 4, True)}). Se tivéssemos escolhido o modelo pela validação e parado ali, "
      "teríamos anunciado um falso padrão.")
    w("")

    # ---------------- 05 ----------------
    w("## 05 — Monte Carlo")
    w("")
    w("![Nula](figures/03_monte_carlo_nula.png)")
    w("")
    w(f"**Padrões no acaso puro.** Em {naive['n']} concursos, a dezena mais sorteada saiu "
      f"{naive['real']['dezena_mais_frequente']} vezes e a menos sorteada {naive['real']['dezena_menos_frequente']} "
      f"(esperado {_f(naive['esperado_por_dezena'], 0)}); o par mais frequente saiu "
      f"{naive['real']['par_mais_frequente']} vezes (esperado por par {_f(naive['esperado_por_par'], 1)}); "
      f"o trio mais frequente saiu {naive['real']['trio_mais_frequente']} vezes — nos históricos simulados "
      f"justos o trio máximo fica entre {_f(naive['nulo_trio_max']['p5'], 0)} e {_f(naive['nulo_trio_max']['p95'], 0)} "
      "(90 % central), isto é, o trio campeão real é até menos repetido que o típico. \"Dezenas quentes\", "
      "\"pares que saem juntos\" e \"trios campeões\" aparecem naturalmente em qualquer sorteio justo.")
    w("")
    w("![Poder](figures/04_poder.png)")
    w("")
    w("| Peso da bola | Frequência relativa média | Poder H01 | Poder H02 |")
    w("|---|---|---|---|")
    for r in power:
        w(f"| {_f(r['peso'], 2)} | {_f(r['freq_relativa_media'], 3)} | {_f(r['poder_H01'], 2)} | {_f(r['poder_H02'], 2)} |")
    w("")

    # ---------------- 06 ----------------
    w("## 06 — Backtesting (walk-forward, sem vazamento)")
    w("")
    w("![Holdout](figures/05_backtest_holdout.png)")
    w("")
    w(f"Holdout selado: concursos {hold['concursos'][0]}–{hold['concursos'][1]} ({_int(hold['n'])} sorteios). "
      "p-valores unilaterais exatos (acertos: soma de hipergeométricas) e condicionais (bits); "
      "Holm sobre 24 testes.")
    w("")
    w("| Modelo | Acertos | Esperado | p | Quadras | bits/concurso | p (bits) | log10 e-valor | Percentil entre aleatórios |")
    w("|---|---|---|---|---|---|---|---|---|")
    for m, r in hold["modelos"].items():
        w(f"| {m} | {r['acertos_total']} | {_f(r['acertos_esperado'], 0)} | {_p(r['p_unilateral'])} | "
          f"{r['quadras']} | {_f(r['bits_media'], 4, True)} | {_p(r['bits_p_unilateral'])} | "
          f"{_f(r['log10_evalor_final'], 2, True)} | {_f(100 * r.get('percentil_entre_aleatorios', np.nan), 0)} % |")
    w("")
    rp = hold.get("jogadores_aleatorios")
    if rp:
        w(f"1.000 jogadores aleatórios no mesmo período: acertos totais entre {rp['min']} e {rp['max']} "
          f"(95 % central: {_f(rp['p2_5'], 0)}–{_f(rp['p97_5'], 0)}).")
        w("")
    w(f"Modelos significativos após Holm: **{', '.join(winners) if winners else 'nenhum'}**.")
    w("")
    w("![e-valores](figures/06_evalores.png)")
    w("")
    w("Desempenho por período (acertos médios por bilhete):")
    w("")
    pers = [p for p in ("desenvolvimento", "validacao", "holdout") if p in btj]
    w("| Modelo | " + " | ".join(pers) + " |")
    w("|---|" + "---|" * len(pers))
    for m in hold["modelos"]:
        w(f"| {m} | " + " | ".join(_f(btj[p]["modelos"][m]["acertos_media"]) for p in pers) + " |")
    w("")

    # ---------------- prospectivo ----------------
    w("## Registro prospectivo (livro-razão)")
    w("")
    if preds:
        lastp = preds[max(preds)]
        c = lastp["conteudo"]
        w(f"Última rodada registrada: concurso **{c['concurso_alvo']}**, em {lastp['registrado_em']} "
          f"(seq {lastp['seq']}, hash `{lastp['hash'][:16]}…`), com a base até o concurso "
          f"{c['base_ultimo_concurso']}.")
        w("")
        w("| Modelo | Bilhete registrado |")
        w("|---|---|")
        for m in [k for k in hold["modelos"] if k in c["previsoes"]]:
            pr = c["previsoes"][m]
            w(f"| {m} | {' '.join(f'{x:02d}' for x in pr['bilhete'])} |")
        w("")
    if prosp["concursos_avaliados"]:
        w("| Modelo | Concursos | Acertos | Esperado | log10 e-valor | Supera e = 240? |")
        w("|---|---|---|---|---|---|")
        for m, s in prosp["modelos"].items():
            w(f"| {m} | {s['concursos']} | {s['acertos']} | {_f(s['acertos_esperado'], 1)} | "
              f"{_f(s['log10_evalor'], 2, True)} | {'sim' if s['supera_limiar'] else 'não'} |")
        w("")
    else:
        w("Nenhum concurso prospectivo avaliado ainda. Após cada sorteio: `megalab data update && "
          "megalab ledger score && megalab ledger predict`.")
        w("")

    # ---------------- descobertas ----------------
    w("## Relatório de descobertas")
    w("")
    w("| # | Hipótese | Evidência | Resultado |")
    w("|---|---|---|---|")
    rows = [
        ("Integridade", "A base pública tem erros ou lacunas",
         f"9 fontes, {_int(integ['concursos'])} concursos, 0 divergências de dezenas", "Rejeitada: base íntegra"),
        ("Uniformidade", "Algumas dezenas saem mais que outras (H01, H02)",
         f"p_D = {_p(next(h for h in hyps if h['id']=='H01')['p_descoberta'])} / {_p(next(h for h in hyps if h['id']=='H02')['p_descoberta'])}; "
         f"p_C = {_p(next(h for h in hyps if h['id']=='H01')['p_confirmacao'])} / {_p(next(h for h in hyps if h['id']=='H02')['p_confirmacao'])}",
         next(h for h in hyps if h['id'] == 'H01')["veredito"]),
        ("Combinações", "Pares, trios ou combinações se repetem além do acaso (H03–H05)",
         "; ".join(f"{h['id']}: p_D = {_p(h['p_descoberta'])}" for h in hyps if h["id"] in ("H03", "H04", "H05")),
         "; ".join(sorted({h['veredito'] for h in hyps if h['id'] in ('H03', 'H04', 'H05')}))),
        ("Memória", "O sorteio depende dos anteriores (H06–H09)",
         "; ".join(f"{h['id']}: p_D = {_p(h['p_descoberta'])}" for h in hyps if h["id"] in ("H06", "H07", "H08", "H09")),
         "; ".join(sorted({h['veredito'] for h in hyps if h['id'] in ('H06', 'H07', 'H08', 'H09')}))),
        ("Física das bolas", "Bolas saem em posições preferenciais da ordem de extração (H11)",
         f"{_int(integ['com_ordem'])} concursos com ordem; p_D = {_p(next(h for h in hyps if h['id']=='H11')['p_descoberta'])}",
         next(h for h in hyps if h['id'] == 'H11')["veredito"]),
        ("Equipamento / procedimento", "Houve mudança de regime ou deriva (H13, H14, H15d)",
         "; ".join(f"{h['id']}: p_D = {_p(h['p_descoberta'])}" for h in hyps if h["id"] in ("H13", "H14", "H15d")),
         "; ".join(sorted({h['veredito'] for h in hyps if h['id'] in ('H13', 'H14', 'H15d')}))),
        ("Fatores conhecidos", "Dia da semana, fim de ano ou prêmio acumulado influenciam (H15a–c)",
         "; ".join(f"{h['id']}: p_D = {_p(h['p_descoberta'])}" for h in hyps if h["id"] in ("H15a", "H15b", "H15c")),
         "; ".join(sorted({h['veredito'] for h in hyps if h['id'] in ('H15a', 'H15b', 'H15c')}))),
        ("Previsibilidade", "Algum modelo prevê melhor que o acaso (H16)",
         f"holdout {_int(hold['n'])} concursos, 12 modelos + hedge, Holm",
         "Nenhum modelo superou o acaso" if not winners else "Candidato(s): " + ", ".join(winners) + " — aguarda prospectivo"),
        ("Sinal exploratório H17", "Frequências desiguais entre dezenas (gerada pelos dados de 2001–3080)",
         f"p_C(H01) = {_p(next(h for h in hyps if h['id']=='H01')['p_confirmacao'])}; "
         f"dezena {int(top.iloc[0]['dezena']):02d}: z = {_f(top.iloc[0]['z'], 2, True)}; "
         f"freq_total no holdout: bits p = {_p(h17.get('bits_p_unilateral'))}, acertos p = {_p(h17.get('p_unilateral'))}",
         "Não confirmada; em teste prospectivo (concursos ≥ 3081)"),
    ]
    for i, (a, b, c, d) in enumerate(rows, 1):
        w(f"| {i} | **{a}** — {b} | {c} | {d} |")
    w("")
    w("## Limitações")
    w("")
    w("- Concordância entre fontes não protege contra erro na origem (Caixa).")
    w("- Não havia registros oficiais datados de troca de globos/bolas; mudanças foram investigadas "
      "apenas por métodos que não dependem dessas datas (H13, H14, H15d).")
    w("- A base só detecta vieses físicos acima do limiar mostrado na análise de poder.")
    w("- Um modelo de linguagem pode ter visto resultados antigos; o teste definitivo é o prospectivo.")
    w("- Ausência de evidência em uma bateria finita de hipóteses não prova aleatoriedade perfeita; "
      "prova que nenhuma das hipóteses testadas oferece vantagem.")
    w("")
    w("## Como reproduzir")
    w("")
    w("```bash")
    w("cd mega_lab && pip install -e .[dev]")
    w("pytest                      # testes automatizados")
    w("megalab data check          # reconstrói a base a partir dos extratos e confere integridade")
    w("megalab analyze             # Família 1 (≈ 3 min)")
    w("megalab backtest            # Família 2 (≈ 5 min)")
    w("megalab report              # este relatório")
    w("```")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(config.REPORTS_DIR / "resumo.json", {
        "gerado_em": now, "base_sha256": base.sha256, "ultimo_concurso": base.last,
        "anomalias_confirmadas": [h["id"] for h in confirmed],
        "candidatas_holm": [h["id"] for h in candidates],
        "modelos_significativos_holdout": winners,
        "prospectivo": prosp,
    })
    return str(REPORT)
