"""Interface de linha de comando: ``megalab <grupo> <ação>``."""

from __future__ import annotations

import argparse
import json
import os
import sys

# Cargas com muitas matrizes pequenas: threads de BLAS concorrentes deixam tudo
# mais lento (e não alteram resultados). Respeita valores já definidos.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from megalab import config  # noqa: E402


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def cmd_data(args) -> int:
    from megalab.data import store
    if args.acao == "build":
        rep = store.build()
    elif args.acao == "update":
        rep = store.update(use_caixa_api=not args.sem_caixa)
    elif args.acao == "add-result":
        rep = store.add_manual_result(args.concurso, args.data, args.dezenas, args.ordem)
    else:  # check
        rep = store.build_from_extracts()
    summary = rep.get("integridade", rep)
    _print({k: v for k, v in summary.items() if k not in ("problemas", "regimes_calendario")})
    if "espelhos" in rep:
        _print({"espelhos": rep["espelhos"], "caixa_api": rep["caixa_api"]})
    return 0 if summary.get("n_problemas", 0) == 0 else 1


def cmd_analyze(args) -> int:
    from megalab import pipeline
    res = pipeline.analyze(args.replicas, args.permutacoes, args.poder)
    for h in res["hipoteses"]:
        print(f"{h['id']:5s} p_D={h['p_descoberta']:.3f} p_Holm={h['p_holm_descoberta']:.3f} "
              f"p_C={h['p_confirmacao']:.3f}  {h['veredito']}")
    return 0


def cmd_backtest(args) -> int:
    from megalab import pipeline
    ev = pipeline.run_backtest(args.jogadores, progress=lambda i, T: print(f"  {i}/{T}", file=sys.stderr))
    hold = ev.get("holdout", {})
    for m, r in hold.get("modelos", {}).items():
        print(f"{m:15s} acertos={r['acertos_total']:5d} (esp. {r['acertos_esperado']:.0f}) "
              f"p={r['p_unilateral']:.3f}  bits/conc={r['bits_media']:+.4f} p={r['bits_p_unilateral']:.3f}")
    print("Significativos no holdout (Holm):", hold.get("modelos_significativos"))
    return 0


def cmd_ledger(args) -> int:
    from megalab import pipeline
    from megalab.registry import ledger as lg
    led = lg.Ledger()
    if args.acao == "predict":
        try:
            e = pipeline.ledger_predict(led)
        except ValueError as err:  # já registrado para este concurso: nada a fazer
            print(f"Nada registrado: {err}")
            return 0
        c = e["conteudo"]
        print(f"Registrado seq={e['seq']} para o concurso {c['concurso_alvo']} em {e['registrado_em']}")
        for m, pr in c["previsoes"].items():
            print(f"  {m:15s} {' '.join(f'{x:02d}' for x in pr['bilhete'])}")
    elif args.acao == "score":
        new = pipeline.ledger_score(led)
        print(f"{len(new)} avaliação(ões) registrada(s)")
    elif args.acao == "verify":
        ok, problems = led.verify()
        print("OK: cadeia íntegra" if ok else "FALHA:\n" + "\n".join(problems))
        return 0 if ok else 1
    else:
        _print(lg.prospective_summary(led))
    return 0


def cmd_report(args) -> int:
    from megalab.reports import build
    path = build.build_report()
    print(f"Relatório: {path}")
    return 0


def cmd_all(args) -> int:
    from megalab import pipeline
    from megalab.reports import build
    pipeline.analyze()
    pipeline.run_backtest()
    pipeline.ledger_score()
    print(build.build_report())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="megalab", description="MEGA LAB — investigação estatística da Mega-Sena")
    sub = p.add_subparsers(dest="grupo", required=True)

    d = sub.add_parser("data", help="base histórica")
    d.add_argument("acao", choices=["build", "update", "check", "add-result"])
    d.add_argument("--sem-caixa", action="store_true", help="não consultar a API da Caixa")
    d.add_argument("--concurso", type=int)
    d.add_argument("--data", help="AAAA-MM-DD")
    d.add_argument("--dezenas", type=int, nargs=6)
    d.add_argument("--ordem", type=int, nargs=6)
    d.set_defaults(func=cmd_data)

    a = sub.add_parser("analyze", help="Família 1: aleatoriedade, anomalias, Monte Carlo")
    a.add_argument("--replicas", type=int, default=config.MC_REPLICATES)
    a.add_argument("--permutacoes", type=int, default=config.PERMUTATIONS)
    a.add_argument("--poder", type=int, default=200, help="réplicas da análise de poder")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("backtest", help="Família 2: walk-forward de todos os modelos")
    b.add_argument("--jogadores", type=int, default=1000, help="jogadores aleatórios de controle")
    b.set_defaults(func=cmd_backtest)

    l = sub.add_parser("ledger", help="registro prospectivo de previsões")
    l.add_argument("acao", choices=["predict", "score", "verify", "status"])
    l.set_defaults(func=cmd_ledger)

    r = sub.add_parser("report", help="gera reports/RELATORIO_DESCOBERTAS.md")
    r.set_defaults(func=cmd_report)

    al = sub.add_parser("all", help="analyze + backtest + ledger score + report")
    al.set_defaults(func=cmd_all)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
