"""Garantias sobre os artefatos versionados (livro-razão, relatórios)."""

import json

from megalab import config
from megalab.data import store
from megalab.registry.ledger import Ledger


def test_committed_ledger_chain_is_intact():
    led = Ledger(config.LEDGER_PATH)
    ok, problems = led.verify()
    assert ok, problems


def test_committed_predictions_were_made_for_unknown_draws():
    """Cada previsão foi registrada com a base terminando no concurso anterior ao alvo."""
    for target, e in Ledger(config.LEDGER_PATH).predictions().items():
        c = e["conteudo"]
        assert c["concurso_alvo"] == c["base_ultimo_concurso"] + 1
        for pr in c["previsoes"].values():
            assert len(set(pr["bilhete"])) == 6
            assert abs(sum(pr["p"]) - 6) < 1e-3


def test_reports_refer_to_current_base():
    base = store.load()
    resumo = json.loads((config.REPORTS_DIR / "resumo.json").read_text(encoding="utf-8"))
    assert resumo["base_sha256"] == base.sha256


def test_report_builds(tmp_path, monkeypatch):
    from megalab.reports import build
    monkeypatch.setattr(build, "REPORT", tmp_path / "r.md")
    monkeypatch.setattr(build, "FIG", tmp_path / "fig")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    path = build.build_report()
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert path.endswith("r.md") and "Relatório de descobertas" in text
    assert len(list((tmp_path / "fig").glob("*.png"))) == 7
