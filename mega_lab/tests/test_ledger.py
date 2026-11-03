import json

import numpy as np
import pytest

from megalab.models.statistical import statistical_models
from megalab.registry.ledger import Ledger, prospective_summary, register_predictions, score
from megalab.stats.montecarlo import simulate_draws


@pytest.fixture
def setup(tmp_path):
    draws = simulate_draws(120, np.random.default_rng(0))[0]
    return Ledger(tmp_path / "ledger.jsonl"), draws


def test_register_score_and_verify(setup):
    led, draws = setup
    models = statistical_models()[:3]
    e = register_predictions(led, draws, 120, "abc", models, registrado_em="2026-11-03T10:00:00Z")
    assert e["conteudo"]["concurso_alvo"] == 121
    assert set(e["conteudo"]["previsoes"]) == {m.name for m in models}
    for pr in e["conteudo"]["previsoes"].values():
        assert len(pr["bilhete"]) == 6 and len(pr["p"]) == 60
    new = score(led, {121: ([1, 2, 3, 4, 5, 6], "2026-11-03")})
    assert len(new) == 1 and not new[0]["conteudo"]["registro_tardio"]
    assert score(led, {121: ([1, 2, 3, 4, 5, 6], "2026-11-03")}) == []  # não avalia duas vezes
    ok, problems = led.verify()
    assert ok, problems
    s = prospective_summary(led)
    assert s["concursos_avaliados"] == 1 and s["modelos"]["aleatorio"]["concursos"] == 1


def test_refuses_duplicate_registration(setup):
    led, draws = setup
    register_predictions(led, draws, 120, "abc", statistical_models()[:1])
    with pytest.raises(ValueError):
        register_predictions(led, draws, 120, "abc", statistical_models()[:1])


def test_refuses_non_contiguous_base(setup):
    led, draws = setup
    with pytest.raises(ValueError):
        register_predictions(led, draws, 150, "abc", statistical_models()[:1])


def test_tampering_is_detected(setup):
    led, draws = setup
    register_predictions(led, draws, 120, "abc", statistical_models()[:2])
    lines = led.path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["conteudo"]["previsoes"]["aleatorio"]["bilhete"] = [1, 2, 3, 4, 5, 6]
    lines[0] = json.dumps(rec)
    led.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, problems = led.verify()
    assert not ok and "alterado" in problems[0]
    with pytest.raises(RuntimeError):
        register_predictions(led, np.vstack([draws, draws[:1]]), 121, "abc", statistical_models()[:1])


def test_late_registration_flagged_and_excluded(setup):
    led, draws = setup
    register_predictions(led, draws, 120, "abc", statistical_models()[:1],
                         registrado_em="2026-11-04T01:00:00Z")
    new = score(led, {121: ([1, 2, 3, 4, 5, 6], "2026-11-03")})
    assert new[0]["conteudo"]["registro_tardio"]
    assert prospective_summary(led)["concursos_avaliados"] == 0


def test_tickets_are_deterministic(tmp_path):
    draws = simulate_draws(120, np.random.default_rng(0))[0]
    a = register_predictions(Ledger(tmp_path / "a.jsonl"), draws, 120, "x", statistical_models())
    b = register_predictions(Ledger(tmp_path / "b.jsonl"), draws, 120, "x", statistical_models())
    assert a["conteudo"]["previsoes"] == b["conteudo"]["previsoes"]
