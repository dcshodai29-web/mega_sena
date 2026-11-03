"""Testes de causalidade: nenhum modelo ou feature pode enxergar o futuro."""

import numpy as np
import pytest

from megalab.evaluation import backtest
from megalab.models.features import feature_tensor, labels
from megalab.models.ml import MLModel, ml_models
from megalab.models.statistical import statistical_models
from megalab.stats.montecarlo import simulate_draws


@pytest.fixture(scope="module")
def draws():
    return simulate_draws(400, np.random.default_rng(1))[0]


def test_feature_tensor_is_causal(draws):
    F = feature_tensor(draws)
    rng = np.random.default_rng(2)
    for s in (1, 2, 3, 57, 201, 399):
        perturbed = draws.copy()
        perturbed[s:] = simulate_draws(len(draws) - s, rng)[0]
        Fp = feature_tensor(perturbed)
        np.testing.assert_array_equal(F[: s + 1], Fp[: s + 1], err_msg=f"vazamento em s={s}")
        assert not np.array_equal(F[s + 1:], Fp[s + 1:])


def test_feature_tensor_prefix_equals_tensor_of_prefix(draws):
    """F calculado na série inteira, cortado em t, = F calculado só com o passado."""
    F = feature_tensor(draws)
    for t in (5, 100, 333):
        np.testing.assert_allclose(F[: t + 1], feature_tensor(draws[:t]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("model", statistical_models(), ids=lambda m: m.name)
def test_statistical_models_ignore_future(model, draws):
    t = 250
    p = model.predict_proba(draws[:t].copy())
    assert p.shape == (60,) and abs(p.sum() - 6) < 1e-9 and (p > 0).all() and (p < 1).all()
    alt = draws.copy()
    alt[t:] = simulate_draws(len(draws) - t, np.random.default_rng(3))[0]
    np.testing.assert_array_equal(p, model.predict_proba(alt[:t].copy()))


def test_history_passed_to_models_is_read_only(draws):
    class Malicious(statistical_models()[0].__class__):
        name = "malicioso"

        def predict_proba(self, history):
            history[0, 0] = 99  # tentativa de alterar dados
            return super().predict_proba(history)

    with pytest.raises(ValueError):
        backtest.run_backtest(draws, np.arange(1, len(draws) + 1), start_concurso=390,
                              models=[Malicious()], random_players=0)


def test_ml_model_requires_exact_past_slices(draws):
    m = ml_models()[0]
    F, Y = feature_tensor(draws), labels(draws)
    with pytest.raises(ValueError):
        m.predict_from_features(F, Y, 200)  # entregou o futuro inteiro → recusado


def test_ml_backtest_path_equals_standalone_path_at_refit_points(draws):
    F, Y = feature_tensor(draws), labels(draws)
    for m in ml_models():
        m.reset()
        t = 300  # múltiplo do intervalo de re-treino
        p_bt = m.predict_from_features(F[: t + 1].copy(), Y[:t].copy(), t)
        p_alone = MLModel(m.name, m.description, m._factory).predict_proba(draws[:t].copy())
        np.testing.assert_allclose(p_bt, p_alone, atol=1e-9)


def test_backtest_predictions_unchanged_when_future_changes(draws):
    concursos = np.arange(1, len(draws) + 1)
    models = statistical_models()[:4] + ml_models()
    r1 = backtest.run_backtest(draws, concursos, start_concurso=301, models=models, random_players=0)
    alt = draws.copy()
    alt[350:] = simulate_draws(len(draws) - 350, np.random.default_rng(9))[0]
    models = statistical_models()[:4] + ml_models()
    r2 = backtest.run_backtest(alt, concursos, start_concurso=301, models=models, random_players=0)
    k = 350 - 300  # índices de avaliação até o concurso 350 (0-based t=349)
    for name in r1.proba:
        np.testing.assert_allclose(r1.proba[name][:k + 1], r2.proba[name][:k + 1], atol=1e-12,
                                   err_msg=name)
        np.testing.assert_array_equal(r1.tickets[name][:k + 1], r2.tickets[name][:k + 1])
