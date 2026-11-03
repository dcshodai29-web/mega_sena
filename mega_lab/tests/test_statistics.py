"""Testes da matemática e calibração das nulas."""

from math import comb

import numpy as np
import pytest

from megalab import combinatorics as cb
from megalab.evaluation import metrics
from megalab.models.base import normalize, scores_to_proba, ticket_from_proba
from megalab.stats import anomalies, randomness
from megalab.stats.montecarlo import benjamini_hochberg, holm, mc_pvalue, simulate_draws


def test_exact_probabilities():
    assert cb.TOTAL_COMBINATIONS == 50_063_860
    pmf = cb.hits_pmf()
    assert pmf[6] == pytest.approx(1 / 50_063_860)
    assert pmf[5] == pytest.approx(6 * 54 / 50_063_860)
    assert pmf[4] == pytest.approx(15 * comb(54, 2) / 50_063_860)
    assert (np.arange(7) * pmf).sum() == pytest.approx(0.6)


def test_sum_of_iid_matches_direct_convolution():
    pmf = cb.hits_pmf()
    direct = np.array([1.0])
    for _ in range(13):
        direct = np.convolve(direct, pmf)
    np.testing.assert_allclose(cb.sum_of_iid_pmf(pmf, 13), direct, atol=1e-14)


def test_elementary_symmetric_small_case():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    e = cb.elementary_symmetric(w, k=3)[0]
    assert list(e) == [1, 10, 35, 50]


def test_uniform_model_has_zero_bits():
    p = np.full((5, 60), 0.1)
    d = simulate_draws(5, np.random.default_rng(0))[0]
    np.testing.assert_allclose(metrics.bits(p, d), 0, atol=1e-9)


def test_set_distribution_sums_to_one_small():
    """P(S) = Π w / e_k soma 1 sobre todos os subconjuntos (n=8, k=3)."""
    from itertools import combinations
    rng = np.random.default_rng(1)
    w = rng.random(8) + 0.1
    e3 = cb.elementary_symmetric(w, k=3)[0, 3]
    total = sum(np.prod(w[list(s)]) for s in combinations(range(8), 3))
    assert total / e3 == pytest.approx(1.0)


def test_llr_null_moments_match_monte_carlo():
    rng = np.random.default_rng(5)
    p = normalize(0.1 + 0.08 * (rng.random(60) - 0.5))[None, :]
    mu, var = cb.set_llr_null_moments(p)
    sims = simulate_draws(40000, rng)[0] - 1
    llr = cb.set_log_likelihood_ratio(np.repeat(p, len(sims), axis=0), sims)
    assert llr.mean() == pytest.approx(mu[0], abs=4 * np.sqrt(var[0] / len(sims)))
    assert llr.var() == pytest.approx(var[0], rel=0.05)
    assert mu[0] < 0  # desigualdade de Gibbs: sob H0 nenhum modelo ganha em média


def test_evalue_is_supermartingale_under_null():
    rng = np.random.default_rng(11)
    p = np.tile(normalize(0.1 + 0.1 * (rng.random(60) - 0.5)), (1, 1))
    # E[LR] = 1 por concurso sob H0 (P_modelo é uma distribuição válida sobre conjuntos)
    d = simulate_draws(50000, rng)[0]
    lr = np.exp(cb.set_log_likelihood_ratio(np.repeat(p, len(d), axis=0), d - 1))
    assert lr.mean() == pytest.approx(1.0, abs=0.03)
    # Desigualdade de Ville: P(max_t E_t ≥ 20) ≤ 1/20
    hits_max = 0
    for _ in range(400):
        d = simulate_draws(50, rng)[0]
        hits_max += metrics.log10_evalue(np.repeat(p, 50, axis=0), d).max() >= np.log10(20)
    assert hits_max / 400 <= 0.05


def test_hits_test_null_is_calibrated():
    """Estratégia qualquer (freq.) em sorteios justos: p < 0,05 em ~5 % dos casos."""
    rng = np.random.default_rng(3)
    rej = 0
    for _ in range(300):
        d = simulate_draws(200, rng)[0]
        tickets = np.tile(np.arange(1, 7), (200, 1))
        h = metrics.hits(tickets, d)
        rej += metrics.hits_test(h)["p_unilateral"] < 0.05
    assert rej / 300 < 0.09


def test_simulator_is_uniform_and_order_is_uniform():
    rng = np.random.default_rng(4)
    s, o = simulate_draws(30000, rng)
    counts = np.bincount(s.ravel() - 1, minlength=60)
    assert counts.min() > 0.9 * 3000 and counts.max() < 1.1 * 3000
    first = np.bincount(o[:, 0] - 1, minlength=60)
    assert first.min() > 0.8 * 500 and first.max() < 1.2 * 500
    assert np.all(np.diff(s, axis=1) > 0)


def test_biased_simulator_increases_frequency():
    rng = np.random.default_rng(6)
    w = np.ones(60)
    w[0] = 1.5
    s, _ = simulate_draws(20000, rng, w)
    c = np.bincount(s.ravel() - 1, minlength=60)
    assert c[0] > 1.3 * c[1:].mean()


def test_battery_detects_strong_bias_and_not_fair():
    rng = np.random.default_rng(8)
    fair = simulate_draws(1500, rng)
    st = randomness.statistics(*fair)
    w = np.ones(60)
    w[10] = 1.6
    biased = simulate_draws(1500, rng, w)
    sb = randomness.statistics(*biased)
    assert sb["H01"] > st["H01"] and sb["H02"] > 5


def test_battery_null_pvalues_roughly_uniform():
    """Calibração: em um histórico justo, nenhum p-valor de MC deve ser extremo
    com frequência acima do esperado (teste leve, 40 réplicas)."""
    rng = np.random.default_rng(12)
    d, o = simulate_draws(600, rng)
    res = randomness.run_battery(d, o, 40, rng)
    ps = np.array([v for v in res.pvalues.values() if np.isfinite(v)])
    assert (ps < 0.05).sum() <= 4


def test_factor_test_permutation_null():
    import pandas as pd
    rng = np.random.default_rng(13)
    d, _ = simulate_draws(800, rng)
    X = randomness.onehot(d)
    lab = pd.Series(rng.choice(["a", "b"], size=800))
    r = anomalies.factor_test(X, lab, 200, rng)
    assert 0 < r["p"] <= 1 and set(r["grupos"]) == {"a", "b"}


def test_consecutive_formula():
    # nº de 6-subconjuntos com j adjacências soma C(60,6)
    tot = sum(comb(5, j) * comb(55, 6 - j) for j in range(6))
    assert tot == comb(60, 6)


def test_multiple_testing_helpers():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    q = benjamini_hochberg(np.array([0.01, 0.02, 0.03, 0.5]))
    assert q[0] == pytest.approx(0.04) and q[3] == pytest.approx(0.5)
    assert mc_pvalue(5.0, np.array([1, 2, 3, 6])) == pytest.approx(2 / 5)


def test_ticket_tie_break_is_not_biased_to_high_numbers():
    rng = np.random.default_rng(0)
    p = np.full(60, 0.1)
    picks = np.concatenate([ticket_from_proba(p, rng) for _ in range(3000)])
    c = np.bincount(picks - 1, minlength=60)
    assert c[:30].sum() == pytest.approx(c[30:].sum(), rel=0.08)


def test_scores_to_proba_properties():
    p = scores_to_proba(np.arange(60) + 1.0)
    assert abs(p.sum() - 6) < 1e-9 and p.min() > 0 and p.max() < 1
    assert np.all(np.diff(p) > 0)
