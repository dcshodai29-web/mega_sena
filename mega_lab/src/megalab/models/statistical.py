"""Linha de base e modelos estatísticos (M00–M09), parâmetros pré-fixados."""

from __future__ import annotations

import numpy as np

from megalab.config import DIRICHLET_PRIOR, N_NUMBERS, P_NUMBER
from megalab.models.base import Model, scores_to_proba


def _onehot(history: np.ndarray) -> np.ndarray:
    X = np.zeros((len(history), N_NUMBERS), dtype=np.float64)
    if len(history):
        X[np.arange(len(history))[:, None], history - 1] = 1.0
    return X


def _counts(history: np.ndarray, window: int | None = None) -> np.ndarray:
    h = history if window is None else history[-window:]
    return np.bincount(h.ravel() - 1, minlength=N_NUMBERS).astype(float) if len(h) else np.zeros(N_NUMBERS)


class Uniform(Model):
    name = "aleatorio"
    description = "Linha de base: todas as dezenas com probabilidade 0,1"

    def predict_proba(self, history):
        return np.full(N_NUMBERS, P_NUMBER)


class FrequencyAll(Model):
    name = "freq_total"
    description = "Frequência acumulada de cada dezena"

    def predict_proba(self, history):
        return scores_to_proba(_counts(history) + 1)


class FrequencyWindow(Model):
    def __init__(self, window: int, name: str, description: str):
        self.window, self.name, self.description = window, name, description

    def predict_proba(self, history):
        return scores_to_proba(_counts(history, self.window) + 0.5)


class EWMA(Model):
    def __init__(self, half_life: float = 50, name: str = "ewma_hl50"):
        self.half_life, self.name = half_life, name
        self.description = f"Frequência com decaimento exponencial (meia-vida {half_life:g})"

    def predict_proba(self, history):
        t = len(history)
        if t == 0:
            return np.full(N_NUMBERS, P_NUMBER)
        w = 0.5 ** (np.arange(t - 1, -1, -1) / self.half_life)
        return scores_to_proba(w @ _onehot(history) + 1e-3)


class Cold(Model):
    name = "frios"
    description = "Inverso da frequência acumulada (aposta na 'compensação')"

    def predict_proba(self, history):
        return scores_to_proba(1.0 / (_counts(history) + 1))


class Overdue(Model):
    name = "atrasados"
    description = "Proporcional ao atraso atual (falácia do apostador, testada)"

    def predict_proba(self, history):
        t = len(history)
        last = np.full(N_NUMBERS, -1)
        for r in range(t):
            last[history[r] - 1] = r
        gap = np.where(last >= 0, t - 1 - last, t)
        return scores_to_proba(gap + 1.0)


class Dirichlet(Model):
    name = "dirichlet"
    description = f"Média a posteriori Dirichlet-multinomial (prior {DIRICHLET_PRIOR:g} por dezena)"

    def predict_proba(self, history):
        return scores_to_proba(_counts(history) + DIRICHLET_PRIOR)


class Markov1(Model):
    name = "markov1"
    description = "Transições dezena→dezena: o que costuma vir depois do concurso anterior"

    def predict_proba(self, history):
        t = len(history)
        if t < 2:
            return np.full(N_NUMBERS, P_NUMBER)
        X = _onehot(history)
        # tr_i = Σ_{r=1}^{t-1} (X[t-1]·X[r-1]) X[r,i]
        ov = X[:-1] @ X[-1]
        return scores_to_proba(ov @ X[1:] + 1.0)


class Affinity(Model):
    name = "afinidade"
    description = "Co-ocorrência histórica com as dezenas do concurso anterior"

    def predict_proba(self, history):
        t = len(history)
        if t < 2:
            return np.full(N_NUMBERS, P_NUMBER)
        X = _onehot(history)
        ov = X[:-1] @ X[-1]
        return scores_to_proba(ov @ X[:-1] + 1.0)


def statistical_models() -> list[Model]:
    return [
        Uniform(),
        FrequencyAll(),
        FrequencyWindow(100, "freq_janela100", "Frequência nos últimos 100 concursos"),
        EWMA(50),
        FrequencyWindow(20, "quentes20", "Dezenas 'quentes': frequência nos últimos 20"),
        Cold(),
        Overdue(),
        Dirichlet(),
        Markov1(),
        Affinity(),
    ]
