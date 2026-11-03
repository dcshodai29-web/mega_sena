"""Modelos de machine learning (M10, M11) com re-treino periódico.

Treino: linhas ``F[s]`` → rótulo ``Y[s]`` para ``WARMUP ≤ s < t``; previsão:
``F[t]``. O modelo é re-treinado quando ``t`` cruza um múltiplo de
``ML_REFIT_EVERY``; entre re-treinos usa o último ajuste (sempre feito com
dados anteriores ao concurso previsto).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from megalab.config import ML_REFIT_EVERY, N_NUMBERS, P_NUMBER
from megalab.models.base import Model, normalize
from megalab.models.features import feature_tensor, labels

WARMUP = 50


class MLModel(Model):
    def __init__(self, name: str, description: str, factory, refit_every: int = ML_REFIT_EVERY):
        self.name, self.description = name, description
        self._factory = factory
        self.refit_every = refit_every
        self._fitted_at: int | None = None
        self._clf = None

    def reset(self):
        self._fitted_at, self._clf = None, None

    def _fit(self, F: np.ndarray, Y: np.ndarray, t: int):
        rows = slice(WARMUP, t)
        Xtr = F[rows].reshape(-1, F.shape[2])
        ytr = Y[rows].reshape(-1)
        clf = self._factory()
        clf.fit(Xtr, ytr)
        self._clf, self._fitted_at = clf, t

    def predict_from_features(self, F: np.ndarray, Y: np.ndarray, t: int) -> np.ndarray:
        """Previsão para o concurso de índice ``t`` usando F[:t+1] e Y[:t]."""
        if F.shape[0] != t + 1 or Y.shape[0] != t:
            raise ValueError("Use exatamente F[:t+1] e Y[:t] — nada além do passado.")
        if t < WARMUP + 20:
            return np.full(N_NUMBERS, P_NUMBER)
        bucket = (t // self.refit_every) * self.refit_every
        if self._clf is None or self._fitted_at is None or self._fitted_at // self.refit_every != t // self.refit_every:
            self._fit(F, Y, max(bucket, WARMUP + 20))
        q = self._clf.predict_proba(F[t])[:, 1]
        return normalize(q)

    def predict_proba(self, history: np.ndarray) -> np.ndarray:
        t = len(history)
        F = feature_tensor(history)
        Y = labels(history)
        self.reset()  # caminho autônomo: treina com todo o histórico disponível
        if t < WARMUP + 20:
            return np.full(N_NUMBERS, P_NUMBER)
        self._fit(F, Y, t)
        return normalize(self._clf.predict_proba(F[t])[:, 1])


def ml_models(seed: int = 0) -> list[MLModel]:
    return [
        MLModel("logistica", "Regressão logística (L2) sobre 10 features causais",
                lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
        MLModel("gboost", "Gradient boosting (histograma) sobre as mesmas features",
                lambda: HistGradientBoostingClassifier(
                    max_iter=100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                    early_stopping=False, random_state=seed)),
    ]
