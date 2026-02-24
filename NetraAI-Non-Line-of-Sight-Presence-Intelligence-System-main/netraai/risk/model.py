from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from netraai.common.logging import get_logger
from .features import build_feature_frame

logger = get_logger(__name__)


@dataclass
class RiskModel:
    model: RandomForestClassifier
    threshold_high: float = 0.70

    @staticmethod
    def fresh(threshold_high: float = 0.70, random_state: int = 42) -> "RiskModel":
        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        return RiskModel(model=rf, threshold_high=threshold_high)

    def fit(self, raw_df, y):
        X = build_feature_frame(raw_df)
        self.model.fit(X, y)
        logger.info("Trained risk model on %d samples", len(X))

    def predict_proba_high(self, raw_df) -> np.ndarray:
        X = build_feature_frame(raw_df)
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        if "High" in classes:
            idx = classes.index("High")
            return proba[:, idx]
        # fallback: max class probability
        return proba.max(axis=1)

    def predict(self, raw_df) -> Tuple[np.ndarray, np.ndarray]:
        score = self.predict_proba_high(raw_df)

        labels = np.where(
            score >= self.threshold_high,
            "High",
            np.where(score >= 0.40, "Medium", "Low"),
        )
        return labels, score

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "threshold_high": self.threshold_high}, path)
        logger.info("Saved model -> %s", path)

    @staticmethod
    def load(path: str) -> "RiskModel":
        payload = joblib.load(path)
        return RiskModel(
            model=payload["model"],
            threshold_high=float(payload.get("threshold_high", 0.70)),
        )
