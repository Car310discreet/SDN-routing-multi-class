"""
ml_classifier.py - the trained 4-class traffic classifier used for path selection.

The saved bundle (geant2/models/best_model.joblib, written by train_models.py) holds the fitted
model, its name, the feature columns it was trained on and the class names, so the controller always
feeds the features in training order.

Feature sets:
  all     end-to-end probe features + controller (passive) link utilisation   (default)
  active  end-to-end probe features only, like the AIRSDN baseline
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry import ACTIVE_FEATURES, FEATURE_NAMES  # noqa: E402

GEANT2_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(GEANT2_DIR, "models", "best_model.joblib")
FEATURE_SETS = {"all": FEATURE_NAMES, "active": ACTIVE_FEATURES}


class CongestionClassifier:
    def __init__(self, bundle: dict):
        self.model, self.name = bundle["model"], bundle["name"]
        self.features, self.classes = list(bundle["features"]), list(bundle["classes"])

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL) -> "CongestionClassifier":
        import joblib
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path} (run geant2/train_models.py first)")
        return cls(joblib.load(path))

    def predict(self, features: dict) -> tuple[int, np.ndarray, float]:
        """(traffic class 0-3, class probabilities, inference time in ms) for one path."""
        row = np.asarray([[features[name] for name in self.features]], dtype=float)
        if not np.all(np.isfinite(row)):
            raise ValueError(f"Non-finite feature vector: {features}")
        start = time.perf_counter()
        probabilities = self.model.predict_proba(row)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return int(np.argmax(probabilities)), probabilities, elapsed_ms

    @staticmethod
    def expected_class(probabilities: np.ndarray) -> float:
        """Probability-weighted traffic class: breaks ties between paths with the same predicted class."""
        return float(np.dot(probabilities, np.arange(len(probabilities))))
