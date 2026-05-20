"""
src/models/isolation_forest.py
Unsupervised anomaly detection using Isolation Forest.
Works without labels — ideal for zero-day threat detection.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os


class IFAnomalyDetector:
    """
    Isolation Forest wrapper tuned for healthcare network anomaly detection.

    Key idea: Anomalies are isolated in fewer splits than normal traffic.
    contamination = expected proportion of attacks in training data.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 max_features: float = 0.8, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self.threshold_ = None

    def fit(self, X_train: np.ndarray) -> "IFAnomalyDetector":
        self.model.fit(X_train)
        # Store the raw anomaly scores for threshold calibration
        scores = self.model.decision_function(X_train)
        self.threshold_ = np.percentile(scores, 5)  # bottom 5% = anomalous
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions: 1 = normal, -1 = anomaly (sklearn convention)."""
        return self.model.predict(X)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return continuous anomaly scores. Lower = more anomalous.
        Negated so that higher = more suspicious (intuitive for alerting).
        """
        return -self.model.decision_function(X)

    def predict_proba_anomaly(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize anomaly scores to [0, 1] range.
        0 = certainly normal, 1 = certainly anomalous.
        """
        scores = self.anomaly_scores(X)
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.zeros(len(scores))
        return (scores - min_s) / (max_s - min_s)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate against ground truth labels.
        y_test: 0 = normal, 1 = attack
        """
        from sklearn.metrics import (classification_report, roc_auc_score,
                                     confusion_matrix)
        preds_raw = self.predict(X_test)
        preds = np.where(preds_raw == -1, 1, 0)  # -1 → 1 (attack)

        scores = self.predict_proba_anomaly(X_test)
        report = classification_report(y_test, preds, target_names=["normal", "attack"],
                                       output_dict=True)
        auc = roc_auc_score(y_test, scores)
        cm  = confusion_matrix(y_test, preds)

        return {
            "classification_report": report,
            "roc_auc":               round(auc, 4),
            "confusion_matrix":      cm.tolist(),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"IsolationForest model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "IFAnomalyDetector":
        return joblib.load(path)
