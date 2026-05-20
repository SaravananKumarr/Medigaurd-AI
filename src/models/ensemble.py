"""
src/models/ensemble.py
Supervised ensemble classifier combining Random Forest + XGBoost.
Trained on labeled data to identify specific attack types.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed — using Random Forest only. pip install xgboost")


class EnsembleDetector:
    """
    Voting ensemble of Random Forest + XGBoost for attack classification.

    Supports both binary (attack / normal) and multi-class (attack type) modes.
    """

    def __init__(self, mode: str = "binary", random_state: int = 42):
        assert mode in ("binary", "multiclass"), "mode must be 'binary' or 'multiclass'"
        self.mode = mode
        self.classes_ = None

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )

        if XGB_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
            self.model = VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb)],
                voting="soft",
            )
        else:
            self.model = rf  # fallback

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "EnsembleDetector":
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def attack_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability of the most likely attack class.
        For binary mode: P(attack). For multiclass: max P(any attack type).
        """
        proba = self.predict_proba(X)
        if self.mode == "binary":
            attack_idx = list(self.classes_).index(1) if 1 in self.classes_ else -1
            return proba[:, attack_idx] if attack_idx >= 0 else proba[:, -1]
        else:
            # exclude index of 'normal' class
            normal_idx = list(self.classes_).index("normal") \
                         if "normal" in self.classes_ else None
            if normal_idx is not None:
                attack_proba = np.delete(proba, normal_idx, axis=1)
                return attack_proba.max(axis=1)
            return proba.max(axis=1)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = self.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds)

        result = {
            "classification_report": report,
            "confusion_matrix":      cm.tolist(),
        }

        if self.mode == "binary":
            proba = self.predict_proba(X_test)
            attack_idx = list(self.classes_).index(1) if 1 in self.classes_ else -1
            if attack_idx >= 0:
                result["roc_auc"] = round(roc_auc_score(y_test, proba[:, attack_idx]), 4)

        return result

    def feature_importances(self, feature_names: list) -> dict:
        """Return feature importances from the Random Forest component."""
        rf_model = None
        if isinstance(self.model, VotingClassifier):
            for name, est in self.model.estimators_:
                if name == "rf":
                    rf_model = est
                    break
        else:
            rf_model = self.model

        if rf_model is None or not hasattr(rf_model, "feature_importances_"):
            return {}

        importances = rf_model.feature_importances_
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True
        ))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"Ensemble model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "EnsembleDetector":
        return joblib.load(path)
