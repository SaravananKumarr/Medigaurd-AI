"""
src/detection/engine.py
MediGuard AI detection orchestrator.
Combines Isolation Forest + Ensemble + LSTM scores into a unified threat score.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime

from src.detection.alert import AlertEngine, Alert


@dataclass
class ThreatEvent:
    """A single detected threat event with full context."""
    timestamp:       str
    src_ip:          str
    dst_ip:          str
    attack_type:     str          # predicted class
    confidence:      float        # 0-1 ensemble confidence
    anomaly_score:   float        # Isolation Forest score
    lstm_score:      float        # LSTM P(attack)
    combined_score:  float        # weighted fusion
    severity:        str          # LOW / MEDIUM / HIGH / CRITICAL
    raw_features:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Weights for score fusion
SCORE_WEIGHTS = {
    "ensemble":   0.50,
    "isolation":  0.25,
    "lstm":       0.25,
}


def _fuse_scores(ensemble: float, isolation: float, lstm: float) -> float:
    return (SCORE_WEIGHTS["ensemble"]  * ensemble +
            SCORE_WEIGHTS["isolation"] * isolation +
            SCORE_WEIGHTS["lstm"]      * lstm)


def _severity_from_score(score: float) -> str:
    if score < 0.35:
        return "LOW"
    elif score < 0.55:
        return "MEDIUM"
    elif score < 0.75:
        return "HIGH"
    return "CRITICAL"


class DetectionEngine:
    """
    Orchestrates all three models and produces ThreatEvents.

    Usage:
        engine = DetectionEngine(preprocessor, if_model, ensemble_model, lstm_model)
        threats = engine.analyze(df)
    """

    def __init__(self, preprocessor, if_model, ensemble_model, lstm_model=None,
                 alert_threshold: float = 0.4, seq_len: int = 20):
        self.preprocessor     = preprocessor
        self.if_model         = if_model
        self.ensemble_model   = ensemble_model
        self.lstm_model       = lstm_model
        self.alert_threshold  = alert_threshold
        self.seq_len          = seq_len
        self.alert_engine     = AlertEngine()
        self._event_buffer: List[np.ndarray] = []

    def analyze(self, df: pd.DataFrame) -> List[ThreatEvent]:
        """
        Analyze a batch of events and return detected ThreatEvents.

        Args:
            df: DataFrame in the same schema as simulator output.

        Returns:
            List of ThreatEvent objects above the alert threshold.
        """
        X = self.preprocessor.transform(df)

        if_scores  = self.if_model.predict_proba_anomaly(X)
        ens_confs  = self.ensemble_model.attack_confidence(X)
        ens_preds  = self.ensemble_model.predict(X)

        lstm_scores = np.zeros(len(X))
        if self.lstm_model is not None:
            self._event_buffer.extend(list(X))
            if len(self._event_buffer) >= self.seq_len:
                seqs = np.array([
                    self._event_buffer[i : i + self.seq_len]
                    for i in range(len(X))
                    if i + self.seq_len <= len(self._event_buffer)
                ], dtype=np.float32)
                if len(seqs):
                    scores = self.lstm_model.predict_proba(seqs)
                    lstm_scores[-len(scores):] = scores
                self._event_buffer = self._event_buffer[-self.seq_len:]

        threats = []
        for i in range(len(X)):
            combined = _fuse_scores(ens_confs[i], if_scores[i], lstm_scores[i])
            if combined < self.alert_threshold:
                continue

            row = df.iloc[i]
            event = ThreatEvent(
                timestamp      = str(row.get("timestamp", datetime.now())),
                src_ip         = str(row.get("src_ip",    "unknown")),
                dst_ip         = str(row.get("dst_ip",    "unknown")),
                attack_type    = str(ens_preds[i]),
                confidence     = round(float(ens_confs[i]), 4),
                anomaly_score  = round(float(if_scores[i]),  4),
                lstm_score     = round(float(lstm_scores[i]),4),
                combined_score = round(float(combined),      4),
                severity       = _severity_from_score(combined),
                raw_features   = {col: float(row[col])
                                  for col in self.preprocessor.feature_cols_
                                  if col in row.index},
            )
            threats.append(event)
            self.alert_engine.dispatch(self._to_alert(event))

        return threats

    def _to_alert(self, event: ThreatEvent) -> "Alert":
        return Alert(
            timestamp      = event.timestamp,
            severity       = event.severity,
            attack_type    = event.attack_type,
            src_ip         = event.src_ip,
            dst_ip         = event.dst_ip,
            combined_score = event.combined_score,
            message        = (f"{event.severity} threat detected: "
                              f"{event.attack_type} from {event.src_ip} "
                              f"(score={event.combined_score:.2f})"),
        )
