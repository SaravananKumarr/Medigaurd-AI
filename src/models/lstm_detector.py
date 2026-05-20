"""
src/models/lstm_detector.py
LSTM-based sequence anomaly detector for MediGuard AI.
Captures temporal attack patterns invisible to single-event classifiers.
"""

import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. pip install tensorflow")


class LSTMDetector:
    """
    Bidirectional LSTM for sequential threat detection.

    Architecture:
        Input  → LSTM(128) → Dropout(0.3) → LSTM(64) → BN → Dense(32) → Dense(1, sigmoid)

    Why LSTM for healthcare security?
    - Slow brute-force attacks span hundreds of events over hours
    - Data exfiltration builds gradually across sessions
    - Normal EHR access has temporal patterns (shift start/end spikes)
    The LSTM sees 20-event windows and learns these temporal signatures.
    """

    def __init__(self, seq_len: int = 20, n_features: int = 20,
                 lstm_units: int = 128, dropout: float = 0.3):
        self.seq_len    = seq_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout    = dropout
        self.model_     = None
        self.history_   = None

    def build(self) -> "LSTMDetector":
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTMDetector.")

        model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.seq_len, self.n_features),
                 return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.lstm_units // 2, return_sequences=False),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(self.dropout / 2),
            Dense(1, activation="sigmoid"),
        ], name="MediGuard_LSTM")

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        self.model_ = model
        return self

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 30, batch_size: int = 256) -> "LSTMDetector":
        """
        X_train shape: (N, seq_len, n_features)
        y_train shape: (N,) binary
        """
        if self.model_ is None:
            self.n_features = X_train.shape[2]
            self.build()

        callbacks = [
            EarlyStopping(monitor="val_loss" if X_val is not None else "loss",
                          patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        ]

        val_data = (X_val, y_val) if X_val is not None else None
        self.history_ = self.model_.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1,
        )
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (0 = normal, 1 = attack)."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(attack) for each sequence window."""
        if self.model_ is None:
            raise RuntimeError("Model not built. Call build() and fit() first.")
        return self.model_.predict(X, verbose=0).flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
        proba = self.predict_proba(X_test)
        preds = (proba >= 0.5).astype(int)
        report = classification_report(y_test, preds,
                                       target_names=["normal", "attack"],
                                       output_dict=True, zero_division=0)
        return {
            "classification_report": report,
            "roc_auc":               round(roc_auc_score(y_test, proba), 4),
            "confusion_matrix":      confusion_matrix(y_test, preds).tolist(),
        }

    def summary(self):
        if self.model_:
            self.model_.summary()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model_.save(os.path.join(path, "lstm_model.keras"))
        import json
        meta = {"seq_len": self.seq_len, "n_features": self.n_features,
                "lstm_units": self.lstm_units, "dropout": self.dropout}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)
        print(f"LSTM model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "LSTMDetector":
        import json
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        detector = cls(**meta)
        detector.model_ = load_model(os.path.join(path, "lstm_model.keras"))
        return detector
