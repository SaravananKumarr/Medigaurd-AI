"""
src/data/preprocessor.py
Feature engineering pipeline for MediGuard AI.
Handles sliding window stats, encoding, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


NUMERIC_FEATURES = [
    "bytes_out", "bytes_in", "duration_ms", "failed_logins",
    "records_accessed", "file_write_rate", "entropy",
    "unique_dst_ips", "port_scan_score", "device_ping_rate",
    "dst_is_external", "access_hour",
]

WINDOW_FEATURES = ["bytes_out", "failed_logins", "file_write_rate", "unique_dst_ips"]


def add_window_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling statistics per source IP as temporal context features."""
    df = df.sort_values("timestamp").copy()
    for feat in WINDOW_FEATURES:
        df[f"{feat}_roll_mean"] = (
            df.groupby("src_ip")[feat]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{feat}_roll_std"] = (
            df.groupby("src_ip")[feat]
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
    return df


def encode_protocol(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the protocol column."""
    return pd.get_dummies(df, columns=["protocol"], prefix="proto")


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all numeric feature column names after preprocessing."""
    base = NUMERIC_FEATURES.copy()
    window_cols = [f"{f}_roll_mean" for f in WINDOW_FEATURES] + \
                  [f"{f}_roll_std"  for f in WINDOW_FEATURES]
    proto_cols  = [c for c in df.columns if c.startswith("proto_")]
    cmd_col     = ["cmd_anomaly"] if "cmd_anomaly" in df.columns else []
    return [c for c in base + window_cols + proto_cols + cmd_col if c in df.columns]


class MediGuardPreprocessor:
    """
    End-to-end preprocessing pipeline.

    Usage:
        prep = MediGuardPreprocessor()
        X_train, X_test, y_train, y_test = prep.fit_transform(df)
        X_new = prep.transform(new_df)
        prep.save("models/saved/preprocessor.pkl")
    """

    def __init__(self):
        self.scaler         = StandardScaler()
        self.label_encoder  = LabelEncoder()
        self.feature_cols_  = None
        self.is_fitted_     = False

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_window_features(df)
        df = encode_protocol(df)
        return df

    def fit_transform(self, df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        """
        Fit the scaler on training data and return train/test splits.

        Returns:
            X_train, X_test, y_train, y_test (numpy arrays)
        """
        df = self._preprocess(df.copy())
        self.feature_cols_ = get_feature_columns(df)

        X = df[self.feature_cols_].values.astype(np.float32)
        y = self.label_encoder.fit_transform(df["label"].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        self.scaler.fit(X_train)
        self.is_fitted_ = True

        return (
            self.scaler.transform(X_train),
            self.scaler.transform(X_test),
            y_train,
            y_test,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted scaler."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit_transform first.")
        df = self._preprocess(df.copy())
        for col in self.feature_cols_:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_cols_].values.astype(np.float32)
        return self.scaler.transform(X)

    def get_attack_features(self, df: pd.DataFrame) -> tuple:
        """
        Return (X, y_multiclass) for attack-type classification.
        y is encoded attack_type strings.
        """
        df = self._preprocess(df.copy())
        X = df[self.feature_cols_].values.astype(np.float32)
        y = df["attack_type"].values
        return self.scaler.transform(X), y

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved: {path}")

    @classmethod
    def load(cls, path: str) -> "MediGuardPreprocessor":
        return joblib.load(path)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 20):
    """
    Convert flat feature matrix into 3-D sequences for LSTM training.

    Returns:
        X_seq: (N - seq_len, seq_len, features)
        y_seq: (N - seq_len,) — label of last step in each window
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys)
