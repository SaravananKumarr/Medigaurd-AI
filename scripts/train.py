"""scripts/train.py
Train all MediGuard AI models on synthetic (or real) healthcare event data.

Usage:
    python scripts/train.py --model all --output models/saved/
    python scripts/train.py --model ensemble --data data/healthcare_events.csv
    python scripts/train.py --model lstm --seq_len 20 --epochs 30
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from src.data.simulator    import generate_dataset
from src.data.preprocessor import MediGuardPreprocessor, make_sequences
from src.models.isolation_forest import IFAnomalyDetector
from src.models.ensemble         import EnsembleDetector
from src.models.lstm_detector    import LSTMDetector


def train_isolation_forest(X_train, X_test, y_test, output_dir):
    print("\n--- Training Isolation Forest ---")
    model = IFAnomalyDetector(contamination=0.06, n_estimators=200)
    model.fit(X_train)
    results = model.evaluate(X_test, y_test)
    print(f"  ROC-AUC: {results['roc_auc']}")
    model.save(os.path.join(output_dir, "isolation_forest.pkl"))
    return model, results


def train_ensemble(X_train, y_train, X_test, y_test, output_dir):
    print("\n--- Training Ensemble (RF + XGBoost) ---")
    model = EnsembleDetector(mode="binary")
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    print(f"  Accuracy: {results['classification_report']['accuracy']:.4f}")
    print(f"  ROC-AUC:  {results.get('roc_auc', 'N/A')}")
    model.save(os.path.join(output_dir, "ensemble.pkl"))
    return model, results


def train_lstm(X_train, y_train, X_test, y_test, seq_len, epochs, output_dir):
    print("\n--- Training LSTM ---")
    try:
        X_train_seq, y_train_seq = make_sequences(X_train, y_train, seq_len)
        X_test_seq,  y_test_seq  = make_sequences(X_test,  y_test,  seq_len)

        n_features = X_train.shape[1]
        model = LSTMDetector(seq_len=seq_len, n_features=n_features)
        model.build()
        model.summary()

        split = int(len(X_train_seq) * 0.85)
        model.fit(
            X_train_seq[:split], y_train_seq[:split],
            X_train_seq[split:], y_train_seq[split:],
            epochs=epochs,
        )
        results = model.evaluate(X_test_seq, y_test_seq)
        print(f"  ROC-AUC: {results['roc_auc']}")
        model.save(os.path.join(output_dir, "lstm"))
        return model, results
    except ImportError as e:
        print(f"  Skipping LSTM: {e}")
        return None, {}


def main():
    parser = argparse.ArgumentParser(description="Train MediGuard AI models")
    parser.add_argument("--model",      choices=["all", "if", "ensemble", "lstm"],
                        default="all")
    parser.add_argument("--data",       type=str, default=None,
                        help="Path to CSV dataset. If not provided, synthetic data is generated.")
    parser.add_argument("--output",     type=str, default="models/saved/")
    parser.add_argument("--n_normal",   type=int, default=8000)
    parser.add_argument("--n_attack",   type=int, default=800)
    parser.add_argument("--seq_len",    type=int, default=20)
    parser.add_argument("--epochs",     type=int, default=25)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=== MediGuard AI — Model Training ===")

    if args.data:
        print(f"Loading dataset: {args.data}")
        df = pd.read_csv(args.data, parse_dates=["timestamp"])
    else:
        print(f"Generating synthetic dataset (n_normal={args.n_normal}, n_attack={args.n_attack})...")
        df = generate_dataset(n_normal=args.n_normal, n_per_attack=args.n_attack, seed=args.seed)

    print(f"Dataset: {len(df)} events, attack rate: {(df['label']=='attack').mean():.1%}")

    prep = MediGuardPreprocessor()
    X_train, X_test, y_train, y_test = prep.fit_transform(df, seed=args.seed)
    prep.save(os.path.join(args.output, "preprocessor.pkl"))

    print(f"Features: {len(prep.feature_cols_)}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    all_results = {}

    if args.model in ("all", "if"):
        _, res = train_isolation_forest(X_train, X_test, y_test, args.output)
        all_results["isolation_forest"] = res

    if args.model in ("all", "ensemble"):
        _, res = train_ensemble(X_train, y_train, X_test, y_test, args.output)
        all_results["ensemble"] = res

    if args.model in ("all", "lstm"):
        _, res = train_lstm(X_train, y_train, X_test, y_test,
                            args.seq_len, args.epochs, args.output)
        all_results["lstm"] = res

    results_path = os.path.join(args.output, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n=== Training complete. Results saved to {results_path} ===")


if __name__ == "__main__":
    main()
