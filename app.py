"""
app.py
MediGuard AI — Flask web application.
REST API + real-time threat monitoring dashboard.
"""

import os
import sys
import json
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="dashboard/templates")

# ── In-memory threat log ──────────────────────────────────────────────────────
_threats = []
_stats   = {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
_lock    = threading.Lock()

# ── Load models (lazy — only if saved models exist) ───────────────────────────
_engine = None

def _try_load_engine():
    global _engine
    saved = "models/saved"
    try:
        from src.data.preprocessor       import MediGuardPreprocessor
        from src.models.isolation_forest  import IFAnomalyDetector
        from src.models.ensemble          import EnsembleDetector
        from src.detection.engine         import DetectionEngine

        if not os.path.exists(os.path.join(saved, "preprocessor.pkl")):
            print("No saved models found — run scripts/train.py first.")
            return

        prep     = MediGuardPreprocessor.load(os.path.join(saved, "preprocessor.pkl"))
        if_model = IFAnomalyDetector.load(os.path.join(saved, "isolation_forest.pkl"))
        ens      = EnsembleDetector.load(os.path.join(saved, "ensemble.pkl"))

        lstm = None
        try:
            from src.models.lstm_detector import LSTMDetector
            lstm_path = os.path.join(saved, "lstm")
            if os.path.exists(lstm_path):
                lstm = LSTMDetector.load(lstm_path)
        except Exception:
            pass

        _engine = DetectionEngine(prep, if_model, ens, lstm)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Could not load models: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_loaded": _engine is not None,
                    "timestamp": datetime.now().isoformat()})


@app.route("/api/threats", methods=["GET"])
def get_threats():
    n = int(request.args.get("n", 50))
    with _lock:
        return jsonify({"threats": _threats[-n:], "stats": _stats})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze a batch of events from a JSON payload.
    POST body: {"events": [...]}  — list of event dicts matching simulator schema.
    """
    if _engine is None:
        return jsonify({"error": "Models not loaded. Run scripts/train.py first."}), 503

    data = request.get_json(force=True)
    events = data.get("events", [])
    if not events:
        return jsonify({"error": "No events provided."}), 400

    try:
        df = pd.DataFrame(events)
        df["timestamp"] = pd.to_datetime(df.get("timestamp", datetime.now()))
        threats = _engine.analyze(df)

        with _lock:
            for t in threats:
                _threats.append(t.to_dict())
                sev = t.severity.lower()
                _stats["total"] += 1
                if sev in _stats:
                    _stats[sev] += 1

        return jsonify({
            "analyzed":    len(events),
            "threats":     len(threats),
            "detections":  [t.to_dict() for t in threats],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Generate and analyze synthetic attack data."""
    from src.data.simulator import generate_dataset

    body     = request.get_json(force=True) or {}
    scenario = body.get("scenario", "all")
    n        = int(body.get("n", 200))

    attack_types = None if scenario == "all" else [scenario]
    df = generate_dataset(n_normal=n, n_per_attack=n // 6 or 30,
                          attack_types=attack_types, seed=int(time.time()))

    # Return stats even without models
    attack_rate = float((df["label"] == "attack").mean())
    sample = df.head(10).to_dict(orient="records")

    return jsonify({
        "generated": len(df),
        "attack_rate": round(attack_rate, 3),
        "distribution": df["attack_type"].value_counts().to_dict(),
        "sample": sample,
    })


@app.route("/api/stats")
def stats():
    with _lock:
        return jsonify(_stats)


# ── Background demo feed (generates synthetic events every 5s if no models) ──

def _demo_feed():
    """Pushes synthetic events to the in-memory log for demo purposes."""
    import random
    attack_types = ["ransomware", "brute_force", "lateral_movement",
                    "ehr_unauthorized", "data_exfiltration", "normal"]
    severities   = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    while True:
        time.sleep(5)
        if _engine is not None:
            continue  # real engine handles events
        n_events = random.randint(0, 3)
        with _lock:
            for _ in range(n_events):
                attack = random.choice(attack_types)
                if attack == "normal":
                    continue
                sev = random.choice(severities)
                event = {
                    "timestamp":      datetime.now().isoformat(),
                    "src_ip":         f"192.168.{random.randint(1,40)}.{random.randint(2,254)}",
                    "dst_ip":         f"192.168.30.{random.randint(2,50)}",
                    "attack_type":    attack,
                    "confidence":     round(random.uniform(0.55, 0.99), 3),
                    "anomaly_score":  round(random.uniform(0.4,  0.95), 3),
                    "lstm_score":     round(random.uniform(0.3,  0.90), 3),
                    "combined_score": round(random.uniform(0.45, 0.97), 3),
                    "severity":       sev,
                    "message":        f"{sev} {attack} from 192.168.x.x",
                }
                _threats.append(event)
                _stats["total"] += 1
                _stats[sev.lower()] += 1
        if len(_threats) > 500:
            _threats[:] = _threats[-500:]


if __name__ == "__main__":
    _try_load_engine()
    t = threading.Thread(target=_demo_feed, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
