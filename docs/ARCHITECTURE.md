# MediGuard AI — Architecture

## Overview

MediGuard AI is a multi-layer threat detection system designed for healthcare network environments. It processes events from four data sources and fuses signals from three independent ML models to produce threat scores with automated response hooks.

---

## Component Breakdown

### Layer 1: Data Ingestion

| Source | Format | Features Extracted |
|--------|--------|--------------------|
| Network traffic | PCAP / NetFlow / JSON | bytes_out, bytes_in, duration_ms, protocol, port |
| EHR access logs | Syslog / CSV / API | failed_logins, records_accessed, access_hour |
| System events | Windows Event Log / Syslog | file_write_rate, entropy, cmd_anomaly |
| IoMT devices | REST API / MQTT | device_ping_rate, dst_is_external, port_scan_score |

### Layer 2: Feature Engineering

The `MediGuardPreprocessor` performs:
1. **Rolling window statistics** — 5-event rolling mean and std per source IP for `bytes_out`, `failed_logins`, `file_write_rate`, `unique_dst_ips`
2. **Protocol encoding** — one-hot encoding of TCP/UDP/HTTPS
3. **Standard scaling** — `StandardScaler` fit on training data only (no data leakage)
4. **Sequence construction** — for LSTM: sliding windows of 20 consecutive events per source IP

### Layer 3: Detection Models

#### Isolation Forest (unsupervised)
- No labels required at training time
- Detects zero-day / novel attack patterns
- Produces a continuous anomaly score in [0, 1]
- Weight in fusion: 25%

#### Ensemble Classifier (supervised)
- Random Forest (300 trees) + XGBoost (300 rounds), soft voting
- Trained on labeled attack/normal data
- Produces attack probability and attack type classification
- Weight in fusion: 50%

#### LSTM Sequence Model (supervised, temporal)
- Architecture: LSTM(128) → Dropout(0.3) → LSTM(64) → BN → Dense(32) → Dense(1)
- Trained on 20-event windows to capture temporal patterns
- Detects slow-burn attacks invisible to single-event models
- Weight in fusion: 25%

### Layer 4: Score Fusion

```
combined_score = 0.50 × ensemble_confidence
               + 0.25 × isolation_forest_score
               + 0.25 × lstm_score
```

Severity thresholds:
- `combined_score < 0.35` → LOW
- `0.35 ≤ score < 0.55` → MEDIUM
- `0.55 ≤ score < 0.75` → HIGH
- `score ≥ 0.75` → CRITICAL

### Layer 5: Alert & Response

- All alerts are logged to in-memory store (pluggable to SIEM)
- HIGH and CRITICAL alerts trigger automated responses
- Response actions are pluggable callables (see `AlertEngine.register_hook()`)

---

## Data Flow

```
Raw events (CSV / API / stream)
    ↓
MediGuardPreprocessor.transform()
    ↓
┌──────────────┬──────────────┬─────────────┐
│  Isolation   │   Ensemble   │    LSTM     │
│   Forest     │  RF+XGBoost  │  Sequence   │
└──────┬───────┴──────┬───────┴──────┬──────┘
       └──────────────┼──────────────┘
                      ↓
              Score fusion (weighted)
                      ↓
              ThreatEvent + Severity
                      ↓
          AlertEngine → auto-response hooks
                      ↓
              Dashboard + API
```

---

## Scalability Notes

For production use, the following extensions are recommended:

1. **Kafka / Redis Streams** for real-time event ingestion instead of batch CSV
2. **Feature store** (Feast / Tecton) to share rolling window features across services
3. **Model versioning** with MLflow or DVC
4. **HIPAA-compliant logging** — anonymize IP addresses and patient identifiers before storage
5. **Federated learning** — train across hospital networks without sharing raw patient data
