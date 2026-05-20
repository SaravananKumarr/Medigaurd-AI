"""
src/data/simulator.py
Synthetic healthcare network event generator.
Produces labeled normal + attack traffic for model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


ATTACK_TYPES = ["normal", "ransomware", "brute_force", "lateral_movement",
                "ehr_unauthorized", "data_exfiltration", "iomt_compromise"]

HOSPITAL_SUBNETS = {
    "clinical":   "192.168.10.",
    "admin":      "192.168.20.",
    "ehr_server": "192.168.30.",
    "iomt":       "192.168.40.",
    "external":   "10.0.0.",
}

KNOWN_PORTS = [80, 443, 22, 3389, 8080, 8443, 1433, 3306, 5432, 9200]
EHR_PORTS   = [8080, 8443, 9200]


def _random_ip(subnet: str) -> str:
    return subnet + str(random.randint(2, 254))


def _timestamp_series(n: int, start: datetime, jitter_seconds: int = 300) -> list:
    ts = start
    result = []
    for _ in range(n):
        ts += timedelta(seconds=random.randint(1, jitter_seconds))
        result.append(ts)
    return result


def _generate_normal(n: int, start: datetime) -> pd.DataFrame:
    rows = []
    timestamps = _timestamp_series(n, start)
    for ts in timestamps:
        src_net = random.choice(list(HOSPITAL_SUBNETS.values())[:4])
        dst_net = random.choice(list(HOSPITAL_SUBNETS.values())[:4])
        row = {
            "timestamp":        ts,
            "src_ip":           _random_ip(src_net),
            "dst_ip":           _random_ip(dst_net),
            "dst_port":         random.choice(KNOWN_PORTS),
            "protocol":         random.choice(["TCP", "UDP", "HTTPS"]),
            "bytes_out":        random.randint(100, 8000),
            "bytes_in":         random.randint(100, 5000),
            "duration_ms":      random.randint(50, 2000),
            "failed_logins":    0 if random.random() > 0.05 else random.randint(1, 3),
            "records_accessed": random.randint(1, 10),
            "file_write_rate":  random.uniform(0, 2),
            "entropy":          random.uniform(3.5, 5.5),
            "unique_dst_ips":   random.randint(1, 5),
            "port_scan_score":  0.0,
            "device_ping_rate": random.uniform(0.1, 1.0),
            "cmd_anomaly":      0,
            "dst_is_external":  0,
            "access_hour":      ts.hour,
            "label":            "normal",
            "attack_type":      "normal",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _generate_ransomware(n: int, start: datetime) -> pd.DataFrame:
    rows = []
    timestamps = _timestamp_series(n, start, jitter_seconds=10)
    victim_ip = _random_ip(HOSPITAL_SUBNETS["clinical"])
    for ts in timestamps:
        row = {
            "timestamp":        ts,
            "src_ip":           victim_ip,
            "dst_ip":           _random_ip(HOSPITAL_SUBNETS["ehr_server"]),
            "dst_port":         random.choice(EHR_PORTS),
            "protocol":         "TCP",
            "bytes_out":        random.randint(50000, 500000),
            "bytes_in":         random.randint(100, 500),
            "duration_ms":      random.randint(5, 200),
            "failed_logins":    0,
            "records_accessed": 0,
            "file_write_rate":  random.uniform(50, 200),   # high write rate
            "entropy":          random.uniform(7.5, 8.0),  # encrypted data
            "unique_dst_ips":   random.randint(1, 3),
            "port_scan_score":  0.0,
            "device_ping_rate": 0.0,
            "cmd_anomaly":      1,
            "dst_is_external":  0,
            "access_hour":      ts.hour,
            "label":            "attack",
            "attack_type":      "ransomware",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _generate_brute_force(n: int, start: datetime) -> pd.DataFrame:
    rows = []
    timestamps = _timestamp_series(n, start, jitter_seconds=5)
    attacker_ip = _random_ip(HOSPITAL_SUBNETS["external"])
    for ts in timestamps:
        row = {
            "timestamp":        ts,
            "src_ip":           attacker_ip,
            "dst_ip":           _random_ip(HOSPITAL_SUBNETS["ehr_server"]),
            "dst_port":         random.choice([22, 3389, 8443]),
            "protocol":         "TCP",
            "bytes_out":        random.randint(200, 800),
            "bytes_in":         random.randint(100, 300),
            "duration_ms":      random.randint(10, 100),
            "failed_logins":    random.randint(3, 20),
            "records_accessed": 0,
            "file_write_rate":  0.0,
            "entropy":          random.uniform(3.0, 4.5),
            "unique_dst_ips":   1,
            "port_scan_score":  0.0,
            "device_ping_rate": 0.0,
            "cmd_anomaly":      0,
            "dst_is_external":  0,
            "access_hour":      ts.hour,
            "label":            "attack",
            "attack_type":      "brute_force",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _generate_lateral_movement(n: int, start: datetime) -> pd.DataFrame:
    rows = []
    timestamps = _timestamp_series(n, start, jitter_seconds=30)
    attacker_ip = _random_ip(HOSPITAL_SUBNETS["admin"])
    for ts in timestamps:
        row = {
            "timestamp":        ts,
            "src_ip":           attacker_ip,
            "dst_ip":           _random_ip(random.choice([HOSPITAL_SUBNETS["clinical"],
                                                           HOSPITAL_SUBNETS["ehr_server"],
                                                           HOSPITAL_SUBNETS["iomt"]])),
            "dst_port":         random.randint(1, 65535),
            "protocol":         random.choice(["TCP", "UDP"]),
            "bytes_out":        random.randint(60, 2000),
            "bytes_in":         random.randint(60, 2000),
            "duration_ms":      random.randint(5, 500),
            "failed_logins":    random.randint(0, 2),
            "records_accessed": 0,
            "file_write_rate":  0.0,
            "entropy":          random.uniform(3.5, 6.0),
            "unique_dst_ips":   random.randint(10, 50),
            "port_scan_score":  random.uniform(0.6, 1.0),
            "device_ping_rate": 0.0,
            "cmd_anomaly":      0,
            "dst_is_external":  0,
            "access_hour":      ts.hour,
            "label":            "attack",
            "attack_type":      "lateral_movement",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _generate_ehr_unauthorized(n: int, start: datetime) -> pd.DataFrame:
    rows = []
    off_hours = [random.choice(list(range(0, 6)) + list(range(22, 24)))
                 for _ in range(n)]
    timestamps = _timestamp_series(n, start, jitter_seconds=120)
    for i, ts in enumerate(timestamps):
        ts = ts.replace(hour=off_hours[i])
        row = {
            "timestamp":        ts,
            "src_ip":           _random_ip(HOSPITAL_SUBNETS["admin"]),
            "dst_ip":           _random_ip(HOSPITAL_SUBNETS["ehr_server"]),
            "dst_port":         random.choice(EHR_PORTS),
            "protocol":         "HTTPS",
            "bytes_out":        random.randint(5000, 50000),
            "bytes_in":         random.randint(1000, 10000),
            "duration_ms":      random.randint(500, 8000),
            "failed_logins":    0,
            "records_accessed": random.randint(100, 1000),
            "file_write_rate":  0.0,
            "entropy":          random.uniform(4.0, 5.0),
            "unique_dst_ips":   1,
            "port_scan_score":  0.0,
            "device_ping_rate": 0.0,
            "cmd_anomaly":      0,
            "dst_is_external":  0,
            "access_hour":      ts.hour,
            "label":            "attack",
            "attack_type":      "ehr_unauthorized",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _generate_exfiltration(n: int, start: datetime) -> pd.DataFrame:
    rows = []
    timestamps = _timestamp_series(n, start, jitter_seconds=60)
    src_ip = _random_ip(HOSPITAL_SUBNETS["clinical"])
    for ts in timestamps:
        row = {
            "timestamp":        ts,
            "src_ip":           src_ip,
            "dst_ip":           _random_ip(HOSPITAL_SUBNETS["external"]),
            "dst_port":         random.choice([443, 80, 8080]),
            "protocol":         "HTTPS",
            "bytes_out":        random.randint(500000, 5000000),
            "bytes_in":         random.randint(100, 500),
            "duration_ms":      random.randint(2000, 30000),
            "failed_logins":    0,
            "records_accessed": random.randint(50, 500),
            "file_write_rate":  0.0,
            "entropy":          random.uniform(6.0, 7.5),
            "unique_dst_ips":   1,
            "port_scan_score":  0.0,
            "device_ping_rate": 0.0,
            "cmd_anomaly":      0,
            "dst_is_external":  1,
            "access_hour":      ts.hour,
            "label":            "attack",
            "attack_type":      "data_exfiltration",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _generate_iomt_compromise(n: int, start: datetime) -> pd.DataFrame:
    rows = []
    timestamps = _timestamp_series(n, start, jitter_seconds=20)
    device_ip = _random_ip(HOSPITAL_SUBNETS["iomt"])
    for ts in timestamps:
        row = {
            "timestamp":        ts,
            "src_ip":           device_ip,
            "dst_ip":           _random_ip(HOSPITAL_SUBNETS["external"]),
            "dst_port":         random.randint(1024, 65535),
            "protocol":         "UDP",
            "bytes_out":        random.randint(1000, 10000),
            "bytes_in":         random.randint(200, 2000),
            "duration_ms":      random.randint(10, 500),
            "failed_logins":    0,
            "records_accessed": 0,
            "file_write_rate":  0.0,
            "entropy":          random.uniform(5.0, 7.0),
            "unique_dst_ips":   random.randint(1, 10),
            "port_scan_score":  random.uniform(0.3, 0.8),
            "device_ping_rate": random.uniform(10, 100),
            "cmd_anomaly":      1,
            "dst_is_external":  1,
            "access_hour":      ts.hour,
            "label":            "attack",
            "attack_type":      "iomt_compromise",
        }
        rows.append(row)
    return pd.DataFrame(rows)


_GENERATORS = {
    "normal":           _generate_normal,
    "ransomware":       _generate_ransomware,
    "brute_force":      _generate_brute_force,
    "lateral_movement": _generate_lateral_movement,
    "ehr_unauthorized": _generate_ehr_unauthorized,
    "data_exfiltration":_generate_exfiltration,
    "iomt_compromise":  _generate_iomt_compromise,
}


def generate_dataset(
    n_normal: int = 5000,
    n_per_attack: int = 500,
    start: datetime = None,
    seed: int = 42,
    attack_types: list = None,
) -> pd.DataFrame:
    """
    Generate a labeled dataset of normal + attack healthcare network events.

    Args:
        n_normal:       Number of normal traffic records.
        n_per_attack:   Number of records per attack type.
        start:          Start timestamp (defaults to now).
        seed:           Random seed for reproducibility.
        attack_types:   Subset of attack types to include (default: all).

    Returns:
        pd.DataFrame with all events, shuffled, with 'label' and 'attack_type' columns.
    """
    np.random.seed(seed)
    random.seed(seed)

    start = start or datetime(2024, 1, 1, 8, 0, 0)
    attacks = attack_types or list(_GENERATORS.keys())[1:]  # skip 'normal'

    frames = [_generate_normal(n_normal, start)]
    for attack in attacks:
        if attack in _GENERATORS:
            frames.append(_GENERATORS[attack](n_per_attack, start))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Generate MediGuard synthetic dataset")
    parser.add_argument("--n_normal",      type=int, default=5000)
    parser.add_argument("--n_per_attack",  type=int, default=500)
    parser.add_argument("--output",        type=str, default="data/healthcare_events.csv")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    df = generate_dataset(args.n_normal, args.n_per_attack, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Dataset saved: {args.output}")
    print(df["attack_type"].value_counts())
