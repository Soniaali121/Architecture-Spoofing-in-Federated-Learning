"""Generate a small synthetic IDS-style CSV for local FL smoke tests."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

CATEGORIES = [
    "Benign",
    "DDoS",
    "DoS",
    "Recon",
    "Mirai",
    "Spoofing",
    "Web",
]

FEATURE_NAMES = [
    "flow_duration",
    "Header_Length",
    "Protocol_Type",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
    "Tot_sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot_size",
    "IAT",
    "Number",
    "Magnitue",
]


def _class_centroid(class_idx: int, n_features: int) -> list[float]:
    """Deterministic per-class mean so the problem is learnable."""
    return [
        math.sin(0.7 * class_idx + 0.13 * j) * 2.0 + 0.4 * class_idx
        for j in range(n_features)
    ]


def generate_rows(n_samples: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    n_features = len(FEATURE_NAMES)
    centroids = [_class_centroid(i, n_features) for i in range(len(CATEGORIES))]
    rows = []
    for i in range(n_samples):
        class_idx = i % len(CATEGORIES)
        # Mild class imbalance: Benign a bit more frequent.
        if rng.random() < 0.15:
            class_idx = 0
        mean = centroids[class_idx]
        features = {
            name: round(mean[j] + rng.gauss(0, 0.75), 6)
            for j, name in enumerate(FEATURE_NAMES)
        }
        features["Attack_Category_x"] = CATEGORIES[class_idx]
        rows.append(features)
    rng.shuffle(rows)
    return rows


def write_csv(path: Path, n_samples: int = 2000, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(n_samples=n_samples, seed=seed)
    fieldnames = FEATURE_NAMES + ["Attack_Category_x"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="data/example_ids_iot.csv",
        help="Output CSV path",
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = write_csv(Path(args.out), n_samples=args.n_samples, seed=args.seed)
    print(f"Wrote {args.n_samples} rows -> {out}")


if __name__ == "__main__":
    main()
