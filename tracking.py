"""Simple CSV-based experiment tracking."""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def log_experiment(
    experiment_name: str,
    model_name: str,
    metrics: dict,
    params: dict | None = None,
) -> Path:
    """Append one row to results/{experiment_name}.csv."""
    filepath = RESULTS_DIR / f"{experiment_name}.csv"
    params = params or {}

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        **{f"param_{k}": v for k, v in params.items()},
        **{f"metric_{k}": v for k, v in metrics.items()},
    }

    file_exists = filepath.exists()
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return filepath
