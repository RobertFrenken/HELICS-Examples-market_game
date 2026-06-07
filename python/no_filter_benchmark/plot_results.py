"""Generate timing histogram PNGs from no-filter benchmark pickle results."""

from __future__ import annotations

import argparse
from datetime import datetime as dt
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = (
    ("without_filter", "Without filter", "results_without_filter.pickle"),
    ("with_filter", "With filter", "results_with_filter.pickle"),
)


def load_transit_times(path: Path) -> np.ndarray:
    """Load one benchmark pickle and return message transit times in ms."""
    with path.open("rb") as file:
        records = pickle.load(file)
    times = []
    for record in records:
        send_time = dt.strptime(record["send"], "%Y-%m-%d %H:%M:%S.%f")
        receive_time = dt.strptime(record["receive"], "%Y-%m-%d %H:%M:%S.%f")
        times.append((receive_time - send_time).total_seconds() * 1000.0)
    return np.asarray(times, dtype=float)


def save_histogram(times: np.ndarray, title: str, output_path: Path) -> None:
    """Save one labeled timing histogram."""
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(times, bins="auto", color="#1f77b4", alpha=0.9)
    axis.set_title(title)
    axis.set_xlabel("Round-trip message transit time (ms)")
    axis.set_ylabel("Message count")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_comparison(data: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    """Save one overlay comparing all available benchmark datasets."""
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for label, times in data:
        axis.hist(times, bins="auto", alpha=0.55, label=label)
    axis.set_title("HELICS message timing benchmark")
    axis.set_xlabel("Round-trip message transit time (ms)")
    axis.set_ylabel("Message count")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="plot no-filter benchmark timing results from pickle files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="directory containing results_with_filter.pickle and results_without_filter.pickle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="directory where PNG plots should be written",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_data = []
    for stem, label, filename in DATASETS:
        path = args.data_dir / filename
        if not path.exists():
            continue
        times = load_transit_times(path)
        comparison_data.append((label, times))
        save_histogram(times, label, args.output_dir / f"{stem}.png")

    if len(comparison_data) >= 2:
        save_comparison(comparison_data, args.output_dir / "filter_timing_comparison.png")
    if not comparison_data:
        raise SystemExit(f"no benchmark pickle files found in {args.data_dir}")


if __name__ == "__main__":
    main()
