"""Aggregate scenario evaluation CSV rows across seeds."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


DEFAULT_METRICS = [
    "total_cost",
    "total_load",
    "final_battery",
    "clamps",
    "invalid_load_adjustment",
    "penalty_cost",
    "price_volatility",
]

GROUP_COLUMNS = ["scenario", "profile_type", "agent"]
OUTPUT_COLUMNS = [
    "scenario",
    "profile_type",
    "agent",
    "seeds",
    "rows",
    "rank",
]


@dataclass(frozen=True)
class AggregateGroup:
    """Numeric values for one policy within one scenario/profile pair."""

    scenario: str
    profile_type: str
    agent: str
    seeds: tuple[str, ...]
    row_count: int
    values: dict[str, tuple[float, ...]]


def read_csv_rows(paths: list[str]) -> list[dict[str, str]]:
    """Read scenario CSV rows from one or more files, or stdin when empty."""
    if not paths:
        return list(csv.DictReader(sys.stdin))
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def aggregate_rows(
    rows: list[dict[str, str]],
    metrics: list[str] | None = None,
    rank_metric: str = "total_cost",
) -> list[dict[str, str]]:
    """Return wide aggregate rows with mean/stdev/min/max metric columns."""
    metrics = metrics if metrics is not None else DEFAULT_METRICS
    if not rows:
        return []
    groups = _groups(rows, metrics)
    output = [_output_row(group, metrics) for group in groups]
    _add_ranks(output, rank_metric)
    return output


def write_aggregate_rows(rows: list[dict[str, str]]) -> None:
    """Write aggregate rows as CSV to stdout."""
    fieldnames = _fieldnames(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _groups(
    rows: list[dict[str, str]],
    metrics: list[str],
) -> list[AggregateGroup]:
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(column, "") for column in GROUP_COLUMNS)
        buckets[key].append(row)

    groups = []
    for key in sorted(buckets):
        bucket = buckets[key]
        values = {
            metric: tuple(_numeric_values(bucket, metric))
            for metric in metrics
        }
        groups.append(
            AggregateGroup(
                scenario=key[0],
                profile_type=key[1],
                agent=key[2],
                seeds=_sorted_seed_values(row.get("seed", "") for row in bucket),
                row_count=len(bucket),
                values=values,
            )
        )
    return groups


def _numeric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(metric, ""))
        except ValueError as exc:
            raise ValueError(f"metric {metric!r} contains non-numeric value") from exc
        if not math.isfinite(value):
            raise ValueError(f"metric {metric!r} contains non-finite value")
        values.append(value)
    return values


def _sorted_seed_values(values: Iterable[object]) -> tuple[str, ...]:
    seed_values = {str(value) for value in values}

    def sort_key(value: str) -> tuple[int, int | str]:
        if value.lstrip("-").isdigit():
            return (0, int(value))
        return (1, value)

    return tuple(sorted(seed_values, key=sort_key))


def _output_row(group: AggregateGroup, metrics: list[str]) -> dict[str, str]:
    row = {
        "scenario": group.scenario,
        "profile_type": group.profile_type,
        "agent": group.agent,
        "seeds": "|".join(group.seeds),
        "rows": str(group.row_count),
        "rank": "",
    }
    for metric in metrics:
        values = group.values[metric]
        row[f"{metric}_mean"] = _format_float(_mean(values))
        row[f"{metric}_stdev"] = _format_float(_sample_stdev(values))
        row[f"{metric}_min"] = _format_float(min(values))
        row[f"{metric}_max"] = _format_float(max(values))
    return row


def _add_ranks(rows: list[dict[str, str]], rank_metric: str) -> None:
    mean_column = f"{rank_metric}_mean"
    by_scenario: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if mean_column not in row:
            raise ValueError(f"rank metric {rank_metric!r} was not aggregated")
        by_scenario[(row["scenario"], row["profile_type"])].append(row)

    for scenario_rows in by_scenario.values():
        ordered = sorted(scenario_rows, key=lambda row: float(row[mean_column]))
        previous_value: float | None = None
        previous_rank = 0
        for index, row in enumerate(ordered, start=1):
            value = float(row[mean_column])
            rank = previous_rank if previous_value == value else index
            row["rank"] = str(rank)
            previous_value = value
            previous_rank = rank


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return OUTPUT_COLUMNS
    metric_columns = [
        key
        for key in rows[0]
        if key not in OUTPUT_COLUMNS
    ]
    return OUTPUT_COLUMNS + metric_columns


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _sample_stdev(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _format_float(value: float) -> str:
    return f"{value:.10f}"


def _parse_metrics(value: str | None) -> list[str]:
    if value is None:
        return DEFAULT_METRICS
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    if not metrics:
        raise SystemExit("--metrics must include at least one column")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="aggregate evaluate_scenarios CSV rows across seeds"
    )
    parser.add_argument(
        "csv",
        nargs="*",
        help="input CSV file(s); reads stdin when omitted",
    )
    parser.add_argument(
        "--metrics",
        help="comma-separated numeric columns to summarize",
    )
    parser.add_argument(
        "--rank-metric",
        default="total_cost",
        help="metric whose mean is used for per-scenario policy ranking",
    )
    args = parser.parse_args()
    metrics = _parse_metrics(args.metrics)
    rows = aggregate_rows(
        read_csv_rows(args.csv),
        metrics=metrics,
        rank_metric=args.rank_metric,
    )
    write_aggregate_rows(rows)


if __name__ == "__main__":
    main()
