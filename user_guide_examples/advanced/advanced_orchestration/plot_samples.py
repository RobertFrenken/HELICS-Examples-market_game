"""Generate the EV Monte Carlo peak-power plot from saved simulation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("ggplot")

SCRIPT_DIR = Path(__file__).resolve().parent


def tsplot(
    x,
    y,
    n=20,
    percentile_min=1,
    percentile_max=99,
    color="r",
    plot_mean=True,
    plot_median=False,
    line_color="k",
    **kwargs,
):
    """Plot percentile bands for repeated time-series samples."""
    perc1 = np.percentile(
        y,
        np.linspace(percentile_min, 50, num=n, endpoint=False),
        axis=0,
    )
    perc2 = np.percentile(
        y,
        np.linspace(50, percentile_max, num=n + 1)[1:],
        axis=0,
    )

    alpha = kwargs.pop("alpha", 1 / n)
    for p1, p2 in zip(perc1, perc2):
        plt.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None)
    if plot_mean:
        plt.plot(x, np.mean(y, axis=0), color=line_color)
    if plot_median:
        plt.plot(x, np.median(y, axis=0), color=line_color)

    return plt.gca()


def load_peak_power(samples: int, results_dir: Path, offset: int = 10) -> pd.DataFrame:
    """Load peak-power CSV files from a previous advanced orchestration run."""
    peak = []
    for index in range(samples):
        path = results_dir / f"peak_power_at_all_evs_{index + offset}.csv"
        df = pd.read_csv(path)
        if index != 0:
            df = df.drop(["Hour"], axis=1)
        peak.append(df)
    return pd.concat(peak, axis=1)


def save_peak_power_plot(peak_power: pd.DataFrame, output_file: Path) -> None:
    """Save the Monte Carlo peak-power envelope plot."""
    plt.figure(figsize=[5, 4])
    t = np.array(peak_power.Hour)
    y = np.array(peak_power.iloc[:, 1:]).T
    tsplot(
        t,
        y,
        n=100,
        percentile_min=2.5,
        percentile_max=97.5,
        plot_median=True,
        plot_mean=False,
        color="g",
        line_color="navy",
    )
    plt.ylabel("kW")
    plt.xlabel("Hours")
    plt.plot()
    plt.gca().set_position([0.14, 0.14, 0.82, 0.82])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="plot EV Monte Carlo peak power from saved result CSVs"
    )
    parser.add_argument("samples", nargs="?", type=int, default=30)
    parser.add_argument("output_path", nargs="?", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--output-file",
        type=Path,
        help="PNG path to write; defaults to OUTPUT_PATH/montecarlo-ev-peak-power.png",
    )
    parser.add_argument("--no-show", action="store_true", help="do not open a plot window")
    args = parser.parse_args()

    results_dir = args.output_path / "results"
    output_file = args.output_file or args.output_path / "montecarlo-ev-peak-power.png"
    print(f"plotting {args.samples} samples from {results_dir}")
    peak_power = load_peak_power(args.samples, results_dir)
    save_peak_power_plot(peak_power, output_file)
    print(f"saved {output_file}")
    if args.no_show:
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
