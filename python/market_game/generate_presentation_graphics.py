from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "presentation_assets"

HOURS = np.arange(24)
BASE_DEMAND = np.array([2, 1, 1, 1, 2, 4, 6, 8, 9, 7, 5, 4, 3, 4, 5, 7, 9, 12, 10, 7, 5, 4, 2, 2], dtype=float)
RUN_PRICE = np.array(
    [
        0.5,
        0.15,
        0.17,
        0.13,
        0.13,
        0.10,
        0.10,
        0.49,
        0.17,
        0.5733333333,
        0.2566666667,
        0.10,
        0.10,
        0.29,
        0.17,
        0.2566666667,
        0.39,
        0.14,
        0.2233333333,
        1.1566666667,
        0.19,
        0.3566666667,
        0.16,
        0.10,
    ],
    dtype=float,
)
HOUSE_RUNS = {
    "FlattenDemandHouse": {
        "color": "#1E4D6B",
        "load": np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 7, 5, 5, 5, 5], dtype=float),
        "cost": np.array([2.5, 0.75, 0.85, 0.65, 0.65, 0.5, 0.5, 2.45, 0.85, 2.8666666667, 1.2833333333, 0.5, 0.5, 1.45, 0.85, 1.2833333333, 1.95, 0.7, 2.2333333333, 8.0966666667, 0.95, 1.7833333333, 0.8, 0.5], dtype=float),
    },
    "PriceAwareHouse": {
        "color": "#2E7D32",
        "load": np.array([2, 5, 1, 1, 2, 9, 11, -2, 9, 3, 5, 9, 8, 2, 5, 7, 9, 12, 10, -1, 8, 1, 2, 7], dtype=float),
        "cost": np.array([1.0, 0.75, 0.17, 0.13, 0.26, 0.9, 1.1, -0.98, 1.53, 1.72, 1.2833333333, 0.9, 0.8, 0.58, 0.85, 1.7966666667, 3.51, 1.68, 2.2333333333, -1.1566666667, 1.52, 0.3566666667, 0.32, 0.7], dtype=float),
    },
    "FullCycleHouse": {
        "color": "#BF6C12",
        "load": np.array([7, 6, 6, 6, -8, -6, 11, 13, 14, 12, -5, -6, 8, 9, 10, 12, -1, 2, 15, 12, 10, 9, -8, -8], dtype=float),
        "cost": np.array([3.5, 0.9, 1.02, 0.78, -1.04, -0.6, 1.1, 6.37, 2.38, 6.88, -1.2833333333, -0.6, 0.8, 2.61, 1.7, 3.08, -0.39, 0.28, 3.35, 13.88, 1.9, 3.21, -1.28, -0.8], dtype=float),
    },
}


def compute_price_from_average_demand(avg_demand: np.ndarray) -> np.ndarray:
    avg_demand = np.asarray(avg_demand, dtype=float)
    price = np.empty_like(avg_demand)

    low = avg_demand < 3.0
    mid1 = (avg_demand >= 3.0) & (avg_demand < 6.0)
    mid2 = (avg_demand >= 6.0) & (avg_demand < 9.0)
    high = (avg_demand >= 9.0) & (avg_demand < 13.0)
    extreme = avg_demand >= 13.0

    price[low] = 0.10
    price[mid1] = 0.10 + 0.03 * (avg_demand[mid1] - 3.0)
    price[mid2] = 0.19 + 0.10 * (avg_demand[mid2] - 6.0)
    price[high] = 0.49 + 0.25 * (avg_demand[high] - 9.0)
    price[extreme] = 1.49 + 1.00 * (avg_demand[extreme] - 13.0)
    return price


def compute_price_from_total_demand(total_demand: np.ndarray, houses: int) -> np.ndarray:
    return compute_price_from_average_demand(np.asarray(total_demand, dtype=float) / houses)


def apply_chart_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F6F1E8",
            "axes.facecolor": "#FFFDF9",
            "axes.edgecolor": "#2B2B2B",
            "axes.labelcolor": "#2B2B2B",
            "axes.titleweight": "bold",
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "xtick.color": "#2B2B2B",
            "ytick.color": "#2B2B2B",
            "grid.color": "#D8D0C2",
            "font.size": 11,
            "savefig.facecolor": "#F6F1E8",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.svg")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220)
    plt.close(fig)


def generate_average_demand_curve() -> None:
    x = np.linspace(0.0, 16.0, 500)
    y = compute_price_from_average_demand(x)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, y, color="#BF3B2B", linewidth=4)

    bands = [
        (0, 3, "#C8E6C9", "Cheap"),
        (3, 6, "#F4E6A6", "Rising"),
        (6, 9, "#F7C97B", "Costly"),
        (9, 13, "#F1A46C", "Painful"),
        (13, 16, "#D96C6C", "Punishing"),
    ]
    for start, end, color, label in bands:
        ax.axvspan(start, end, color=color, alpha=0.45)
        ax.text((start + end) / 2, 2.55, label, ha="center", va="center", fontsize=11, color="#3B3128")

    breakpoints = np.array([3.0, 6.0, 9.0, 13.0])
    breakpoint_prices = compute_price_from_average_demand(breakpoints)
    ax.scatter(breakpoints, breakpoint_prices, color="#1E4D6B", s=90, zorder=4)

    labels = [
        "M = 3\n$0.10",
        "M = 6\n$0.19",
        "M = 9\n$0.49",
        "M = 13\n$1.49",
    ]
    offsets = [(0.2, 0.14), (0.2, 0.18), (0.2, 0.22), (0.2, 0.15)]
    for px, py, label, (dx, dy) in zip(breakpoints, breakpoint_prices, labels, offsets):
        ax.annotate(
            label,
            xy=(px, py),
            xytext=(px + dx, py + dy),
            fontsize=10,
            color="#1E4D6B",
            arrowprops={"arrowstyle": "-", "color": "#1E4D6B", "lw": 1.2},
        )

    ax.set_title("Market Price Curve by Average Demand per House")
    ax.set_xlabel("Average Demand per House, M (kWh)")
    ax.set_ylabel("Price ($/kWh)")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 2.8)
    ax.grid(True, alpha=0.7)
    ax.text(
        0.02,
        0.95,
        "Below 3 kWh per house, the price stays flat.\nAfter 9 kWh, the curve becomes steep quickly.",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#3B3128",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFF6DD", "edgecolor": "#D8D0C2"},
    )

    save_figure(fig, "market_price_curve_average_demand")


def generate_total_demand_curves() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    house_counts = [4, 8, 12, 20]
    colors = ["#1E4D6B", "#2E7D32", "#BF6C12", "#8E2F5B"]

    for houses, color in zip(house_counts, colors):
        x = np.linspace(0.0, houses * 16.0, 500)
        y = compute_price_from_total_demand(x, houses)
        ax.plot(x, y, label=f"{houses} houses", color=color, linewidth=3)

    ax.set_title("Same Rule, Different Neighborhood Sizes")
    ax.set_xlabel("Total Neighborhood Demand (kWh)")
    ax.set_ylabel("Price ($/kWh)")
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 2.8)
    ax.grid(True, alpha=0.7)
    ax.legend(frameon=True, facecolor="#FFFDF9", edgecolor="#D8D0C2")
    ax.text(
        0.02,
        0.95,
        "The market maker prices from average demand, not raw total demand.\nBigger neighborhoods hit the same price tiers at larger totals.",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#3B3128",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EAF4FB", "edgecolor": "#D8D0C2"},
    )

    save_figure(fig, "market_price_curve_total_demand")


def generate_price_tiers_reference() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")

    tiers = [
        ("Cheap", "M < 3", "$0.10/kWh", "#C8E6C9"),
        ("Rising", "3 <= M < 6", "$0.10 + 0.03 * (M - 3)", "#F4E6A6"),
        ("Costly", "6 <= M < 9", "$0.19 + 0.10 * (M - 6)", "#F7C97B"),
        ("Painful", "9 <= M < 13", "$0.49 + 0.25 * (M - 9)", "#F1A46C"),
        ("Punishing", "M >= 13", "$1.49 + 1.00 * (M - 13)", "#D96C6C"),
    ]

    x_positions = np.linspace(0.02, 0.82, len(tiers))
    width = 0.16
    for x, (name, rule, formula, color) in zip(x_positions, tiers):
        rect = plt.Rectangle((x, 0.2), width, 0.6, facecolor=color, edgecolor="#2B2B2B", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + width / 2, 0.68, name, ha="center", va="center", fontsize=13, fontweight="bold", color="#2B2B2B")
        ax.text(x + width / 2, 0.52, rule, ha="center", va="center", fontsize=11, color="#2B2B2B")
        ax.text(x + width / 2, 0.35, formula, ha="center", va="center", fontsize=10, color="#2B2B2B", wrap=True)

    ax.text(0.02, 0.93, "Price Tiers Players Can Remember", fontsize=20, fontweight="bold", color="#2B2B2B")
    ax.text(
        0.02,
        0.08,
        "M is average neighborhood demand per house. Once the group pushes M above 9, each extra kWh gets expensive fast.",
        fontsize=12,
        color="#3B3128",
    )

    save_figure(fig, "market_price_tiers_reference")


def generate_house_load_comparison() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(HOURS, BASE_DEMAND, color="#2B2B2B", linewidth=3, linestyle="--", label="Base demand")

    for name, data in HOUSE_RUNS.items():
        ax.plot(HOURS, data["load"], linewidth=3, color=data["color"], label=name)

    ax.axhline(0.0, color="#8E8A84", linewidth=1.2, alpha=0.8)
    ax.set_title("Hourly Market Load by House Strategy")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Market Load (kWh)")
    ax.set_xticks(np.arange(0, 24, 1))
    ax.set_xlim(0, 23)
    ax.grid(True, alpha=0.7)
    ax.legend(ncols=2, frameon=True, facecolor="#FFFDF9", edgecolor="#D8D0C2")
    ax.text(
        0.02,
        0.95,
        "FlattenDemandHouse stays near 5 kWh for much of the day.\nPriceAwareHouse shifts energy around expensive hours.\nFullCycleHouse swings hardest because it ignores price.",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#3B3128",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFF6DD", "edgecolor": "#D8D0C2"},
    )

    save_figure(fig, "house_strategy_hourly_loads")


def generate_house_cost_comparison() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    for name, data in HOUSE_RUNS.items():
        ax.plot(HOURS, data["cost"], linewidth=3, color=data["color"], label=name)

    ax.axhline(0.0, color="#8E8A84", linewidth=1.2, alpha=0.8)
    ax.set_title("Hourly Cost by House Strategy")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Hourly Cost ($)")
    ax.set_xticks(np.arange(0, 24, 1))
    ax.set_xlim(0, 23)
    ax.grid(True, alpha=0.7)
    ax.legend(frameon=True, facecolor="#FFFDF9", edgecolor="#D8D0C2")
    ax.text(
        0.02,
        0.95,
        "The most expensive shared moment is hour 19.\nPriceAwareHouse avoids a large positive purchase there,\nwhile FullCycleHouse pays the biggest penalty.",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#3B3128",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EAF4FB", "edgecolor": "#D8D0C2"},
    )

    save_figure(fig, "house_strategy_hourly_costs")


def generate_cumulative_cost_comparison() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    for name, data in HOUSE_RUNS.items():
        ax.plot(HOURS, np.cumsum(data["cost"]), linewidth=3, color=data["color"], label=name)

    ax.set_title("Cumulative Cost Through the Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_xticks(np.arange(0, 24, 1))
    ax.set_xlim(0, 23)
    ax.grid(True, alpha=0.7)
    ax.legend(frameon=True, facecolor="#FFFDF9", edgecolor="#D8D0C2")
    ax.text(
        0.02,
        0.95,
        "PriceAwareHouse separates from the others as expensive hours arrive.\nFlattenDemandHouse stays middling.\nFullCycleHouse falls behind after repeated badly timed swings.",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#3B3128",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FDEEE6", "edgecolor": "#D8D0C2"},
    )

    save_figure(fig, "house_strategy_cumulative_costs")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    apply_chart_style()
    generate_average_demand_curve()
    generate_total_demand_curves()
    generate_price_tiers_reference()
    generate_house_load_comparison()
    generate_house_cost_comparison()
    generate_cumulative_cost_comparison()
    print(f"Created presentation graphics in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
