# Downstream HELICS Compatibility Notes

The canonical HELICS house template and student-facing game guide now live in
`python/market_game`. Start there for local play and classroom/CTF submission
shape:

- `python/market_game/house_template.py`
- `python/market_game/market_game.md`

This downstream HELICS directory is kept only for older local experiments that
already imported it. New strategy work should use the canonical template above,
and downstream RL/export work should target a standalone `compute_demand(...)`
function.

## Historical Student Start

Each team controls one house for a 24-hour electricity market game. Every hour
your house sees the current price, its base demand, and its battery charge. Your
strategy decides how much energy to buy from the market.

You do not need to understand HELICS to play. Start by editing a house strategy.

## 5-Minute Path

1. Copy an example from `python/market_game/houses/` into a new file ending in
   `_house.py`.
2. Give the class and player name a unique name.
3. Implement either `compute_demand(...)` or the simpler `choose_action(...)`.
4. From `python/market_game`, run:

```bash
uv run python run_neighborhood.py houses
uv run helics run --path=houses.json
```

If you are using an activated pip virtual environment instead of uv, generate
the runner with `python run_neighborhood.py houses --launcher plain`.

## Easiest Strategy Hook

Use `ActionHouse` when you want to choose only charge, neutral, or discharge.

```python
import argparse

from house_template import ActionHouse, BatteryAction


class MyHouse(ActionHouse):
    def choose_action(self, price, hour, battery_charge, demand, price_history):
        if price <= 0.12:
            return BatteryAction.CHARGE
        if price >= 0.49:
            return BatteryAction.DISCHARGE
        return BatteryAction.NEUTRAL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    house = MyHouse("MyHouse", args.broker)
    house.run()
    if not args.no_plot:
        house.plot_results()
```

`ActionHouse` converts the action into a legal market load for you.

## Original Strategy Hook

Existing examples use `House.compute_demand(...)`, which returns the exact
market-facing load for the current hour.

```python
from house_template import House


class FollowDemandHouse(House):
    def compute_demand(self, price, hour, battery_charge, demand, price_history):
        return demand[hour]
```

Let `base = demand[hour]`.

| Return value | Meaning |
|---|---|
| `base` | Buy exactly the base demand; battery is unchanged. |
| More than `base` | Buy extra energy and charge the battery. |
| Less than `base` | Use battery energy and buy less from the market. |

## Terms

| Term | Meaning |
|---|---|
| `base_demand` | The house's normal energy need before battery use. |
| `battery_delta` | Positive charges the battery; negative discharges it. |
| `market_load` | What the market sees: `base_demand + battery_delta`. |
| `price` | Current hourly price in dollars per kWh. |
| `cost` | `price * market_load` for that hour. |

## What Your Strategy Sees

`compute_demand(...)`, `choose_action(...)`, and `choose_delta(...)` receive:

| Input | Meaning |
|---|---|
| `price` | Current price in $/kWh. |
| `hour` | Current hour, `0` through `23`. |
| `battery_charge` | Energy currently stored in your battery. |
| `demand` | Full 24-hour base demand profile. |
| `price_history` | Prices seen so far, including the current price. |

You do not directly see other houses' actions, batteries, costs, or code.

## Examples To Read

| File | What it demonstrates |
|---|---|
| `houses/house_test.py` | Small charge/discharge example. |
| `houses/flatten_demand_house.py` | Smooths demand without using price. |
| `houses/full_cycle_house.py` | Cycles the battery without using price. |
| `houses/price_aware_house.py` | Charges low, discharges high. |

## Rules And Deeper Guides

- `docs/game_rules.md`: battery limits, pricing tiers, timing diagram, and legal information.
- `house_strategy_tutorial.md`: walkthrough of the included strategy examples.
- `../market_game_downstream.rl/README.md`: fast simulator and RL training helpers.
