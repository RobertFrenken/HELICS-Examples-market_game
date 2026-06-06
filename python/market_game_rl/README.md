# Market Game RL Helpers

This directory contains a pure-Python simulator and training/evaluation helpers
for the HELICS market game.

It intentionally does not import from `python/market_game`.

Reasons:

- `market_maker.py` executes the HELICS market loop at import time.
- `house_template.py` imports HELICS and matplotlib.
- RL needs a fast single-process inner loop.
- The pure simulator should be usable without broker/process orchestration.

The simulator mirrors the inspected game mechanics:

- initial price `0.5`;
- 24-hour episode;
- current price included in `price_history`;
- one-hour price lag;
- pricing from average market-facing load;
- battery capacity `20`, max charge `5`, max discharge `10`;
- market-maker effective clamping;
- negative market-facing load allowed when battery discharge supports it.

## Parity Check

From the repo root:

```bash
python3 -m python.market_game_rl.evaluate
```

Expected stock `profile1` output:

```text
agent,total_load,total_cost,final_battery
FlattenDemandHouse,127.0000000000,35.4466666667,7.0000000000
FullCycleHouse,120.0000000000,47.7466666667,0.0000000000
PriceAwareHouse,125.0000000000,21.9533333333,5.0000000000
```

These values match a live HELICS run of the included example houses.
