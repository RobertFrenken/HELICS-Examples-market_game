# Market Game Export Helpers

This directory is the boundary between flexible local training and a
competition-safe submitted house policy.

The local workbench can use Gymnasium, Ray, Torch, checkpoints, oracle traces,
and scenario randomization. The exported artifact should be a plain function:

```python
def compute_demand(price, hour, battery_charge, demand, price_history):
    ...
```

Use `validators.py` to check that an exported function has the expected
signature, avoids obviously unsafe runtime dependencies, returns finite numeric
loads, and survives a 24-hour pure-simulator run.

Examples:

```bash
python3 -m python.market_game_downstream.tests.export_check
```

```python
from python.market_game_downstream.rl.export import validate_submission_file

report = validate_submission_file(
    "python/market_game_downstream/rl/export/example_threshold_submission.py"
)
print(report)
```

`compute_demand_template.py` is intentionally small and self-contained. It is a
starting point for hand-written heuristics or distilled policies, not a training
entry point.
