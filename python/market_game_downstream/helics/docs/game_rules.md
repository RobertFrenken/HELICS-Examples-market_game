# Market Game Rules Compatibility Note

The downstream HELICS rule document is retired. The maintained rule references
are now:

- `python/market_game/market_game.md` for the classroom/CTF game rules and the
  canonical HELICS house interface;
- `python/market_game_downstream/docs/rule_inventory.md` for the current map of
  duplicated rule behavior and canonical implementations;
- `python/market_game_downstream/docs/invalid_demand_behavior.md` for
  invalid-demand clamping and penalty behavior;
- `python/market_game_downstream/rl/docs/scenario_config_schema.md` for local
  simulator and RL scenario configuration.

Do not add new runtime logic under `python/market_game_downstream/helics`.
Local HELICS validation should use `python/market_game` directly, including the
generated runner configs from `python.market_game_downstream.rl.helics_config`
when evaluating exported or hand-written `compute_demand(...)` submissions.
