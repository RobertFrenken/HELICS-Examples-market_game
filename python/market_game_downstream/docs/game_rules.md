# Market Game Rules Compatibility Note

The downstream HELICS rule document is retired. The maintained rule references
are now:

- `python/market_game/market_game.md` for the classroom/CTF game rules and the
  canonical HELICS house interface;
- `python/market_game_downstream/docs/rule_inventory.md` for the current map of
  duplicated rule behavior and canonical implementations;
- `python/market_game_downstream/docs/invalid_demand_behavior.md` for
  invalid-demand clamping and penalty behavior;
- `python/market_game_downstream/rl/README.md` for local RL scenario and
  training commands.

Do not add a new `python/market_game_downstream/helics` runtime directory.
Local HELICS validation should use `python/market_game` directly with exported
or hand-written `compute_demand(...)` submissions.
