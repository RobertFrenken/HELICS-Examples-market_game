# Market Game Upstreaming Status

Read this file first in future Codex sessions. It is the current source of
truth for the relationship between the comprehensive/source repo and the clean
fork worktree used for upstream PRs.

## Repositories

Comprehensive/source repo:

```text
/home/rober/HELICS-Examples-market_game
remote: RobertFrenken/HELICS-Examples-market_game
branch: market_game
purpose: source of broader downstream code, experiments, notes, and future PR material
```

Clean fork worktree for upstream PR branches:

```text
/home/rober/HELICS-Examples-market_game-pr-invalid-demand
remote: RobertFrenken/HELICS-Examples
current branch: market-game-simulator-parity
purpose: clean worktree for branches that target GMLC-TDC/HELICS-Examples
```

Upstream target:

```text
repo: GMLC-TDC/HELICS-Examples
base branch: market_game
```

## Current State

Accepted upstream PR:

- Scope: invalid demand handling and focused validation checks.
- Upstream commit: `826b5a7 Fix market game invalid demand handling (#134)`.
- Result: `RobertFrenken/HELICS-Examples:main` and
  `GMLC-TDC/HELICS-Examples:market_game` were aligned to `826b5a7`.

Current PR branch:

- Branch: `market-game-simulator-parity`.
- Commit: `cb7e213 Add pure-Python market game simulator checks`.
- Fork remote: `RobertFrenken/HELICS-Examples`.
- PR target: `GMLC-TDC/HELICS-Examples:market_game`.
- Scope: dependency-free pure-Python simulator, parity checks, and simulator
  docs only.
- PR compare URL:
  `https://github.com/GMLC-TDC/HELICS-Examples/compare/market_game...RobertFrenken:market-game-simulator-parity?expand=1`

## Porting Rule

Use `/home/rober/HELICS-Examples-market_game` as the comprehensive source of
ideas and code.

Use `/home/rober/HELICS-Examples-market_game-pr-invalid-demand` as the clean
fork worktree for PR branches.

Do not upstream `python/market_game_downstream` directly. Port selected pieces
into upstream-shaped paths under `python/market_game`.

Each PR should be small, reviewable, and dependency-conscious. Avoid bundling
RL, Gymnasium, RLlib, export validators, scenario curricula, and broad
downstream handoff docs into early PRs.

## Completed PRs

1. Invalid-demand runtime fix.
2. Pure-Python simulator and parity checks.

## Current PR Scope

The simulator/parity PR added these upstream-shaped paths in the clean fork
worktree:

```text
python/market_game/docs/simulation.md
python/market_game/simulation/
python/market_game/tests/check_simulation.py
python/market_game/tests/check_simulation_core.py
python/market_game/tests/check_simulation_parity.py
```

Validation commands:

```bash
python3 python/market_game/tests/check_simulation.py
python3 python/market_game/tests/check_battery_validation.py
git diff --cached --check
```

## Next Planned PR

Next recommended PR: baseline evaluator.

Goal:

- Make the simulator immediately useful by running built-in baseline strategies
  and reporting costs, loads, final battery state, and clamps.
- Keep this dependency-free.
- Do not include Gymnasium, RLlib, trained models, export validators, or broad
  scenario curricula yet.

Likely source material from the comprehensive repo:

```text
python/market_game_downstream/rl/agents/policies.py
python/market_game_downstream/rl/evaluate.py
possibly small pieces from python/market_game_downstream/rl/core/
```

Likely upstream destination in the clean fork branch:

```text
python/market_game/simulation/policies.py
python/market_game/simulation/evaluate.py
python/market_game/tests/check_simulation_evaluate.py
python/market_game/docs/simulation.md
```

Expected CLI shape:

```bash
python3 -m python.market_game.simulation.evaluate
```

Expected output shape:

```text
house,total_load,total_cost,final_battery,clamps
```

## Future PR Sequence

1. Baseline evaluator.
2. Scenario configs.
3. Strategy workbench / local submitted-policy runner.
4. Lightweight feature helpers.
5. Optional Gymnasium adapter.
6. Submission/export validator.
7. RL training examples only if maintainers explicitly want them.

## Session Startup Instruction

At the start of a future session, tell Codex:

```text
Read market_game_upstreaming_status.md first. Use it as the source of truth for
what has been accepted, what is currently in PR, and what to port next.
```

After each PR, update this file with:

- PR URL.
- Branch name.
- Commit SHA.
- Accepted/not accepted status.
- Scope that actually landed.
- Any changes to the next planned PR.
