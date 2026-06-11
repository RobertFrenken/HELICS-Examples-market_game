# Market Game Environment Setup

This guide covers the Python environment needed to run the HELICS market game
on Windows, Linux, and macOS. It includes two supported setup paths:

| Path | Best for | How dependencies are controlled |
| --- | --- | --- |
| `uv` | Most users, classes, workshops, repeatable local runs | `pyproject.toml` and `uv.lock` |
| `pip` with `venv` | Users who cannot install `uv` | A manually activated virtual environment |

The important rule is that every HELICS process must use the same Python
environment. The market game launches several processes: one broker, one market
maker, and one process per house. If the parent command uses one Python
environment but `houses.json` launches plain `python` from another environment,
the run may fail or behave differently across machines.

## Requirements

| Requirement | Recommended | Notes |
| --- | --- | --- |
| Python | `3.11` or `3.12` | Newer versions may work, but package wheels can lag behind new Python releases. |
| Package setup | `uv sync --extra helics` | Recommended because it installs from the project lockfile. |
| HELICS package | `helics[cli]==3.6.1` | The `[cli]` extra provides CLI support used by `helics run`. |
| Plotting package | `matplotlib==3.10.9` | Only needed for optional plots. Generated runs use `--no-plot`. |

## Install uv

If `uv` is already installed, skip to [Recommended uv Setup](#recommended-uv-setup).

The commands below come from Astral's official uv installation guide:
<https://docs.astral.sh/uv/getting-started/installation/>.

| System | Recommended command | Notes |
| --- | --- | --- |
| Linux or macOS | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | Uses the standalone installer. No existing Python installation is required. |
| Linux or macOS without `curl` | `wget -qO- https://astral.sh/uv/install.sh \| sh` | Same installer using `wget`. |
| Windows PowerShell | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` | Run in PowerShell. Open a new terminal after installation if `uv` is not found. |
| Homebrew | `brew install uv` | Useful on macOS or Linux if Homebrew manages developer tools on the machine. |
| WinGet | `winget install --id=astral-sh.uv -e` | Useful on managed Windows systems with WinGet enabled. |
| Scoop | `scoop install main/uv` | Useful if Scoop is already your Windows package manager. |
| pipx | `pipx install uv` | Isolates uv as a Python CLI tool, but requires Python and pipx first. |
| pip | `pip install uv` | Works, but is less clean because it installs the tool that manages Python into an existing Python environment. |

After installing, verify:

```bash
uv --version
```

If the command is not found, close and reopen the terminal so your updated
`PATH` is loaded. On Windows, use a new PowerShell window.

## Recommended uv Setup

Run these commands from the repository root:

```bash
uv sync --extra helics
```

Then run the market game from `python/market_game`:

```bash
cd python/market_game
uv run python run_neighborhood.py houses
uv run helics run --path=houses.json
```

`run_neighborhood.py` generates `houses.json` with `uv run` in every federate
command. That keeps the broker, market maker, and all houses inside the same
uv-managed environment.

Verify the environment before running the full federation:

```bash
uv run python -c "import helics as h; print(h.helicsGetVersion())"
uv run helics --version
```

Expected result: both commands report HELICS `3.6.1`.

### uv With A Specific Python Version

Use this if the default Python on the machine is too old or too new:

```bash
uv sync --python 3.11 --extra helics
```

You can also use `3.12`:

```bash
uv sync --python 3.12 --extra helics
```

## pip With venv

Use this path if you cannot use `uv`. The commands install the same practical
dependency set, but the environment is controlled manually by activating the
virtual environment before generating and running `houses.json`.

| System | Create and activate venv | Install dependencies |
| --- | --- | --- |
| Linux or macOS | `python3.11 -m venv .venv` then `source .venv/bin/activate` | `python -m pip install --upgrade pip` then `python -m pip install "helics[cli]==3.6.1" "matplotlib==3.10.9" "gymnasium==1.2.2" "numpy>=1.26"` |
| Windows PowerShell | `py -3.11 -m venv .venv` then `.\.venv\Scripts\Activate.ps1` | `python -m pip install --upgrade pip` then `python -m pip install "helics[cli]==3.6.1" "matplotlib==3.10.9" "gymnasium==1.2.2" "numpy>=1.26"` |

Then run from `python/market_game`:

```bash
python run_neighborhood.py houses --launcher plain
helics run --path=houses.json
```

Use `--launcher plain` for pip environments. It generates plain `python` and
`helics_broker` commands in `houses.json`, which will resolve through the
currently activated virtual environment.

### Windows Activation Policy

If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then open a new PowerShell window and activate the environment again.

## Launcher Modes

`run_neighborhood.py` controls the commands written into `houses.json`.

| Launcher | Generate command | Example `houses.json` command | Use when |
| --- | --- | --- | --- |
| `uv` | `uv run python run_neighborhood.py houses` | `uv run python -u -m houses.price_aware_house ...` | You installed dependencies with `uv sync --extra helics`. |
| `plain` | `python run_neighborhood.py houses --launcher plain` | `python -u -m houses.price_aware_house ...` | You activated a pip virtual environment. |

Regenerate `houses.json` whenever you change launchers, Python environments, or
the set of house files.

## Running Different House Sets

By default, the runner includes files matching `*_house.py`:

```bash
uv run python run_neighborhood.py houses
```

To include a different pattern:

```bash
uv run python run_neighborhood.py houses --pattern "*house*.py"
```

With an activated pip environment:

```bash
python run_neighborhood.py houses --pattern "*house*.py" --launcher plain
```

## Environment Checks

Use these before debugging HELICS behavior. They confirm that commands resolve
from the environment you expect.

| Environment | Check Python | Check HELICS CLI | Check broker |
| --- | --- | --- | --- |
| uv | `uv run python -c "import sys; print(sys.executable)"` | `uv run helics --version` | `uv run helics_broker --version` |
| Linux or macOS venv | `which python` | `which helics` | `which helics_broker` |
| Windows PowerShell venv | `Get-Command python` | `Get-Command helics` | `Get-Command helics_broker` |

For pip/venv, the commands should resolve under `.venv`. For uv, the Python
executable should resolve under the project's `.venv`.

## Smoke Tests

From the repository root with uv:

```bash
uv run python python/market_game/tests/check_all.py
uv run python -m python.market_game_downstream.tests.check_all
```

With an activated pip environment:

```bash
python python/market_game/tests/check_all.py
python -m python.market_game_downstream.tests.check_all
```

The second command uses `gymnasium`, so install the full dependency list shown
above before running it.

## Common Failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'helics'` | A federate is using a Python environment without HELICS installed. | Use `uv run ...`, or activate the venv and regenerate `houses.json` with `--launcher plain`. |
| `helics_broker` not found | CLI support is missing or the environment is not active. | Install `helics[cli]==3.6.1`, then verify `helics_broker` resolves from the environment. |
| Parent command works, federates fail | `houses.json` launches a different Python than the parent shell. | Regenerate `houses.json` with the right launcher mode. |
| Port already in use | A previous broker is still running on port `23404`. | Stop the old process or change the port in `run_neighborhood.py`, then regenerate `houses.json`. |
| Run blocks on a plot window | A script was started without `--no-plot`. | Use the generated runner, or add `--no-plot` when running house scripts manually. |
| pip install cannot find a wheel | Python version or platform does not have a compatible prebuilt package. | Try Python `3.11` or `3.12`, or use uv with `uv sync --python 3.11 --extra helics`. |

## Expected End-to-End Result

For the stock `profile1` run with the included houses, the market-maker log
should end with totals close to:

```text
fed FlattenDemandHouse:total Cost=$35.446666666666665 total consumption=127.0
fed FullCycleHouse:total Cost=$47.74666666666667 total consumption=120.0
fed PriceAwareHouse:total Cost=$21.953333333333333 total consumption=125.0
the winner is Fed PriceAwareHouse total cost=$21.953333333333333
```

Small formatting differences are fine. Large numeric differences usually mean
different houses, a different profile, or a different code version.
