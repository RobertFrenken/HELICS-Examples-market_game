import argparse
import fnmatch
import json
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run the neighborhood in standalone mode")
    parser.add_argument("folder", type=str, nargs="?", default="houses", help="folder to search for house files")
    parser.add_argument("--pattern", type=str, default="*_house.py", help="pattern to match house files")
    parser.add_argument(
        "--profile",
        type=str,
        default="profile1",
        help="type of load profile to use: flat, spike, dspike, random, profile1, profile_solar",
    )
    parser.add_argument(
        "--launcher",
        choices=("uv", "plain"),
        default="uv",
        help="command prefix for generated federates: uv uses 'uv run', plain uses the active environment",
    )
    return parser


def launcher_prefix(launcher: str) -> str:
    return "uv run " if launcher == "uv" else ""


def resolve_houses_path(script_directory: str, houses_dir: str) -> str:
    if os.path.isabs(houses_dir):
        return os.path.abspath(houses_dir)
    return os.path.abspath(os.path.join(script_directory, houses_dir))


def module_prefix(script_directory: str, houses_path: str) -> str:
    houses_dir_for_exec = os.path.relpath(houses_path, script_directory).replace("\\", "/")
    if houses_dir_for_exec == ".":
        return ""
    return houses_dir_for_exec.replace("/", ".")


def house_module_name(prefix: str, house_file: str) -> str:
    module_name = house_file[:-3]
    return f"{prefix}.{module_name}" if prefix else module_name


def build_runner(script_directory: str, house_files: list[str], prefix: str, profile: str, launcher: str) -> dict:
    command_prefix = launcher_prefix(launcher)
    runner = {"name": "house_evaluation", "federates": []}
    runner["federates"].append(
        {
            "directory": script_directory,
            "host": "localhost",
            "name": "broker",
            "exec": f"{command_prefix}helics_broker -f {len(house_files) + 1} -t zmqss --ipv4 -p 23404 --loglevel=warning",
        }
    )

    for house_file in house_files:
        runner["federates"].append(
            {
                "directory": script_directory,
                "host": "localhost",
                "name": house_file[:-9],
                "exec": (
                    f"{command_prefix}python -u -m {house_module_name(prefix, house_file)} "
                    "--broker localhost:23404 --no-plot"
                ),
            }
        )

    runner["federates"].append(
        {
            "directory": script_directory,
            "host": "localhost",
            "name": "market_maker",
            "exec": (
                f"{command_prefix}python -u market_maker.py --auto --broker localhost:23404 "
                f"--no-plot --profile {profile}"
            ),
        }
    )
    return runner


def main() -> None:
    args = build_parser().parse_args()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    houses_path = resolve_houses_path(script_directory, args.folder)
    prefix = module_prefix(script_directory, houses_path)
    house_files = sorted(f for f in os.listdir(houses_path) if fnmatch.fnmatch(f, args.pattern))
    runner = build_runner(script_directory, house_files, prefix, args.profile, args.launcher)

    with open("houses.json", "w") as f:
        json.dump(runner, f, indent=3)


if __name__ == "__main__":
    main()
