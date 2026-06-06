import os
import fnmatch
import json
import argparse


parser = argparse.ArgumentParser(description="run the neighborhood in standalone mode")
parser.add_argument("folder", type=str, nargs="?", default="houses", help="folder to search for house files")
parser.add_argument("--pattern", type=str, default="*_house.py", help="pattern to match house files")
parser.add_argument("--profile", type=str, default="profile1", help="type of load profile to use: flat, spike, dspike, random, profile1, profile_solar")

args = parser.parse_args()
    
houses_dir = args.folder
pattern = args.pattern

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
houses_path = houses_dir if os.path.isabs(houses_dir) else os.path.join(script_directory, houses_dir)
houses_path = os.path.abspath(houses_path)
houses_dir_for_exec = os.path.relpath(houses_path, script_directory).replace("\\", "/")
houses_module_prefix = houses_dir_for_exec.replace("/", ".") if houses_dir_for_exec != "." else ""

# Scan the directory for matching files
house_files = sorted(f for f in os.listdir(houses_path) if fnmatch.fnmatch(f, pattern))

# Create a JSON object
runner = {"name": "house_evaluation"}
runner["federates"]=[]

broker = {
    "directory": script_directory,
    "host": "localhost",
    "name": "broker",
    "exec": f"helics_broker -f {len(house_files) + 1} -t zmqss --ipv4 -p 23404 --loglevel=warning",
}
runner["federates"].append(broker)

for house_file in house_files:
    module_name = house_file[:-3]
    house_exec_target = f"{houses_module_prefix}.{module_name}" if houses_module_prefix else module_name
    federate={"directory":script_directory,
              "host":"localhost",
              "name":house_file[:-9],
              "exec":f"python -u -m {house_exec_target} --broker localhost:23404 --no-plot"
              }
    runner["federates"].append(federate)



# Print or use the JSON object

federate = {
    "directory": script_directory,
    "host": "localhost", 
    "name": "market_maker",
    "exec": f"python -u market_maker.py --auto --broker localhost:23404 --no-plot --profile {args.profile}"}

runner["federates"].append(federate)

with open("houses.json", "w") as f:
    json.dump(runner, f, indent=3)
    
