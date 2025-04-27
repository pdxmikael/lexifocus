import argparse
import subprocess
import sys
import os

parser = argparse.ArgumentParser(description="Run LexiFocus with optional debug logging.")
parser.add_argument('--debug', action='store_true', help='Enable debug logging of prompts and responses')
args = parser.parse_args()

if args.debug:
    os.environ['LEXIFOCUS_DEBUG'] = '1'

# Get the directory of the run.py script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to app.py relative to the script directory
app_path = os.path.join(script_dir, "app.py")

# Command to run Chainlit
command = [sys.executable, "-m", "chainlit", "run", app_path, "-w"]

print(f"Running command: {' '.join(command)}")
try:
    # Run the command
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Chainlit: {e}", file=sys.stderr)
    sys.exit(1)
except FileNotFoundError:
    print("Error: Ensure Python and Chainlit are installed and accessible.", file=sys.stderr)
    sys.exit(1)
