"""CLI wrapper. `python run.py rfps/rfp_01.json` → pretty JSON to stdout + out/.

The eval suite (test_evals.py) imports run_agent directly, so this is just
for manual runs / the live demo.
"""
import json
import os
import sys

from agent import run_agent


def main():
    if len(sys.argv) < 2:
        print("usage: python run.py <rfp.json> [--quiet]", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    quiet = "--quiet" in sys.argv[2:]

    with open(path) as f:
        rfp = json.load(f)

    result = run_agent(rfp, verbose=not quiet)

    os.makedirs("out", exist_ok=True)
    out_path = os.path.join("out", os.path.basename(path))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
