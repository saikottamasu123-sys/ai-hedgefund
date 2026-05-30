"""
run.py

Quick CLI entry point. Run the committee from the terminal:

    python run.py NVDA
    python run.py AAPL
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from workflows.committee_graph import run_committee


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    state = run_committee(ticker)
    print(state["final_report"])

    if state["errors"]:
        print("\n⚠️  Non-fatal errors during run:")
        for err in state["errors"]:
            print(f"  • {err}")


if __name__ == "__main__":
    main()
