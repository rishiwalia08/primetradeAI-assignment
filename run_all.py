"""Run complete project pipeline with one command.

Usage:
    python run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

REQUIRED_FILES = [
    BASE_DIR / "historical_data (1).csv",
    BASE_DIR / "fear_greed_index.csv",
]

PHASE_SCRIPTS = [
    "preprocess_phase2.py",
    "eda_phase4.py",
    "insights_phase5.py",
    "phase6_modeling.py",
    "phase8_xai_shap.py",
]


def validate_inputs() -> None:
    missing = [p.name for p in REQUIRED_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset file(s): "
            + ", ".join(missing)
            + "\nPlace them in the same folder as run_all.py"
        )


def run_script(script_name: str) -> None:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 100)
    print(f"Running: {script_name}")
    print("=" * 100)

    completed = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed while running {script_name} (exit code: {completed.returncode})")



def main() -> None:
    print("Starting full pipeline...")
    validate_inputs()

    for script in PHASE_SCRIPTS:
        run_script(script)

    print("\n" + "=" * 100)
    print("All phases completed successfully.")
    print("Outputs include plots, insights, model diagnostics, and SHAP explanations.")
    print("See README.md and phase7_report.md for the final summary.")
    print("=" * 100)


if __name__ == "__main__":
    main()
