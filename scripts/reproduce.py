"""Regenerate research outputs in dependency order; run from any directory."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    steps = [
        [sys.executable, "-c", "from seriea.data.download import download_all; download_all()"],
        *[[sys.executable, str(ROOT / "scripts" / name)] for name in (
            "tune_decay.py", "run_backtest.py", "fiorentina_leverage.py",
            "make_figures.py", "run_shots_experiment.py", "build_artifact.py",
            "paper_supplements.py", "pool_profile_extended.py", "build_arxiv_bundle.py"
        )],
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "seriea-leverage-arxiv.tex"],
    ]
    for command in steps:
        subprocess.run(command, cwd=ROOT / "paper" if command[0] == "latexmk" else ROOT, check=True)

if __name__ == "__main__":
    main()
