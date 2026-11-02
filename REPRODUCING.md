# Journal reproduction instructions

## Version and manuscript
Prepared 2026-11-02 from c5d771d12898d51f10a8fb7936c9d80a84cc8ae9.
Use the full manuscript paper/seriea-leverage-arxiv.pdf, its .tex, and
paper/seriea-leverage-arxiv.tar.gz (source plus all four referenced vector figures).
References are embedded in thebibliography; no separate .bib is required.

## Install and reproduce
Use Python 3.12 and a TeX Live installation providing pdflatex, latexmk,
lmodern, natbib, microtype, amsmath, booktabs and graphicx.

    python3.12 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[dev]"
    python scripts/reproduce.py
    python -m pytest

The entry point downloads/caches all 19 football-data.co.uk Serie A CSVs,
tunes decay, runs the main backtest, leverage and shots experiments, creates
figures and supplements, exports the extended pooling profile, builds the HTML
report and source bundle, and builds the full paper. It stops at the first failure.
Outputs overwrite generated reports; run in a fresh checkout to retain published
reference outputs for comparison. The source manuscript contains static numbers:
building it does not automatically update or verify those numbers.

For the main table only, after downloading data and with the supplied decay
settings, use python scripts/run_backtest.py. This is not the complete workflow.
Short paper build, independently: (cd paper && latexmk -pdf seriea-leverage.tex).
Full paper build: (cd paper && latexmk -pdf seriea-leverage-arxiv.tex).

## Expected checks
19 seasons 2007-08 through 2025-26; 7,220 matches; 2,280 validation and
2,660 test matches. Selected decay 0.002/day. Test market RPS 0.190541,
log loss 0.962038; calibrated Dixon-Coles RPS 0.197210, log loss 0.985787.
Do not require bitwise agreement across platforms. The paper documents
calibration and one-match accuracy sensitivity. Bootstrap and simulation
default seed: 20260810. Preserve and report optimization warnings.

## Remaining archive requirements
Raw CSVs are not tracked in Git. Archive the exact input snapshot with SHA-256
checksums and retrieval dates after checking redistribution rights. Do not
relicense third-party data without permission. Add a reviewed code licence
(the font licence covers fonts only) and a validated pinned environment before
acceptance. Lower bounds in pyproject.toml are not a lockfile.
Retain logs and compare all generated tables/figures to the paper, including the
negative-weight pooling claims; passing unit tests alone is insufficient.
