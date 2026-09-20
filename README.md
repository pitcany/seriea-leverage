# Serie A forecast and season-scenario evaluation

This isolated revision repairs the scientific defects documented in the September 2026 review. The supplied manuscript and original review copy are preserved outside this directory. The active manuscript is paper/seriea-leverage-arxiv.tex; its tables and numerical claims are generated from revised results.

The question is whether simple structural forecasts improve specified combinations with closing prices, and whether their season probabilities are useful. A pooling weight is not a general information test. Match importance is established prior work and is not the causal value of resting a player.

## Main changes

- Dixon–Coles fitting uses a feasible parameterization covering all low-score cells and all modeled club pairings; prediction rejects invalid support.
- Shots on target use an independent-Poisson likelihood, venue-specific goals-per-shot conversion, and the shared unseen-club rate fallback.
- Closing-only benchmark with per-match provenance: Pinnacle before 23 July 2025, then Bet365 closing; average closing is an explicit fallback. Pre-closing snapshots are not silently mixed in.
- Exact pooling endpoints, a market-only recalibrator and a nested structural-logit comparison, and paired season-bootstrap sensitivity.
- Head-to-head ranking and the season-specific title/survival playoffs. Shootouts use an explicit 50/50 assumption.
- A retrospective all-club season evaluation at three dates across seven seasons, plus a separate prospective study design.
- Corrected interpretation, direct prior work, reproducible generated tables, and a Sloan-format abstract.

The old historical test period has already been inspected. Newly added analyses are retrospective and exploratory.

## Reproduce

Use Python 3.12.14 for the recorded environment. Other compatible versions may change numerical optimization results slightly.

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.lock
    export PYTHONPATH="$PWD/src"
    .venv/bin/python -m pytest
    .venv/bin/python scripts/tune_decay.py
    .venv/bin/python scripts/run_backtest.py
    .venv/bin/python scripts/run_shots_experiment.py
    .venv/bin/python scripts/revised_analysis.py
    .venv/bin/python scripts/benchmark_sensitivity.py
    .venv/bin/python scripts/validate_seasons.py
    .venv/bin/python scripts/analyze_survival_subgroup.py
    .venv/bin/python scripts/fiorentina_leverage.py
    .venv/bin/python scripts/analyze_fixture_stakes.py
    .venv/bin/python scripts/build_revised_paper.py
    .venv/bin/python scripts/write_sloan_abstract.py
    .venv/bin/python scripts/verify_reproduction.py
    cd paper
    latexmk -pdf -interaction=nonstopmode -halt-on-error seriea-leverage-arxiv.tex
    cd ..
    .venv/bin/python scripts/build_arxiv_bundle.py
    .venv/bin/python scripts/record_manifest.py

The scientific environment is already installed locally in .venv. All Python commands require PYTHONPATH=src unless the package is installed. The package also declares pyarrow, which is required for saving the forecasts.

Local data files are in data/raw/I1_0708.csv through I1_2526.csv. Downloaded source files are not included in Git. reports/data_manifest.json records their URLs, snapshot date and SHA-256 hashes. Check provider permissions before obtaining or redistributing data. Source changes can change the results even when filenames are identical.

The full decay sweep is the longest step. run_shots_experiment.py reuses the repaired goals forecast cache after checking the selected decay and revision diagnostics; rerun run_backtest.py after any change to the goals implementation or inputs. The manifest records code and output hashes for the completed run.

## Outputs

- paper/seriea-leverage-arxiv.pdf: revised paper.
- paper/seriea-sloan-bundle.zip: compiled paper, flat LaTeX sources, generated tables and figures, Sloan abstract, and rebuild instructions.
- paper/SLOAN_ABSTRACT.md: title and Introduction/Methods/Results/Conclusion, fewer than 500 words.
- reports/revised_scores.csv and revised_comparisons.csv: match scores and paired match/season intervals.
- reports/revised_calibration.csv and revised_reliability.csv: all three outcomes, uncertainty and bin counts.
- reports/market_provenance.csv and benchmark_sensitivity.csv: price source and sensitivity results.
- reports/season_validation_forecasts.csv: every club, origin and model; summary and paired comparisons in adjacent files.
- reports/survival_subgroup_membership.csv, survival_subgroup_scores.csv and survival_subgroup_comparisons.csv: exploratory bottom-eight versus other-twelve breakdown, defined using standings available at each origin. The settings file records eligibility rules, hashes and seven-season uncertainty.
- reports/fiorentina_projection.json and fiorentina_leverage.csv: corrected case study, conditional counts and Monte Carlo uncertainty.
- reports/fiorentina_stakes.csv: win-over-draw and draw-over-loss survival differences for all 12 fixtures, simulation-only intervals, break-even probability ratios, and separate hypothetical probability-transfer scenarios. reports/fiorentina_stakes_settings.json records the input hash and assumptions.
- reports/data_manifest.json and reproduction_manifest.json: local input, runtime, code and forecast hashes.
- VALIDATION_STUDY.md: a concrete future validation and action-study design.
- REVISION_SUMMARY.md: results, verification and remaining submission issues.

Raw and revised row-level forecasts are saved locally as Parquet. Legacy manuscripts, graphics and dashboards are retained under legacy/ for provenance and must not be used as current results.

## Important limits

The model forecasts can be up to 13 days older than closing prices. Season forecasts hold ratings fixed and do not propagate estimation uncertainty or future player changes. The league-average season comparator is simple. Three observations on one club-season are not independent; only seven seasons are available.

The survival study includes known Juventus sanctions in 2022/23, and the actual Verona–Spezia playoff survival labels. It does not anticipate future sanctions. Title venue exceptions, shootout skill and European qualification rules are not modeled. Top-six position is not synonymous with qualification.

The paper's draw-versus-win extension applies established conditional-importance methods. It assumes a redistribution of outcome probabilities that preserves conditional scorelines and all other fixtures. Its ratio and scenario intervals measure Monte Carlo error only, including the shared draw conditional; they exclude model uncertainty and do not identify an actual tactical or rotation effect. Each scenario changes one fixture separately.

## Submission and rights status

The GitHub publication target for this revision is `pitcany/seriea-leverage`, branch `sloan-scientific-revision`. No conference submission has been made. The conference requires a repository containing the research data; a hash manifest alone may not satisfy that requirement. Public source-data redistribution permission is not established, and raw downloads remain excluded from Git and the manuscript ZIP. The original repository has no general code license; the font license applies only to fonts. The author must choose a code license and resolve data sharing before calling the submission open source.

Competition guidance: https://www.sloansportsconference.com/research-paper-competition
Data provider notes: https://football-data.co.uk/data
