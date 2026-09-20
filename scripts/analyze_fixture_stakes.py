"""Derive draw-aware Fiorentina stakes from the saved conditional-probability CSV.

Run: ``PYTHONPATH=src python scripts/analyze_fixture_stakes.py``.
No model is refitted and no new seasons are simulated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from seriea.config import PROJECT_ROOT, REPORTS_DIR
from seriea.evaluation.fixture_stakes import analyze_fixture_stakes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPORTS_DIR / "fiorentina_leverage.csv")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    arguments = parser.parse_args()
    source = arguments.input.resolve()
    raw = source.read_bytes()
    frame = pd.read_csv(source, float_precision="round_trip")
    result = analyze_fixture_stakes(frame)
    try:
        source_label = str(source.relative_to(PROJECT_ROOT))
    except ValueError:
        source_label = str(source)
    settings = {
        "schema_version": 1,
        "input_file": source_label,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "output_file": "fiorentina_stakes.csv",
        "n_fixtures": len(result),
        "scope": {
            "club": "Fiorentina", "season": "2025/26", "forecast_origin": "2026-03-01",
            "design": "Retrospective, single-club illustration using the existing saved simulation.",
            "generalization": "Does not establish effects for other clubs, origins, or seasons.",
        },
        "units": {
            "probabilities_and_contrasts": "Probability units (0-1); multiply by 100 for percentage points.",
            "mc_covariance": "Squared probability units.",
            "ratio": "Dimensionless.", "expected_points": "League points.",
        },
        "definitions": {
            "win_over_draw": "U = q_win - q_draw",
            "draw_over_loss": "D = q_draw - q_loss",
            "total_importance": "U + D = q_win - q_loss",
            "draw_loss_minus_win_draw": "D - U",
            "draw_loss_to_win_draw_ratio": "D / U; undefined when U = 0.",
            "ratio_is_positive_upside_threshold": "Only when U > 0 does a/b > D/U imply positive survival change for b > 0.",
            "component_mc_covariance": "Cov(U,D) = -q_draw*(1-q_draw)/n_draw",
        },
        "uncertainty": {
            "confidence_level": 0.95,
            "component_variance": "Var(q_r) = q_r*(1-q_r)/n_r, conditional on category counts.",
            "component_and_contrast_intervals": "Approximate normal intervals; not clipped to admissible bounds.",
            "ratio_interval": "Fieller confidence set from the correlated normal approximation for (D,U).",
            "ratio_interval_encoding": "ratio_mc_ci_intervals is JSON; null lower/upper endpoint means negative/positive infinity. Empty array with status undefined differs from status empty. Low/high columns represent a single interval only.",
            "includes": "Monte Carlo sampling error conditional on fitted forecasts and category counts only.",
            "excludes": "Model-estimation, model-specification, future-state and causal-intervention uncertainty.",
            "boundary_warning": "Binomial plug-in normal approximations can be unreliable with sparse counts or q near 0/1; q exactly 0/1 yields zero estimated variance, not proof of certainty.",
            "rank_warning": "Ranks are descriptive point-estimate orderings, with no multiplicity correction or joint rank inference. Fixtures share simulated seasons.",
        },
        "scenario": {
            "draw_to_win": 0.05, "draw_to_loss": 0.05,
            "probability_change_W_D_L": [0.05, -0.10, 0.05],
            "baseline_probabilities": "Empirical n_r/(n_win+n_draw+n_loss) from the input CSV.",
            "feasibility": "a+b <= empirical draw probability; infeasible rows suppress scenario effects.",
            "delta_expected_points": "2*a-b = 0.05, using win/draw/loss league points 3/1/0.",
            "delta_survival": "a*U-b*D",
            "delta_survival_variance": "a^2*Var(U)+b^2*Var(D)-2*a*b*Cov(U,D)",
            "uncertainty_scope": "Fixed probability transfers; empirical outcome-frequency uncertainty is not propagated.",
        },
        "assumptions": [
            "The simulator uses independent fixtures and frozen fitted parameters.",
            "Hypothetical reweighting preserves scoreline distributions conditional on W/D/L.",
            "All other fixtures, league rules and the playoff model remain fixed.",
            "No feasible tactical, lineup, rotation or in-game intervention is estimated.",
            "These consequences do not identify the causal value of a league point or a player.",
        ],
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output = arguments.output_dir / "fiorentina_stakes.csv"
    result.to_csv(output, index=False, float_format="%.17g")
    settings_path = arguments.output_dir / "fiorentina_stakes_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(result)} fixture summaries to {output} and {settings_path}.")


if __name__ == "__main__":
    main()
