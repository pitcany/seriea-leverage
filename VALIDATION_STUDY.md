# Validation study for a stronger Sloan submission

This is a study design, not a claim that prospective validation has happened. The revised repository also contains an executed retrospective season study; its dates and models were selected after the historical outcomes had been inspected.

## Research question and contribution

Can a transparent model produce useful survival probabilities and quantify the consequences of *specified changes in match probabilities*, even when the model loses to the closing market on individual matches?

Match importance itself is established prior work. A stronger new contribution would show that an independently estimated action model, combined with season probabilities, changes decisions in a measurable and reliable way. A list of high-importance fixtures is insufficient.

## Completed retrospective study

- Every club in 2019/20–2025/26, at January 1, March 1 and May 1: 140 club-season observations per horizon.
- Fixed 1,460-day training history, strict prior-date information boundary, validated goals decay.
- Correct head-to-head tables; 2022/23 neutral playoff and subsequent two-leg survival playoff; known Juventus points deductions.
- 10,000 simulations per origin and model; league-average Poisson comparator using the same current table and fixtures.
- Survival Brier score, points CRPS, points mean absolute error, nominal 90% interval coverage; paired season-bootstrap comparisons.
- Reproduce with scripts/validate_seasons.py. Outputs are season_validation_forecasts.csv, season_validation_scores.csv and season_validation_comparisons.csv in reports.
- Interpretation: exploratory model validation. The comparator is deliberately simple, and only seven independent season clusters are available.

## Prospective forecast study

Freeze this protocol, code revision, model candidates, dates, inclusion rules and primary endpoint before collecting new evaluation outcomes. Retain the seven historical seasons for design and diagnostics only.

**Cohort.** Archive all eligible Serie A forecasts after protocol registration through the end of a complete future season. Prefer continued collection across multiple seasons; 20 teams sharing three relegation places do not provide 20 independent season outcomes. Expansion to other leagues requires predeclared league-specific rule modules and separate league reporting.

**Information times.** For match comparisons, issue both model and market forecasts at the same declared time, for example 24 hours before scheduled kickoff, with a separate closing-price comparison. Archive the actual price capture time, source, publication time of match results, scheduled kickoff and later rescheduling. Do not reconstruct an unavailable historical snapshot from a later price. The date-only historical data cannot establish this matched-time comparison.

**Frozen candidates.** Revised goals model; revised shots model; market; market-only recalibration; the nested structural-logit addition. Add an Elo or hierarchical strength baseline and a club-season empirical-rate baseline *before* freezing the prospective cohort. Select all parameters on historical development data. Log failures and missing prices; use a prespecified common-fixture sample and show exclusions.

**Match endpoint.** Primary: paired mean RPS difference between the nested structural-logit model and market-only recalibration. Secondary: log loss, all-outcome calibration, and comparisons with raw prices. A candidate minimum worthwhile RPS gain is 0.001, subject to an explicit decision-value justification before registration. Do not declare equivalence from a nonsignificant superiority test.

**Season endpoints.** Primary: survival Brier score at March 1. Secondary: January and May horizons, points CRPS, point-interval coverage, and calibration in the lower half of the table. Preserve all clubs in the main analysis. Define any at-risk subgroup using only current standings, e.g. the bottom eight at the origin. Future outcomes must not determine eligibility.

**Inference and power.** Resample whole seasons, or league-seasons when expanding. Repeated origins and teams in one league table remain in the same cluster. Use the historical cluster score differences to simulate power for the prespecified effect threshold, allowing an explicit between-season shift. Report the detectable effect with the available cohort rather than presenting one new season as decisive. Propagate fitted-parameter uncertainty in a declared sensitivity analysis; compare it with frozen-rating intervals.

**Forecast archive.** Save probabilities, input hashes, train cutoff, model revision, environment, competition rules, random seed, fit diagnostics and timestamp before outcomes. Store revised forecasts as new records. Never overwrite the forecast used for evaluation. A local archive is prepared by this repository; a public timestamped deposit still needs to be arranged.

## Action study: turning importance into a decision

Choose a concrete action, such as resting a specified player in one of two fixtures under a minutes budget. Estimate its effect on home and away scoring rates or W/D/L probabilities using a separately trained model with lineup, minutes, injuries, opponent strength, congestion and selection context. Those inputs are absent from the current aggregate dataset.

For a narrowly defined probability-transfer intervention that preserves conditional scoreline distributions and future matches, compute:

    change in survival = sum over W/D/L of (new probability - old probability)
                                                × conditional survival

Use that identity only for its stated intervention. General rest decisions can change score margins, future fatigue, injuries and opponents' choices. Simulate the changed score distributions and any independently estimated future effects rather than multiplying an arbitrary win-probability change by match importance.

Compare an importance-informed policy with fixture-difficulty-only and fixed-rest policies under the same minutes constraint. Report the expected gain, sensitivity to player-effect uncertainty, and the regret when the action model is misspecified. A simulated policy comparison is a model-based decision analysis, not an observed causal benefit.

A real causal evaluation would require a club partnership and an appropriate design. If actions are observational, state exchangeability/positivity assumptions, include the variables affecting selection, and evaluate sensitivity to unmeasured confounding. Do not invent a randomized experiment or infer rotation effects from one team's final standing.

## Priorities for the conference

1. Lead with the completed retrospective results, including failures or uncertain gains. The abstract must report actual results.
2. Make the practical contribution precise: validated season probabilities plus an explicit action model, not a renamed match-importance statistic.
3. Obtain the necessary source-data sharing permission, choose a code license, and prepare the accessible research repository. Data hashes and download scripts alone may not satisfy Sloan's stated repository requirement.
4. Register and begin the prospective archive before collecting its outcomes. It can strengthen a later manuscript but cannot be advertised as completed evidence in the current abstract.

Official competition guidance, checked 20 September 2026: abstract fewer than 500 words including title/body, sections Introduction/Methods/Results/Conclusion, up to two figures or tables combined; abstract due October 1 and invited manuscript due December 4, 2026, 11:59 p.m. Eastern Time. The rules page has some old-year boilerplate; confirm the invited manuscript template when supplied.

Source: https://www.sloansportsconference.com/research-paper-competition
