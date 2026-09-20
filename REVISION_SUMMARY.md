# Scientific revision completed

## Combined angles 1 and 2: points accuracy, survival forecasts and fixture stakes

The manuscript and Sloan abstract now use the title **Beyond Points Accuracy: Survival Forecasts and Fixture Stakes in Serie A**. The central result is improved final-points forecasting with unresolved survival-forecast gains. An exploratory bottom-eight-at-origin audit shows that the points improvement also occurs among clubs nearer the relegation contest. A separate Fiorentina illustration distinguishes the survival value of a win rather than a draw from that of a draw rather than a loss, using existing conditional probabilities and sample counts. The fixture table reports both differences and break-even ratios with approximate 95% Monte Carlo intervals. A short appendix documents the shared-draw covariance and Fieller calculation; the prior-art discussion cites Scarf and Shi (2008) and de Lorenzo, O'Bree and Grundy (2016).

- March bottom-eight points MAE is **3.24 versus 5.36**: difference **-2.13 [-2.74, -1.58]** points. Survival Brier is **0.1379 versus 0.1338**: difference **+0.0041 [-0.0143, +0.0226]**. Bottom-eight points improvements have intervals excluding zero at all three dates; survival differences include zero at all dates.
- Membership uses only points, goal difference and goals scored before each origin, known deductions, and alphabetical name for remaining ties. The split is exploratory, preserves both subgroups and all-club results, and resamples the same seven seasons. It adds no new fitted forecasts or season simulations.

- Lecce: draw-over-loss survival difference **4.82 percentage points**, win-over-draw **3.12**, break-even ratio **1.54 [1.08, 2.19]**.
- Lazio: corresponding differences **1.99** and **3.12**, ratio **0.64 [0.34, 1.05]**.
- Moving five percentage points from draws to wins and another five to losses increases expected league points by **0.05**. Fiorentina's simulated survival change is **-0.085 percentage points [-0.156, -0.014]** in the Lecce scenario, and **+0.057 [-0.006, +0.119]** in the Lazio scenario. The latter interval includes zero.
- Scenarios change one fixture separately and preserve conditional scorelines, all other fixtures and the playoff model. Intervals measure simulation error only. No new tactic, player effect or policy improvement is estimated.
- Reproduce with `scripts/analyze_fixture_stakes.py` before the manuscript generators. `reports/fiorentina_stakes.csv` includes all 12 fixtures; the adjacent settings file includes input provenance and assumptions.
- **177 tests pass**, including 24 tests for conditional stakes and 13 for the origin-defined subgroup analysis. Components, covariance, Fieller endpoints, scenario variances, membership joins, subgroup scores and paired differences are cross-checked from saved inputs. Original match forecasts and season results remain the basis of the study.
- The combined Sloan abstract contains **352 words**, including title and headings.

Work is isolated in /Users/yannik/Work/seriea-sloan-revision on branch sloan-scientific-revision. The supplied paper in /Users/yannik/Work/seriea-arxiv-corrected and the original review copy were preserved. The user has authorized committing and pushing this revision to `pitcany/seriea-leverage`. No conference submission has been made. The manuscript ZIP contains the PDF, flat sources, generated tables and figures, abstract and rebuild instructions; raw source-data files remain excluded.

## What changed

1. **Valid count likelihoods.** Dixon–Coles now optimizes a fraction of an admissible rho interval covering all low-score cells and modeled rating pairings. Predictions reject invalid support. The shots model fixes rho=0 and shares the goals model's rate/fallback path. The paper explicitly acknowledges that changing the shots likelihood changes the experiment.
2. **Correct price benchmark.** Closing columns are ingested directly. Pinnacle is excluded from 23 July 2025; 2025/26 uses Bet365 closing. Each match records source and timing. Pre-closing snapshots are not substituted.
3. **Scoped information claims.** Zero or positive pooling weight is no longer interpreted as a general information test. A nested logit model compares added goals/shots forecasts against a market-only recalibrator. Exact endpoints are considered in scalar pooling.
4. **Correct season rules.** Head-to-head mini-tables, multi-team ties, drawing lots, 2022/23 neutral playoffs, and subsequent two-leg survival playoffs are implemented. Equal aggregate goals lead directly to penalties. Shootouts are explicitly modeled 50/50.
5. **Broader validation.** All 20 clubs are evaluated at three dates in each of seven seasons. Known Juventus sanctions and the actual Verona–Spezia survival outcome are included.
6. **Revised inference and presentation.** Paired season-bootstrap sensitivities, all-class calibration intervals and bin counts, direct prior work, corrected decay units, and a clear distinction between match importance and rotation value.
7. **Reproduction.** Exact dependency versions, pyarrow dependency, source/forecast hashes, saved origin diagnostics, generated manuscript numbers and a verified flat source bundle. Superseded materials are in legacy/.

## Results that now support the paper

| Comparison | Revised estimate | 95% season-bootstrap interval |
|---|---:|---:|
| Calibrated goals minus closing market, RPS | +0.006836 | [+0.005960, +0.008050] |
| Goals/shots pool minus raw goals, RPS | -0.000555 | [-0.001222, +0.000236] |
| Added structural logits minus market recalibration, RPS | -0.000117 | [-0.000400, +0.000120] |
| March survival Brier: goals model minus league-Poisson baseline | +0.001601 | [-0.005780, +0.009029] |
| March final-points MAE: goals model minus baseline | -1.9745 points | [-2.6226, -1.4004] |

Lower is better. The seven season clusters limit interval precision; these intervals condition on fitted forecasts. New analyses were developed after inspecting the historical test period and are described as retrospective.

- The validation sweep again selects **0.002/day**, a half-life of about 346.6 days.
- Closing-market test RPS: **0.190374**; calibrated goals: **0.197210**; repaired shots: **0.198711**.
- The goals/shots pool selects weights **0.60/0.40**; market-containing nonnegative pools select the market alone.
- The nested market/structural combination scores **0.189736**, versus **0.189853** for market recalibration. Its advantage is small and uncertain.
- Structural points MAE and CRPS improve against the league-average baseline at all three dates; survival Brier differences include zero at every date.
- March final-points MAE is **3.48 versus 5.45**. March survival Brier is **0.05515 versus 0.05355**. Better points forecasts have not demonstrated better survival probabilities.
- Corrected Fiorentina survival is **96.00%**, versus **96.405%** under simplified ranking on the same simulated league draws. Survival playoffs occur in **11.235%** of simulated seasons league-wide.
- Fiorentina's highest importance fixture is Lecce: **7.94 percentage points**, versus **3.49** for Inter. These are win-versus-loss conditional differences, not costs of resting a player.

## Verification

- **177 tests pass**, including regressions for unobserved invalid probability cells, unseen-club rates, stale prices, exact pooling endpoints, head-to-head ordering, multi-team ties, playoff formats, penalties instead of away goals, negative probability rejection, conditional-stakes calculations and prior-only subgroup eligibility.
- **262/262 goals origins and 262/262 shots origins converged.** Minimum forecast tau is 0.370476 for goals and exactly 1 for shots.
- Re-ran the seven-rate validation sweep, both full match experiments, the nested calibration analysis, benchmark sensitivities, the 20,000-season Fiorentina case and the all-club season study.
- **840 model/club/origin forecasts** in the season study: 420 situations evaluated by two models, with 10,000 season draws per model and origin.
- Recomputed every revised score-table entry from its saved test probabilities.
- Verified the 7,220-match corpus, complete seasonal counts, no duplicate fixtures, and the single missing shots record.
- The original revised Sloan abstract contained **303 words**; the combined 352-word abstract above supersedes it.
- The manuscript source bundle compiles in an isolated flat directory. Final PDF pages were rendered and inspected.

## What still limits acceptance

The statistical and implementation defects are substantially repaired. **The main remaining research risk is practical impact and novelty:** the study does not demonstrate a reliable improvement in survival forecasts over its simple baseline, or an actual benefit from player rotation. The points-forecast result is stronger but does not answer those other questions.

The manuscript should therefore be presented as a transparent validation study with mixed results, not as proof of unpriced-information absence or a validated lineup policy. VALIDATION_STUDY.md lays out the stronger prospective and action-model work, with explicit endpoints, timing, baselines and inference.

Two submission matters remain unresolved: the author must choose a code license and establish permission for the required public data sharing. The existing font license does not license the code. A source-data hash manifest is useful for reproduction but may not satisfy Sloan's data-repository requirement. No permission request has been sent to anyone.

Official guidance: https://www.sloansportsconference.com/research-paper-competition
