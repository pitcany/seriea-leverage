# Serie A: calibrated forecasting and match leverage

A reproducible rebuild of *Soccer Match Prediction in the Serie A* (Y. Pitcan,
accepted for publication, International Journal of Computer Applications, 2018;
written 2016), extended from 10 seasons to 19 and re-posed around the question a
football club would actually ask.

**[Illustrated report →](https://claude.ai/code/artifact/a86d8eca-e4d9-4fd1-92e7-b020415a5b16)**  ·  **[Short paper →](paper/seriea-leverage.pdf)**  ·  **[Full paper →](paper/seriea-leverage-arxiv.pdf)**

The original study reported 53% three-way accuracy and compared that favourably
against betting-market accuracy of 55.3%. This rebuild reaches the same
conclusion the original was one step away from, and states it plainly:

> **A structural match-outcome model adds nothing to a de-vigged closing price.**
> The optimal weight on Dixon-Coles in a logarithmic pool with the market is
> 0.000 — on validation *and* on test. So the useful product is not a better
> outcome forecast. It is what you build on top of a calibrated one.

That "on top" is **match leverage**: how much a single fixture moves a club's
probability of achieving its season objective.

---

## Headline results

Walk-forward, 2019-20 to 2025-26 (2,660 matches). Lower RPS is better.

| Model | RPS | 95% CI | log loss | accuracy |
|---|---|---|---|---|
| Market (Shin de-vigged) | **0.1905** | [0.1856, 0.1957] | 0.9620 | 54.8% |
| Market + Dixon-Coles pool | 0.1905 | [0.1856, 0.1957] | 0.9620 | 54.9% |
| Dixon-Coles (calibrated) | 0.1972 | [0.1921, 0.2024] | 0.9858 | 53.4% |
| Base rate | 0.2318 | [0.2292, 0.2344] | 1.0846 | 40.6% |
| Uniform (the 2016 benchmark) | 0.2340 | [0.2312, 0.2367] | 1.0986 | 40.6% |

Paired bootstrap, Dixon-Coles minus market: **+0.0067 RPS [+0.0046, +0.0088]** —
the model is significantly *worse* than the market, not comparable to it.

Three points worth drawing out:

1. **The right baseline is 40.6%, not 33%.** Home advantage alone gets you to
   the base rate. Measuring against uniform guessing inflates the apparent edge
   by roughly seven percentage points.
2. **A properly built model in 2026 gets 53.4% accuracy** — statistically
   indistinguishable from the 2016 study's 53.0%. Nine extra seasons, a
   principled likelihood and tuned time decay bought no accuracy. That is
   informative: three-way match accuracy is close to its ceiling.
3. **Calibration and discrimination are different things.** The model is
   *better calibrated* than the market on the home-win margin (slope 0.995 vs
   1.103) while being decisively less sharp. Fitted temperature was 0.99, i.e.
   no correction needed. The market's advantage is information, not honesty.

## Is it just that goals are noisy?

The obvious objection to the headline: goals are the noisiest thing a football
match produces, so maybe the model failed because of its *target* rather than
its method. Expected goals is the standard fix — and no free, licence-permitting
source covers Serie A back to 2007 (StatsBomb's open data has one modern Serie A
season; FBref blocks automated access; Understat's `robots.txt` is
`Disallow: /`). So this uses the pre-xG proxy the corpus already contains: the
same Dixon-Coles machinery fitted to **shots on target**, converted to goal rates
by one league-wide finishing factor.

| Pool (weights fitted on validation) | Market | Goals | Shots |
|---|---|---|---|
| Market + shots | 1.00 | — | **0.00** |
| Goals + shots | — | 0.65 | **0.35** |
| Market + goals + shots | 1.00 | **0.00** | **0.00** |

The middle row is the finding. **Shots on target carry information the goals
model lacks** — 35% weight against goals alone — so the original result was not
an artefact of a noisy target. Against the market that same signal earns exactly
zero, and in the three-way pool both models collapse to zero together.

Two model families on different targets, each holding information the other
lacks, and the closing price has already absorbed both. That is a stronger claim
than the goals-only result, and it is the version real xG would most likely have
confirmed rather than overturned.

## Match leverage: the Fiorentina case

Run as of **1 March 2026**, with Fiorentina 16th on 24 points and 12 fixtures
left. The model projected **41.4** expected final points against an actual **42**,
and a 96.4% survival probability. Fiorentina finished 15th.

Remaining fixtures ranked by leverage on survival:

| Fixture | Venue | P(survive \| win) | P(survive \| lose) | Leverage |
|---|---|---|---|---|
| Lecce | A | 0.987 | 0.914 | **0.073** |
| Cremonese | A | 0.986 | 0.922 | 0.063 |
| Verona | A | 0.982 | 0.927 | 0.055 |
| … | | | | |
| Juventus | A | 0.988 | 0.951 | 0.037 |
| Roma | A | 0.988 | 0.952 | 0.036 |
| Inter | H | 0.988 | 0.955 | **0.033** |

The glamour fixture is the *least* consequential one on the list. Away at Lecce
carries **2.3× the leverage** of hosting Inter, and the gap is roughly eight Monte
Carlo standard errors wide, so the ordering is not noise. Six-pointers against
fellow strugglers dominate; matches against the top six barely move survival
either way, because the model expects defeat in both branches.

That is a rotation and load-management argument expressed in probability, and it
is the thing a 1X2 forecast cannot tell you on its own.

## Method

- **Data.** football-data.co.uk, Serie A, 2007-08 to 2025-26 — 19 complete
  seasons, 7,220 matches. Results, shots, shots on target, corners, and prices
  from Bet365 (all seasons) and Pinnacle including closing lines (2012-13 on).
- **Market probabilities.** Shin (1992) margin removal rather than proportional
  normalisation, which leaves a favourite-longshot bias. Pinnacle closing
  preferred, falling back to Pinnacle open then Bet365 open.
- **Model.** Dixon-Coles bivariate Poisson with the low-score dependence
  correction and exponential time decay. Attack ratings constrained to sum to
  zero for identifiability. Unseen (newly promoted) clubs pool to the league
  mean.
- **Decay rate.** Selected on 2013-19 validation: **0.002/day**, a half-life of
  347 days — considerably slower than the 0.0065 in the original Dixon-Coles
  paper. Clean interior optimum.
- **Validation.** Rolling-origin walk-forward with a 14-day refit cadence. No
  k-fold on time-ordered data.
- **Uncertainty.** Bootstrap intervals on every mean; paired bootstrap for every
  model comparison, exploiting the fact that competing models score the same
  fixtures.
- **Protocol.** 2013-19 validation for all hyperparameters, pooling weight and
  calibration temperature. 2019-26 test touched once.

## What changed from the 2016 study

| 2016 study | This rebuild |
|---|---|
| Accuracy on 3 classes | RPS (ordinal), log loss, Brier, accuracy |
| Benchmark: 33% uniform | Benchmark: de-vigged market, plus base rate |
| Raw decimal odds as SVM features | Shin-de-vigged probabilities, pooled explicitly |
| 5-fold CV on time-ordered data | Rolling-origin walk-forward |
| Fixed `e^-i` weights, ÷ n | Tuned decay, normalised by weight sum |
| No confidence intervals | Bootstrap intervals throughout |
| Point predictions | Calibration diagnostics + recalibration |
| MATLAB Classification Learner | Reproducible package, 126 tests, independently audited |
| Predicts outcomes | Simulates seasons, ranks fixtures by leverage |

Two arithmetic notes on the original, recovered from its own confusion matrices
(both total 218 matches, against a stated test set of 320):

- Its Table 4.2 is captioned "Linear SVM" but its diagonal (114/218 = 52.3%)
  identifies it as AdaBoost.
- Its binary away-win classifiers all scored at or below the 72.5% obtained by
  never predicting an away win, and the Linear SVM's reported 57.8% home
  accuracy is exactly 126/218 — the score for never predicting a home win.

## Reproducing

```bash
conda create -n seriea python=3.12 -y && conda activate seriea
pip install -e ".[dev]"

python -c "from seriea.data.download import download_all; download_all()"
python scripts/tune_decay.py          # ~10 min: decay selection on validation
python scripts/run_backtest.py        # ~5 min: walk-forward vs the market
python scripts/fiorentina_leverage.py # season simulation and leverage table
python scripts/make_figures.py
python scripts/run_shots_experiment.py # shot-based signal vs goals vs market
python scripts/build_artifact.py      # self-contained HTML report
python scripts/paper_supplements.py    # season splits, pool profile, vector figures
python scripts/build_arxiv_bundle.py   # self-contained arXiv tarball
cd paper && latexmk -pdf seriea-leverage.tex           # short paper
cd paper && latexmk -pdf seriea-leverage-arxiv.tex     # full paper

pytest                                 # 126 tests
```

## Layout

```
src/seriea/
  config.py              constants, paths, season codes
  data/                  download, parse, validate, team helpers
  odds/devig.py          implied probabilities, Shin and multiplicative margin removal
  models/                base-rate and market baselines, Dixon-Coles, market-model pool
  evaluation/            RPS/log loss/Brier, calibration, bootstrap, rolling-origin backtest
  simulation/            Monte Carlo seasons, match leverage
paper/                   LaTeX source and PDFs (short + full), arXiv bundle
scripts/                 the runnable experiments and the report build
assets/fonts/            EB Garamond subsets (OFL-1.1) inlined into the report
reports/                 generated tables, figures, settings
```

## Limitations

- **No player-level information.** No injuries, suspensions, rest days or
  transfers. This is the single largest gap against what a club holds
  internally, and the most likely source of genuine edge over a market price.
- **No expected goals.** Goal-based strength is noisier than xG-based strength;
  Understat or StatsBomb xG would be the next addition.
- **Tiebreakers simplified.** Serie A resolves equal points on head-to-head
  first; the simulator uses points, then goal difference, then goals scored.
- **Leverage is single-objective.** It conditions on one position band at a
  time. A club trading survival against a cup run needs a multi-objective
  utility, not a probability swing.
- **The market benchmark is a closing price**, which incorporates late team-news
  the model never sees. Part of the 0.0068 RPS gap is information, not
  modelling.

## References

- Dixon, M. and Coles, S. (1997). Modelling association football scores and
  inefficiencies in the football betting market. *Applied Statistics* 46(2).
- Constantinou, A. and Fenton, N. (2012). Solving the problem of inadequate
  scoring rules for assessing probabilistic football forecast models. *Journal
  of Quantitative Analysis in Sports* 8(1).
- Shin, H. S. (1993). Measuring the incidence of insider trading in a market for
  state-contingent claims. *Economic Journal* 103(420).
- Štrumbelj, E. (2014). On determining probability forecasts from betting odds.
  *International Journal of Forecasting* 30(4).
- Rue, H. and Salvesen, Ø. (2000). Prediction and retrospective analysis of
  soccer matches in a league. *The Statistician* 49(3).
