# arXiv submission notes

Upload `seriea-leverage-arxiv.tar.gz` (built by `scripts/build_arxiv_bundle.py`).
It is flat and self-contained: one `.tex`, five vector figures, no `.bib` —
the bibliography is inline `thebibliography`, so no BibTeX pass is needed.

`\pdfoutput=1` is the first line, which tells arXiv's AutoTeX to use pdfLaTeX.

## Metadata for the submission form

**Title**

```
Does a Structural Model Add Anything to the Closing Price? Calibrated forecasting, incremental information, and match leverage in the Italian Serie A
```

**Authors**

```
Yannik Pitcan
```

**Primary category:** `stat.AP` (Applications)

**Cross-lists worth considering:** `stat.ME` (Methodology) for the pooling
diagnostic; `econ.EM` (Econometrics) for the market-efficiency angle. Neither is
required.

**MSC class:** 62P99, 62M20 · **ACM class:** G.3

**Comments field**

```
15 pages, 5 figures. Code and data pipeline: https://github.com/pitcany/seriea-leverage
```

**Abstract** (plain text — arXiv's form does not accept LaTeX markup beyond
inline math, so this is the de-formatted version)

```
Studies of association-football forecasting routinely report three-way accuracy in
the low fifties and present it as competitive with the betting market. We argue that
accuracy against a uniform benchmark answers the wrong question, and that the
question worth asking is whether a model carries information a margin-free closing
price has not already absorbed. We formalise that test as the fitted weight in a
logarithmic opinion pool and apply it to nineteen complete Serie A seasons (7,220
matches, 2007-08 to 2025-26).

The answer is negative and stable. A Dixon-Coles model with tuned exponential decay
attains 53.4% accuracy and a Ranked Probability Score of 0.1972 against the market's
0.1905; the paired difference is +0.0067 (95% CI [0.0046, 0.0088]) and the market
wins in all seven test seasons. The fitted pooling weight on the structural model is
0.000, and the log-loss profile is monotone increasing in that weight on the
validation period and on the test period alike, so the result is a genuine boundary
solution rather than an optimisation artefact. To test whether this reflects the
noisiness of goals rather than the method, we refit the identical machinery to shots
on target, a pre-expected-goals proxy for chance creation. That variant earns weight
0.35 against the goals model -- it demonstrably carries information the goals model
lacks -- and 0.000 against the market. Two structural signals, each informative
relative to the other, both already priced.

Separately, we find that the structural model is better calibrated than the market on
the home-win margin (calibration slope 0.995 versus 1.103) while being clearly less
sharp, so the market's advantage is discrimination rather than honesty; studies
reporting only accuracy cannot distinguish these.

The constructive consequence is that value lies not in a better outcome forecast but
in what is built on a calibrated one. We define match leverage, the change in a club's
probability of achieving a season objective between winning and losing a given
fixture, give conditions under which it can be estimated by post-hoc conditioning on a
single simulation, and compute it for ACF Fiorentina. The ordering inverts intuition:
an away fixture against a relegation rival carried 2.3x the leverage of hosting the
eventual champions.

The paper also documents and corrects specific errors in an earlier study of our own
on the same problem.
```

## Before you submit

- **Endorsement is not required.** You have prior arXiv submissions
  (`arXiv:2102.07115`, `arXiv:1712.06160`), so `stat.AP` posting rights should
  already be in place.
- **Licence.** arXiv's default (non-exclusive licence to distribute) is fine and
  is the least restrictive for a work you may later submit to a journal. CC BY
  is the alternative if you want unrestricted reuse.
- **The 2018 reference** is listed as *accepted for publication*, which is
  accurate — the paper was accepted by IJCA in 2018 but the camera-ready and
  article-processing charge were never completed, so it does not appear in the
  IJCA digital library. Do not upgrade this to "published" anywhere in the
  metadata.
- **Rebuild before uploading** if any result changes:
  ```
  python scripts/run_backtest.py
  python scripts/run_shots_experiment.py
  python scripts/paper_supplements.py
  python scripts/build_arxiv_bundle.py
  ```
