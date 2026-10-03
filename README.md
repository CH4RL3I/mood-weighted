# mood-weighted

News sentiment as Black-Litterman views, tested out of sample against honest baselines.

[![ci](https://github.com/CH4RL3I/mood-weighted/actions/workflows/ci.yml/badge.svg)](https://github.com/CH4RL3I/mood-weighted/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

The Python package is called `sentport`. The question is simple: if you score financial headlines with
FinBERT and feed the result into a portfolio optimiser as views, do you get a better portfolio than
equal weight or a shrinkage minimum-variance portfolio, after costs, with no lookahead?

**On this data, no.** The sentiment signal has no measurable rank correlation with next-period returns
(monthly IC -0.008, t = -0.29), and none of the sentiment portfolios beats equal weight net of costs.

![Equity curves](docs/equity_curves.png)

## Results

29 US large and mid caps, 62 monthly rebalances, out of sample from 2015-04-01 to 2020-06-30
(1,322 trading days, 45,545 headlines). Costs are 10 bp per unit of traded notional. The risk-free rate is
zero for every strategy. Confidence intervals are stationary bootstrap (5,000 resamples, mean block
21 days). The last column uses paired resamples, so it tests the Sharpe difference directly.

| Strategy | CAGR | Vol | Sharpe (net) | 95% CI | Sharpe (gross) | Max DD | One-way turnover/yr | Sharpe diff vs EW (95% CI) |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Equal weight | 10.5% | 21.3% | 0.58 | [-0.19, 1.52] | 0.58 | -42.2% | 40% | - |
| Min variance (Ledoit-Wolf) | 5.3% | 17.2% | 0.39 | [-0.31, 1.31] | 0.40 | -37.1% | 125% | -0.19 [-0.58, +0.20] |
| Black-Litterman + sentiment | 7.9% | 21.2% | 0.46 | [-0.29, 1.38] | 0.52 | -44.5% | 599% | -0.11 [-0.38, +0.15] |
| Black-Litterman + sentiment (IC = 0.20) | 6.2% | 21.8% | 0.39 | [-0.37, 1.27] | 0.45 | -44.0% | 712% | -0.19 [-0.65, +0.18] |
| Sentiment top-6 | 8.3% | 22.2% | 0.47 | [-0.33, 1.42] | 0.53 | -46.6% | 683% | -0.11 [-0.54, +0.30] |

Does sentiment predict returns? Rank information coefficient of the signal against the next-period
return, with a t-statistic (mean over standard error) and a bootstrap interval:

| Horizon | n | Mean IC | t-stat | Bootstrap 95% CI | Hit rate |
|---|---:|---:|---:|---|---:|
| Monthly | 61 | -0.008 | -0.29 | [-0.060, 0.045] | 44% |
| Weekly | 269 | 0.006 | 0.43 | [-0.017, 0.030] | 53% |

The top-minus-bottom quintile spread is -0.35% per month (t = -0.59) and +0.05% per week (t = 0.33).
Everything is statistically indistinguishable from zero. Sentiment portfolios lose Sharpe mainly through
turnover: gross of costs BL + sentiment is at 0.52 against 0.58 for equal weight, and the cost gap widens
quickly at higher cost assumptions (see `docs/results.md`). The wide Sharpe intervals also mean this
sample cannot rank any of the strategies with confidence, in either direction.

![IC over time](docs/ic_over_time.png)
![Sentiment vs next return](docs/sentiment_deciles.png)

Full tables, including a cost sensitivity (0, 10, 25, 50 bp), are in [docs/results.md](docs/results.md).

## Method

**Timestamps.** Decisions are taken at the US close (16:00 America/New_York) of the last trading day of
the month. A headline may be used only if it was published before that cutoff. Headlines with a real
timestamp are converted to New York time (DST-aware) and counted from that day if published before 16:00,
otherwise from the next day. The dataset used here has dates only, no time of day, so a headline dated
`D` may have come out after the close on `D`. Those headlines are treated as available from `D+1`
(conservative). Orders execute at the close of the next trading day. This logic lives in
`src/sentport/timing.py` and is tested, including by deliberately breaking it once to confirm the tests fail.

**Signal.** Each unique headline is scored by `ProsusAI/finbert` (score = P(positive) - P(negative);
headline text only, CPU, 4 threads, cached). Scores are aggregated to a per-stock daily count and sum.
At a decision date the signal is an exponentially decayed mean of available scores (half-life 15 calendar
days) plus the decayed headline count, then z-scored across stocks.

**Portfolios** (long-only, 20% weight cap, monthly, covariance from the last 252 daily returns):

1. Equal weight.
2. Minimum variance with a Ledoit-Wolf covariance (shrinkage towards a scaled identity).
3. Black-Litterman with an equal-weight prior. Equilibrium returns are `pi = delta * Sigma * w_eq`
   (delta = 2.5). Each stock gets one absolute view `q = pi + IC * sigma * z` (Grinold's rule, assumed
   IC 0.05). View confidence rises with the headline count, `c = n / (n + 5)`, mapped to view variance
   as `Omega = tau * Sigma_ii * (1 - c) / c` (Idzorek). Zero-count stocks get no view, and zero confidence
   returns the prior exactly. Weights maximise `w'mu - delta/2 w'Sigma w` under the constraints. A second
   variant uses IC = 0.20 as a robustness check on view strength.
4. Sentiment top-k: equal weight in the 6 stocks with the highest decayed sentiment (at least 2 decayed
   headlines), so each weight is 1/6, inside the cap.

No historical market caps are available for free, so the prior is equal weight rather than cap weight.

**Costs.** 10 bp on every unit of traded notional (buys and sells, `sum |w_target - w_drifted|`),
charged on the execution day. Turnover in the table is one-way (`0.5 * sum |dw|`), annualised.

**Inference.** IC is the cross-sectional Spearman correlation per date, with a t-statistic on the mean
across dates and a stationary bootstrap interval. Sharpe intervals use a stationary bootstrap of daily net
returns.

## Data

* **News:** [`ashraq/financial-news`](https://huggingface.co/datasets/ashraq/financial-news) on Hugging
  Face, 1.85 million headlines for about 6,500 tickers, 2010 to June 2020. It is derived from the
  Benzinga partner-headline feed as published on Kaggle ("Massive Stock News Analysis DB for NLP
  Backtests"). The Hugging Face card states no licence and the upstream terms are unclear, so
  **raw headlines are not redistributed here**: they are downloaded into the git-ignored `data/` directory
  by `sentport fetch`. Only the aggregate `(ticker, date, headline count, summed score)` table is committed.
  I looked at FNSPID (CC BY-NC 4.0, but roughly 5.7 GB and 23 GB CSVs) and a gated multi-source dataset;
  neither was practical without keys or large downloads.
* **Universe:** 29 stocks. The dataset covers many tickers thinly, and most mega caps have almost no
  coverage before 2020. I kept tickers with at least 3 headlines in at least 97% of months from 2015-01 to
  2020-05, dropped ETFs and ADRs, and kept those still priced on Yahoo in 2026. Selection never used returns.
  Publishers are mostly Seeking Alpha (45%), Zacks (23%), GuruFocus (14%) and Investor's Business Daily (13%).
  There are about 24 headlines per stock per month (median 21).
* **Prices:** daily adjusted closes from Yahoo Finance through `yfinance` (Stooq now sits behind a
  JavaScript check and cannot be fetched from a script). Prices are not redistributed here, since
  Yahoo's terms are for personal use; they are downloaded and cached in `data/` on first use.
* **Sample:** `sample/` holds the per-stock-per-day sentiment aggregates (no headline text).
  `sentport backtest --sample` and `sentport report --sample` reproduce all figures from it, fetching
  only the prices, so the FinBERT scoring never has to be rerun.

## Usage

```
uv sync
uv run sentport fetch                 # headlines + prices into data/
uv run sentport score                 # FinBERT, cached; about 40 minutes on a busy CPU
uv run sentport backtest              # add --sample to use the committed sample
uv run sentport report --sample       # figures in docs/ and docs/results.md
uv run pytest && uv run ruff check src tests
```

## Limitations

* **Coverage bias.** Only large and mid caps with dense headline coverage are included. Heavily covered
  names are the ones where news is least likely to be mispriced, so this is a hard test for sentiment.
* **Survivorship.** The universe is conditioned on stocks that still had Yahoo price history in 2026.
  Three otherwise eligible tickers (PXD, VMW, JWN) were dropped for that reason, which shows the filter
  is real. Returns are biased upwards for all strategies, roughly equally.
* **Source bias.** Most headlines come from Seeking Alpha, Zacks and GuruFocus: aggregator and
  analyst-blog content, often repeating earnings or ratings news that is already public. Date-only
  stamps mean the first day of any reaction is discarded.
* **FinBERT domain shift.** FinBERT was fine-tuned on Financial PhraseBank sentences and news; many
  headlines here are titles of blog posts or promotional pieces. Headline-only scoring loses context.
* **Short sample.** About five years, 62 monthly decisions, one market crash (March 2020). Sharpe
  intervals are roughly one unit wide, so small differences cannot be detected either way.
* **Choices not tuned.** Half-life, IC assumption, confidence mapping, cap and top-k were fixed before
  looking at results, except that IC = 0.20 was added as a robustness variant. Nothing was optimised.
* Risk-free rate zero, equal-weight prior, close-to-close execution with no market impact.

## References

* Black, F. and Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal* 48(5).
* He, G. and Litterman, R. (1999). The Intuition Behind Black-Litterman Model Portfolios. Goldman Sachs
  Investment Management Research.
* Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.
* Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance
  matrices. *Journal of Multivariate Analysis* 88(2).
* Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model.
* Politis, D. and Romano, J. (1994). The Stationary Bootstrap. *Journal of the American Statistical
  Association* 89(428).
* Grinold, R. and Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.

## Licence

MIT, Emilio Gappa, 2026. Model weights and raw data are not included.
