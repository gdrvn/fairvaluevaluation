# FairValue Scout 2.0 — six-model stock screening agent

**What it does:** Screens U.S.-listed equities with market capitalization above USD 10 billion, calculates a weighted fair-value estimate from up to **six appropriate valuation models**, filters for **15–60% implied upside**, and lists **up to 50** ranked stocks in an industry-filterable HTML report and CSV, with published 12-month analyst consensus targets when available. The standard GitHub Actions setup runs daily, and optionally emails the **top five every Saturday**.

This is a runnable **independent research prototype**, inspired by [ACCE Investments' publicly published method](https://www.acceinvestments.com/data/how-we-value-stocks), **not a copy of its source code, database, proprietary calibration or ratings**. Its price estimates may differ substantially. Fair-value gaps are not forecasts and do not justify trades by themselves.

## Quick start (local)

Python 3.11+ recommended.

```bash
unzip fair_value_agent.zip
cd fair_value_agent
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python agent.py --limit 50
```

Open `reports/report.html` in your browser. Also see `reports/candidates.csv`, `reports/run_summary.json`, and `reports/weekly_email.txt`.

The first run downloads financial information for the companies. It needs internet access. `yfinance` access may sometimes fail or be rate-limited; **no live financial data is bundled** and an empty screen is not a buy/sell signal.

**Full-universe research scan** (can be slow, especially the first time):

```bash
python agent.py --limit 0
```

`--limit 0` requests all entries that Yahoo's custom screening endpoint returns for U.S. exchanges and market cap above $10B; there may be omissions and duplicates. The number screened is reported. If you care about coverage, independently reconcile the universe against a licensed exchange feed.

**Scan your own selection** (one ticker per line, first column in CSV):

```bash
python agent.py --tickers sample_tickers.csv
```

For speed, use `--limit 500`. This is **not** a market-wide scan because it covers only the first 500 screener results, generally larger companies. The run report will tell you how many companies were screened. `--cache-hours 24` keeps a local copy of financial data for a day. The scan re-fetches a company's fundamentals only after its cache expires; it's not truly intraday live repricing. To force a fresh run, remove `.cache/`.

## Valuation models and safeguards

| Model | Formula / policy | Suppressed when |
|---|---|---|
| **DCF** | Equity free-cash-flow, 5 projected years, 10.5% annual discount, terminal growth 2.5% | Negative/missing FCF, lenders, property |
| **Forward P/E** | Sector peer median forward P/E × forward EPS | EPS unavailable, fewer than three valid peers |
| **EV/EBITDA** | (Sector EV/EBITDA × company EBITDA − net debt) / shares | Lenders, incomplete net debt or peers |
| **Graham number** | √(22.5 × trailing EPS × BVPS) | No positive EPS/book, asset-light/pharma/lenders/property |
| **Residual income** | BVPS × [1 + (ROE−10.5%)/(10.5%−2.5%)] | Non-lenders, missing data |
| **Shareholder yield** | Next-year dividends + net buybacks / (10.5%−2.5%) | Missing verified share-count data or yield under 2.5% |
| **Reverse DCF** | Invert DCF to find implied five-year growth | Information only; it **never** votes in fair value |

Rules applied to each company:

1. Equity listed on NYSE/Nasdaq/NYSE American, screen market cap ≥$10B, quoted and financially reported in USD. (This conservative prototype **omits foreign-reporting-currency ADRs** rather than mishandle FX.)
2. Never fill missing financial data or sector multiples with guesses. Sector median needs at least three valid covered peers.
3. Drop an individual model valuation if non-positive or above 3× share price; drop if over 3× or under 0.4× the median of the remaining models.
4. Need at least **two** surviving model estimates; **four or more** earns stronger evidential coverage. The published ACCE methodology says at least two surviving models even though it attempts six.
5. Compute a confidence-weighted average using documented weights in `MODEL_WEIGHTS`; report the surviving number, their individual estimates, dropped reasons, and model dispersion.
6. Keep stocks with fair value 15%–60% **above** price, ranked by upside. Exclude your existing open picks by putting tickers in `exclude.txt` if desired.

### Important differences from ACCE

- The DCF uses **reported free cash flow** (an equity-oriented measure), a smoothed **reported revenue-growth proxy** and a 2.5% terminal rate. It does **not fully reproduce ACCE's reinvestment = growth / ROIC identity**, growth persistence model or cost-of-equity calibration.
- Valuation weights are **our explicit prototype assumptions**, not ACCE's undisclosed weights.
- Peer multiples come from the actual subset fetched on that run. `--limit 50` gives **incomplete sector medians**, so use broader scans or a licensed complete financial feed before making decisions.
- The reverse DCF estimate is exploratory and can hit a 5-year growth-search boundary. There is no claim that this matches the original engine.
- Data is not point-in-time adjusted and may contain lookahead bias if used for historical backtests. It is not suitable for trading automation.
- Many shares do not have reliable history through the public source; the shareholder-yield model is excluded rather than inventing repurchases.
- The Yahoo screener can skip some stocks or return incomplete data; reports show actual coverage.
- **No LLM hallucinates prices or financial numbers in the code**. The conclusions are deterministic given the data fetched.

## Daily automatic runs via GitHub Actions

1. Create a **private GitHub repository** and upload all files (including `.github/workflows/daily-screener.yml`). Do not upload passwords or `.env` files.
2. Open your GitHub repository → **Actions** → `FairValue Scout daily valuation screen` → **Run workflow** for the first trial.
3. The workflow automatically runs every day at **08:30 UTC**, screens up to **500** companies by default for resource safety, and uploads `report.html`, `candidates.csv`, etc. as a GitHub Actions artifact. Scheduled execution can be delayed or blocked by rate limiting.
4. Set `--limit 0` / change `RUN_LIMIT` to `0` in the workflow if you need the broad scan, accepting longer runtime and likely data-provider rate limits. For reliable high-volume work, replace Yahoo with a licensed market-data API.
5. Optional Saturday email: in **Settings → Secrets and variables → Actions**, set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `REPORT_TO`, and `REPORT_FROM`. The script uses SMTP over SSL (commonly 465), **not your normal account password**. If secrets are absent, it writes the report instead of emailing it. The email contains the top-five plain-text list plus HTML table. Test with `python agent.py --limit 50 --email` after configuring secrets locally through environment variables.

The project **does not connect to Interactive Brokers or place orders**. It creates research artifacts only. It does not run in the background in ChatGPT until you deploy the GitHub Actions workflow.

## Files

- `agent.py` — screening, six-model valuation, quality checks, reports, email
- `.github/workflows/daily-screener.yml` — GitHub schedule
- `exclude.txt` — positions you want excluded from the public-screen list
- `sample_tickers.csv` — optional sample universe for faster testing
- `tests/` — offline unit tests with entirely fictional stock inputs
- `reports/` — generated HTML/CSV, no financial data included in the starter package

## Licensing and safety

The [yfinance project](https://github.com/ranaroussi/yfinance) says the Yahoo Finance data access is intended **for personal use only**. For public or commercial newsletters, paid stock research products, or a production business, use a provider whose market-data license expressly permits that use. Verify all figures with filings. This agent is for educational research, not personalized investment advice; securities analysis or promotion may be regulated in your jurisdiction. Do not rely on a valuation gap alone as a buy signal.

## Preview without market-data access

Open `example/demo_report.html` to preview the output. It uses **entirely fictional companies and synthetic numbers** to illustrate the layout; it is not an actual investment shortlist.

## Version 2.0 improvements

- **Up to 50** ranked qualifying stocks in the report and CSV, screened from up to **500** companies in the daily workflow; if fewer than 50 qualify, fewer are shown. The broader option is `--limit 0`, subject to API coverage and rate limits. A sample of 500 is not the whole market.
- **Sector and industry** columns for sorting/grouping the CSV, plus an industry filter on the HTML dashboard.
- **12-month analyst consensus price target**, analyst count, analyst-implied upside and a source label are reported separately from the agent's model fair value. These use Yahoo Finance's aggregated `targetMeanPrice` and `numberOfAnalystOpinions` via yfinance, which are generally described as 12-month targets. Individual analyst dates cannot be verified in this feed. If missing, the dashboard prints N/A rather than fabricating a target.
- **Price/split validation:** inspect recent raw close and corporate-action history; reject stale (>7 days), inconsistent (>15% close/quote difference) and recently split (last 30 days) securities until the per-share data are reconciled. When history is missing, a stock can appear only in the review tier, never high-confidence.
- **High-confidence screening tier** requires >=4 surviving valuation models, <=50% model dispersion, a verified recent price, and enough debt/liquidity data without flagged risks. All other qualifying stocks are labelled Review only. The term describes agreement among these models, not certainty of returns.
- **Model disagreement** above 50% is explicitly flagged. Basic financial-health checks flag net debt/EBITDA >3.5x, debt/equity >250%, current ratio <1, and interest coverage <2; missing metrics remain Unknown. Financial companies are labelled Not comparable and require sector-specific review.
- **Tests run on every workflow execution** before the market-data scan; the daily schedule, manual trigger, report artifact, and optional Saturday SMTP email remain unchanged.

The valuation model is a heuristic, not trading advice. See `reports/run_summary.json` for the actual number screened, matched and displayed. The generated HTML and CSV intentionally contain at most 50 candidates.
