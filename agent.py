#!/usr/bin/env python3
"""FairValue Scout: educational, reproducible US-listed equity screen.

Data: Yahoo Finance via yfinance (personal research use). This is an independent,
simplified implementation of the publicly documented ACCE Investments methodology.
No LLM-generated prices or hidden price targets. See README for limitations.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import html
import json
import logging
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

LOG = logging.getLogger("fairvalue")
MIN_CAP = 10_000_000_000
RATE = 0.105
TERMINAL_G = 0.025
MODEL_WEIGHTS = {"DCF": 1.25, "Forward P/E": 1.0, "EV/EBITDA": 1.0,
                 "Graham number": 0.45, "Residual income": 1.25,
                 "Shareholder yield": 0.8}


def number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError, OverflowError):
        return None


def pos(v: Any) -> float | None:
    x = number(v)
    return x if x is not None and x > 0 else None


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


@dataclass
class Stock:
    symbol: str
    name: str
    price: float
    cap: float
    currency: str = "USD"
    financial_currency: str = "USD"
    sector: str = ""
    industry: str = ""
    shares: float | None = None
    fcf: float | None = None
    revenue_growth: float | None = None
    roic: float | None = None
    fwd_eps: float | None = None
    trailing_eps: float | None = None
    forward_pe: float | None = None
    ebitda: float | None = None
    ev_ebitda: float | None = None
    debt: float | None = None
    cash: float | None = None
    bvps: float | None = None
    roe: float | None = None
    dividend_yield: float | None = None
    buyback_yield: float | None = None
    fetched_at: str = ""


@dataclass
class Evaluation:
    symbol: str
    name: str
    price: float
    fair_value: float
    upside_pct: float
    cap: float
    sector: str
    industry: str
    currency: str
    valid_models: int
    model_values: dict[str, float]
    excluded_models: dict[str, str]
    reverse_dcf: str
    dispersion_pct: float
    fetched_at: str


def is_lender(s: Stock) -> bool:
    i = s.industry.lower()
    return any(k in i for k in ("bank", "insurance", "credit service", "mortgage",
                                "savings", "financial services", "capital market")) and \
           not any(k in i for k in ("financial data", "exchange", "fintech", "software"))


def is_property(s: Stock) -> bool:
    x = (s.industry + " " + s.sector).lower()
    return ("reit" in x or "real estate" in x)


def dcf_per_share(s: Stock, initial_growth: float | None = None) -> float | None:
    """Simplified five-year equity-FCF DCF. FCF is after interest; don't subtract debt again."""
    if is_lender(s) or is_property(s):
        return None
    if not (pos(s.fcf) and pos(s.shares)):
        return None
    if initial_growth is None:
        if s.revenue_growth is None:
            return None
        initial_growth = clamp(s.revenue_growth, -0.08, 0.18)
        if pos(s.roic):
            initial_growth = min(initial_growth, s.roic)
    fcf_ps = s.fcf / s.shares
    pv = 0.0
    for t in range(1, 6):
        growth = initial_growth + (TERMINAL_G - initial_growth) * t / 6
        fcf_ps *= 1 + growth
        pv += fcf_ps / (1 + RATE) ** t
    terminal = fcf_ps * (1 + TERMINAL_G) / (RATE - TERMINAL_G)
    result = pv + terminal / (1 + RATE) ** 5
    return pos(result)


def reverse_dcf(s: Stock) -> str:
    if not pos(s.fcf) or not pos(s.shares) or is_lender(s) or is_property(s):
        return "N/A"
    low, high = -0.15, 0.60
    lo = dcf_per_share(s, low)
    hi = dcf_per_share(s, high)
    if lo is None or hi is None:
        return "N/A"
    if s.price < lo:
        return "below -15% five-year growth"
    if s.price > hi:
        return "above 60% five-year growth"
    for _ in range(50):
        mid = (low + high) / 2
        v = dcf_per_share(s, mid)
        if v is None:
            return "N/A"
        if v < s.price:
            low = mid
        else:
            high = mid
    return f"{(low+high)/2:.1%} implied 5-year FCF growth"


def get_models(s: Stock, pe_medians: dict[str, float], ev_medians: dict[str, float]) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    exclusions: dict[str, str] = {}

    if is_lender(s) or is_property(s):
        exclusions["DCF"] = "Not suitable for lenders/property"
    else:
        v = dcf_per_share(s)
        if v:
            values["DCF"] = v
        else:
            exclusions["DCF"] = "Missing/negative FCF, share count or growth"

    if pos(s.fwd_eps) and pos(pe_medians.get(s.sector)):
        values["Forward P/E"] = s.fwd_eps * pe_medians[s.sector]
    else:
        exclusions["Forward P/E"] = "Missing forward EPS or >=3 valid sector peers"

    if is_lender(s):
        exclusions["EV/EBITDA"] = "Enterprise multiples not suitable for lenders"
    elif pos(s.ebitda) and pos(s.shares) and pos(ev_medians.get(s.sector)) and s.debt is not None and s.cash is not None:
        ev = s.ebitda * ev_medians[s.sector]
        v = (ev - (s.debt - s.cash)) / s.shares
        if pos(v):
            values["EV/EBITDA"] = v
        else:
            exclusions["EV/EBITDA"] = "Enterprise value below net debt"
    else:
        exclusions["EV/EBITDA"] = "Missing EBITDA, net debt or >=3 valid sector peers"

    if (s.sector in {"Technology", "Communication Services", "Healthcare", "Health Care"}
            or is_property(s) or is_lender(s)):
        exclusions["Graham number"] = "Not used for asset-light, pharma, lenders or property"
    elif pos(s.trailing_eps) and pos(s.bvps):
        values["Graham number"] = math.sqrt(22.5 * s.trailing_eps * s.bvps)
    else:
        exclusions["Graham number"] = "Requires positive EPS and book value"

    if not is_lender(s):
        exclusions["Residual income"] = "Used only for banks/lenders/insurers"
    elif pos(s.bvps) and s.roe is not None:
        v = s.bvps * (1 + (s.roe - RATE) / (RATE - TERMINAL_G))
        if pos(v):
            values["Residual income"] = v
        else:
            exclusions["Residual income"] = "Non-positive implied equity value"
    else:
        exclusions["Residual income"] = "Missing positive book value or ROE"

    # Share buybacks have to be verified from historical share count: never invent them.
    if s.dividend_yield is None or s.buyback_yield is None:
        exclusions["Shareholder yield"] = "Missing dividend or verified share-count history"
    else:
        y = s.dividend_yield + s.buyback_yield
        if y >= 0.025 and y <= 0.25:
            values["Shareholder yield"] = s.price * y * (1 + TERMINAL_G) / (RATE - TERMINAL_G)
        else:
            exclusions["Shareholder yield"] = "Net shareholder yield below 2.5% or invalid"
    return values, exclusions


def filter_models(raw: dict[str, float], price: float) -> tuple[dict[str, float], dict[str, str]]:
    dropped: dict[str, str] = {}
    # ACCE's published ±200% rule is interpreted as 0-3x the price for positive values.
    stage1 = {}
    for model, v in raw.items():
        if not (pos(v) and v <= 3 * price):
            dropped[model] = "Outside 0–3x market-price sanity bounds"
        else:
            stage1[model] = v
    if not stage1:
        return {}, dropped
    median = statistics.median(stage1.values())
    stage2 = {}
    for model, v in stage1.items():
        if v > 3 * median:
            dropped[model] = "Over 3x median model estimate"
        elif v < 0.4 * median:
            dropped[model] = "Below 0.4x median model estimate"
        else:
            stage2[model] = v
    return stage2, dropped


def evaluate(s: Stock, pe_medians: dict[str, float], ev_medians: dict[str, float]) -> Evaluation | None:
    if not pos(s.price) or s.cap < MIN_CAP:
        return None
    # All per-share data must be measured in one currency; never silently use bad FX.
    if s.currency != s.financial_currency or s.currency != "USD":
        return None
    models, excluded = get_models(s, pe_medians, ev_medians)
    kept, removed = filter_models(models, s.price)
    excluded.update(removed)
    if len(kept) < 2:
        return None
    fair = sum(v * MODEL_WEIGHTS[k] for k, v in kept.items()) / sum(MODEL_WEIGHTS[k] for k in kept)
    if not pos(fair):
        return None
    upside = (fair / s.price - 1) * 100
    spread = (max(kept.values()) - min(kept.values())) / fair * 100 if len(kept) > 1 else 0
    return Evaluation(s.symbol, s.name, round(s.price, 4), round(fair, 4), round(upside, 2),
                      s.cap, s.sector, s.industry, s.currency, len(kept),
                      {k: round(v, 4) for k, v in kept.items()}, excluded,
                      reverse_dcf(s), round(spread, 1), s.fetched_at)


def sector_medians(stocks: list[Stock]) -> tuple[dict[str, float], dict[str, float]]:
    """Use actual covered peers. No invented sector-median fallback."""
    pe, ev = {}, {}
    for s in stocks:
        if not s.sector:
            continue
        if pos(s.forward_pe) and s.forward_pe < 100:
            pe.setdefault(s.sector, []).append(s.forward_pe)
        if not is_lender(s) and pos(s.ev_ebitda) and s.ev_ebitda < 100:
            ev.setdefault(s.sector, []).append(s.ev_ebitda)
    return ({k: statistics.median(v) for k, v in pe.items() if len(v) >= 3},
            {k: statistics.median(v) for k, v in ev.items() if len(v) >= 3})


def get_universe(max_companies: int, tickers_file: str | None) -> list[dict]:
    if tickers_file:
        with open(tickers_file, newline="", encoding="utf-8") as f:
            symbols = [r[0].strip().upper() for r in csv.reader(f) if r and r[0].strip() and not r[0].startswith("#")]
        return [{"symbol": s} for s in (symbols[:max_companies] if max_companies else symbols)]
    import yfinance as yf
    q = yf.EquityQuery("and", [
        yf.EquityQuery("gte", ["intradaymarketcap", MIN_CAP]),
        yf.EquityQuery("is-in", ["exchange", "NYQ", "NMS", "ASE"]),
    ])
    result = []
    step = 250
    for offset in range(0, 5000, step):
        attempt = yf.screen(q, offset=offset, size=step, sortField="intradaymarketcap", sortAsc=False)
        quotes = attempt.get("quotes", [])
        if not quotes and isinstance(attempt.get("finance"), dict):
            quotes = attempt["finance"].get("result", [{}])[0].get("quotes", [])
        if not quotes:
            break
        result.extend(quotes)
        LOG.info("Found %s US-listed equities above $10B", len(result))
        if max_companies and len(result) >= max_companies:
            return result[:max_companies]
        if len(quotes) < step:
            break
    return result


def get_buyback_yield(ticker: Any) -> float | None:
    """Compare recent share count to a point ~1yr prior; omitted if unverifiable."""
    try:
        now = datetime.now(timezone.utc)
        series = ticker.get_shares_full(start=(now - timedelta(days=540)).date().isoformat())
        if series is None or getattr(series, "empty", True):
            return None
        series = series.dropna().sort_index()
        if len(series) < 2:
            return None
        latest_date = series.index[-1].to_pydatetime()
        latest = float(series.iloc[-1])
        past = [(d, float(v)) for d, v in series.items()
                if 310 <= (latest_date - d.to_pydatetime()).days <= 440]
        if not past:
            return None
        prior = min(past, key=lambda x: abs((latest_date - x[0]).days - 365))[1]
        if prior <= 0 or latest <= 0:
            return None
        return clamp((prior - latest) / prior, -0.2, 0.2)
    except Exception as e:
        LOG.debug("No share count history: %s", e)
        return None


def fetch_stock(quote: dict) -> Stock | None:
    import yfinance as yf
    symbol = quote.get("symbol")
    if not symbol:
        return None
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        price = pos(info.get("currentPrice")) or pos(info.get("regularMarketPrice")) or pos(quote.get("regularMarketPrice"))
        cap = pos(info.get("marketCap")) or pos(quote.get("marketCap"))
        if not price or not cap or cap < MIN_CAP:
            return None
        ccy = (info.get("currency") or quote.get("currency") or "USD").upper()
        finccy = (info.get("financialCurrency") or ccy).upper()
        if ccy != "USD" or finccy != ccy:
            LOG.info("Skipping %s mismatched or non-USD currency %s/%s", symbol, ccy, finccy)
            return None
        shares = pos(info.get("sharesOutstanding")) or cap / price
        dy = number(info.get("dividendYield"))
        if dy is not None and (dy < 0 or dy > 1):
            dy = None
        # yfinance ROE/revenue growth usually returned as fractions (e.g. .12 for 12%).
        return Stock(
            symbol=symbol, name=info.get("shortName") or quote.get("shortName") or symbol,
            price=price, cap=cap, currency=ccy, financial_currency=finccy,
            sector=info.get("sector", ""), industry=info.get("industry", ""),
            shares=shares, fcf=number(info.get("freeCashflow")),
            revenue_growth=number(info.get("revenueGrowth")), roic=number(info.get("returnOnCapital")),
            fwd_eps=number(info.get("forwardEps")), trailing_eps=number(info.get("trailingEps")),
            forward_pe=number(info.get("forwardPE")), ebitda=number(info.get("ebitda")),
            ev_ebitda=number(info.get("enterpriseToEbitda")),
            debt=number(info.get("totalDebt")), cash=number(info.get("totalCash")),
            bvps=number(info.get("bookValue")), roe=number(info.get("returnOnEquity")),
            dividend_yield=dy, buyback_yield=get_buyback_yield(ticker),
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    except Exception as e:
        LOG.warning("Could not fetch %s: %s", symbol, e)
        return None


def load_cache(path: Path, ttl_hours: int) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc)
        valid = {}
        for sym, item in data.items():
            stamp = datetime.fromisoformat(item["fetched_at"])
            if now - stamp <= timedelta(hours=ttl_hours):
                valid[sym] = item
        return valid
    except Exception as e:
        LOG.warning("Ignoring cache: %s", e)
        return {}


def gather(universe: list[dict], cache_file: Path, ttl_hours: int, delay_seconds: float) -> tuple[list[Stock], int]:
    cached = load_cache(cache_file, ttl_hours)
    merged = dict(cached)
    stocks = []
    fetched = 0
    seen = set()
    for i, quote in enumerate(universe, 1):
        symbol = quote.get("symbol", "").upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        if symbol in cached:
            stock = Stock(**cached[symbol])
        else:
            stock = fetch_stock(quote)
            fetched += 1
            if stock:
                merged[symbol] = asdict(stock)
            if delay_seconds:
                time.sleep(delay_seconds)
        if stock:
            stocks.append(stock)
        if i % 25 == 0:
            LOG.info("Processed %d/%d; %d usable; %d fetched", i, len(universe), len(stocks), fetched)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return stocks, fetched


def money(v: float) -> str:
    return f"${v:,.2f}"


def build_report(evaluations: list[Evaluation], inspected: int, fetched: int, all_evals: list[Evaluation]) -> tuple[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    main = evaluations[:5]
    heading = "Top 5 US-listed stocks with estimated 15–60% value gap"
    rows = []
    for index, e in enumerate(evaluations, 1):
        details = "".join(f"<li><strong>{html.escape(k)}</strong>: {money(v)}</li>" for k, v in e.model_values.items())
        dropped = "".join(f"<li>{html.escape(k)} — {html.escape(v)}</li>" for k, v in e.excluded_models.items())
        rows.append(f"""<tr><td>{index}</td><td><strong>{html.escape(e.symbol)}</strong><br>{html.escape(e.name)}</td>
<td>{money(e.price)}</td><td>{money(e.fair_value)}</td><td class='positive'>{e.upside_pct:.1f}%</td>
<td>{e.valid_models}/6</td><td>{e.dispersion_pct:.0f}%</td>
<td><details><summary>Models & assumptions</summary><ul>{details}</ul><p>Reverse DCF: {html.escape(e.reverse_dcf)}</p>
<p>Excluded:</p><ul>{dropped}</ul></details></td></tr>""")
    body = "\n".join(rows) if rows else "<tr><td colspan='8'>No companies passed the screen with at least two valid models.</td></tr>"
    html_doc = f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>FairValue Scout | {timestamp}</title><style>
:root{{font-family:Inter,system-ui,-apple-system,sans-serif;color:#17263f;background:#f3f6fc}}
body{{max-width:1250px;margin:35px auto;padding:0 18px}}h1{{font-size:31px;margin:10px 0}}p{{line-height:1.5}}
header{{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px}}.tag{{background:#e8f0ff;color:#2457d3;border-radius:20px;padding:7px 12px}}
.card{{background:white;border:1px solid #dbe4f4;border-radius:18px;padding:24px;margin:20px 0;box-shadow:0 3px 18px #1b3b7508}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:13px 9px;border-bottom:1px solid #e6edf7;text-align:left;vertical-align:top}}
th{{color:#62718b;font-size:12px;text-transform:uppercase}}tr:nth-child(even){{background:#f9fbff}}.positive{{color:#137a57;font-weight:750}}
details{{min-width:150px}}.note{{font-size:13px;color:#607086}}.scroll{{overflow-x:auto}}li{{margin:5px 0}}
</style><body><header><div><span class='tag'>Independent research prototype</span><h1>{heading}</h1>
<p>{timestamp} • Screened {inspected} listings • {fetched} new fundamentals requests • {len(all_evals)} evaluable • {len(evaluations)} qualified</p></div></header>
<div class='card'><h2>Ranked by model-implied upside</h2><div class='scroll'><table><thead><tr>
<th>#</th><th>Company</th><th>Price</th><th>Fair value</th><th>Implied upside</th><th>Valid models</th><th>Model range / fair value</th><th>Evidence</th>
</tr></thead><tbody>{body}</tbody></table></div></div>
<div class='card'><h2>Methodology</h2><p>US-listed equities, market cap above $10 billion, USD price and financial reporting, estimated fair value 15–60% above market price, at least two surviving valuation models. Six models are attempted when suitable: simplified 5-year equity free-cash-flow DCF (10.5% discount and 2.5% terminal growth), forward P/E peer median, EV/EBITDA peer median, Graham number, residual income for lenders, and shareholder yield. Reverse DCF is informational only. Outlier rules: remove values outside 0–3× price and below 0.4× or above 3× other-model median; weighted average of survivors. Sector peers are built from the fetched sample; no fabricated peer estimates.</p>
<p class='note'>Fair value is not a time-specific price target. Yahoo data may be delayed, incomplete or erroneous. Financial figures are cached for up to the configured hours, so this is not a genuine tick-by-tick repricing service. Growth and discount-rate assumptions are simplified; this does not reproduce ACCE Investments' implementation. Large disagreement between models signals low confidence. Run full-universe scanning with an appropriately licensed data feed for commercial usage. No investment advice.</p>
<p class='note'>Sources: <a href='https://www.acceinvestments.com/data/how-we-value-stocks'>ACCE published methodology</a>; <a href='https://ranaroussi.github.io/yfinance/'>yfinance data client (personal research terms)</a>.</p></div>
</body></html>"""
    plain = [f"{heading} | {timestamp}", f"Scanned: {inspected}; candidates: {len(evaluations)}", ""]
    for i, e in enumerate(main, 1):
        plain.append(f"{i}. {e.symbol} — {e.name}: price {money(e.price)}; fair value {money(e.fair_value)}; upside {e.upside_pct:.1f}%; {e.valid_models} models; dispersion {e.dispersion_pct:.0f}%")
    plain.append("\nEstimated valuation gaps, not price targets. Research only. Sources and model details in HTML report.")
    return html_doc, "\n".join(plain)


def save_csv(results: list[Evaluation], path: Path) -> None:
    fieldnames = ["symbol", "name", "price", "fair_value", "upside_pct", "cap", "sector", "industry",
                  "currency", "valid_models", "dispersion_pct", "reverse_dcf", "fetched_at", "model_values", "excluded_models"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in results:
            item = asdict(e)
            item["model_values"] = json.dumps(item["model_values"])
            item["excluded_models"] = json.dumps(item["excluded_models"])
            writer.writerow(item)


def send_email(subject: str, text: str, html_content: str) -> None:
    from email.message import EmailMessage
    import smtplib
    host = os.environ.get("SMTP_HOST", "")
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASS", "")
    to = os.environ.get("REPORT_TO", "")
    sender = os.environ.get("REPORT_FROM", user)
    port = int(os.environ.get("SMTP_PORT") or "465")
    if not all((host, user, password, to, sender)):
        raise ValueError("SMTP_HOST, SMTP_USER, SMTP_PASS, REPORT_TO and REPORT_FROM (or SMTP_USER) are required")
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    msg.add_alternative(html_content, subtype="html")
    with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)


def run(args: argparse.Namespace) -> None:
    quotes = get_universe(args.limit, args.tickers)
    if not quotes:
        raise RuntimeError("No listings returned. Check data-source access or supply --tickers symbols.csv")
    excluded_tickers = set()
    if args.exclude and Path(args.exclude).exists():
        excluded_tickers = {s.strip().upper() for s in Path(args.exclude).read_text().splitlines() if s.strip() and not s.startswith("#")}
    quotes = [q for q in quotes if q.get("symbol", "").upper() not in excluded_tickers]
    stocks, fetched = gather(quotes, Path(args.cache), args.cache_hours, args.delay)
    pe, ev = sector_medians(stocks)
    LOG.info("Sector P/E medians: %s; EV/EBITDA medians: %s", pe, ev)
    all_evaluations = [e for s in stocks if (e := evaluate(s, pe, ev))]
    results = [e for e in all_evaluations if 15 <= e.upside_pct <= 60]
    results.sort(key=lambda e: (-e.upside_pct, e.dispersion_pct))
    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)
    page, mail_text = build_report(results, len(quotes), fetched, all_evaluations)
    (out / "report.html").write_text(page, encoding="utf-8")
    (out / "weekly_email.txt").write_text(mail_text, encoding="utf-8")
    (out / "run_summary.json").write_text(json.dumps({
        "inspected": len(quotes), "available_financials": len(stocks), "evaluated": len(all_evaluations),
        "matched": len(results), "top5": [asdict(e) for e in results[:5]],
        "generated_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    save_csv(results, out / "candidates.csv")
    print(mail_text)
    print(f"\nFiles: {out / 'report.html'} | {out / 'candidates.csv'}")
    should_email = args.email or (args.scheduled and datetime.now(timezone.utc).weekday() == 5)
    if should_email:
        if os.environ.get("SMTP_HOST"):
            send_email("FairValue Scout — Saturday's top five", mail_text, page)
            LOG.info("Sent report email")
        else:
            LOG.warning("Email skipped: SMTP secrets not configured. Report remains available in generated files.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank $10B+ US-listed stocks by six-model fair-value gap")
    p.add_argument("--limit", type=int, default=0, help="Max names to fetch. 0 means full screener universe; 50 is a quick test")
    p.add_argument("--tickers", help="Optional CSV of tickers, one per row, for a custom watchlist")
    p.add_argument("--exclude", default="exclude.txt", help="Tickers to exclude (one per line)")
    p.add_argument("--cache", default=".cache/fundamentals.json")
    p.add_argument("--cache-hours", type=int, default=24, help="Cached fundamentals lifetime (default 24h)")
    p.add_argument("--delay", type=float, default=0.35, help="Seconds to pause between requests")
    p.add_argument("--output", default="reports")
    p.add_argument("--email", action="store_true", help="Email report now (needs SMTP env vars)")
    p.add_argument("--scheduled", action="store_true", help="Email report Saturdays (UTC) when SMTP is configured")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)
