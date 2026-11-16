import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Stock, dcf_per_share, evaluate, filter_models, get_models, reverse_dcf, sector_medians, build_report, price_check, debt_health


def stock(**kw):
    base = dict(symbol="EXAMPLE", name="Fictional Industrial Inc.", price=100,
                cap=20_000_000_000, sector="Industrials", industry="Machinery",
                shares=200_000_000, fcf=1_000_000_000, revenue_growth=.06,
                fwd_eps=7, trailing_eps=6.5, forward_pe=15, ev_ebitda=12,
                ebitda=2_000_000_000, debt=5_000_000_000, cash=2_000_000_000,
                bvps=50, roe=.15, dividend_yield=.025, buyback_yield=.015)
    base.update(kw)
    return Stock(**base)


class ValuationTests(unittest.TestCase):
    def test_dcf_is_finite(self):
        self.assertGreater(dcf_per_share(stock()), 0)
        self.assertLess(dcf_per_share(stock()), 150)

    def test_graham_equation(self):
        raw, exc = get_models(stock(), {"Industrials": 15}, {"Industrials": 12})
        self.assertAlmostEqual(raw["Graham number"], (22.5 * 6.5 * 50) ** .5)
        self.assertEqual(raw["Forward P/E"], 105)
        self.assertAlmostEqual(raw["EV/EBITDA"], 105)
        self.assertIn("Shareholder yield", raw)
        self.assertIn("Residual income", exc)

    def test_financials_use_residual_not_dcf_or_ev(self):
        lender = stock(sector="Financial Services", industry="Banks - Regional", roe=.16)
        raw, exc = get_models(lender, {"Financial Services": 12}, {"Financial Services": 10})
        self.assertIn("Residual income", raw)
        self.assertNotIn("DCF", raw)
        self.assertNotIn("EV/EBITDA", raw)
        self.assertAlmostEqual(raw["Residual income"], 50 * (1 + (.16 - .105)/(.105-.025)))

    def test_outlier_filter(self):
        kept, dropped = filter_models({"a": 120, "b": 130, "c": 370, "d": 10}, 100)
        self.assertEqual(kept, {"a": 120, "b": 130})
        self.assertIn("c", dropped)
        self.assertIn("d", dropped)

    def test_evaluation_fails_with_one_model(self):
        s = stock(fcf=None, trailing_eps=None, fwd_eps=8, ebitda=None,
                  dividend_yield=None, buyback_yield=None)
        self.assertIsNone(evaluate(s, {"Industrials": 13}, {}))

    def test_mismatched_currency_rejected(self):
        self.assertIsNone(evaluate(stock(financial_currency="CAD"), {"Industrials": 15}, {"Industrials": 12}))

    def test_sector_needs_three_peers(self):
        pe, ev = sector_medians([stock(symbol="A", forward_pe=10, ev_ebitda=10), stock(symbol="B", forward_pe=20, ev_ebitda=20)])
        self.assertEqual(pe, {})
        self.assertEqual(ev, {})
        pe, ev = sector_medians([stock(symbol="A", forward_pe=10, ev_ebitda=10), stock(symbol="B", forward_pe=20, ev_ebitda=20), stock(symbol="C", forward_pe=15, ev_ebitda=15)])
        self.assertEqual(pe["Industrials"], 15)
        self.assertEqual(ev["Industrials"], 15)

    def test_reverse_dcf_diagnostic(self):
        self.assertIsInstance(reverse_dcf(stock()), str)
        self.assertNotEqual(reverse_dcf(stock()), "N/A")

    def test_price_history_quote_and_split_validation(self):
        import pandas as pd
        from datetime import datetime, timezone
        now = datetime(2026, 11, 16, 14, tzinfo=timezone.utc)
        idx = pd.to_datetime(["2026-11-12", "2026-11-13"], utc=True)
        bars = pd.DataFrame({"Close": [100.0, 102.0], "Stock Splits": [0, 0]}, index=idx)
        self.assertEqual(price_check(103, bars, now)[0], "verified")
        self.assertEqual(price_check(200, bars, now)[0], "mismatch")
        bars.loc[idx[-1], "Stock Splits"] = 2.0
        self.assertEqual(price_check(103, bars, now)[0], "recent_split")
        bars.index = pd.to_datetime(["2026-10-01", "2026-10-02"], utc=True)
        self.assertEqual(price_check(103, bars, now)[0], "stale")
        self.assertEqual(price_check(103, None, now)[0], "unverified")

    def test_financial_health_flags_and_missing_metrics(self):
        self.assertEqual(debt_health(stock())[0], "Unknown")
        self.assertEqual(debt_health(stock(current_ratio=0.7))[0], "Elevated risk")
        self.assertEqual(debt_health(stock(ebitda=100_000, debt=4_000_000, cash=0,
                                           current_ratio=2))[0], "Elevated risk")
        self.assertEqual(debt_health(stock(debt_to_equity=90, current_ratio=2))[0], "No red flags")
        self.assertEqual(debt_health(stock(industry="Banks - Regional"))[0], "Not comparable")

    def test_high_confidence_requires_four_agreeing_models_and_health(self):
        healthy = stock(fcf=1_450_000_000, trailing_eps=8.5, buyback_yield=None,
                        price_status="verified", debt_to_equity=90, current_ratio=1.6)
        result = evaluate(healthy, {"Industrials": 15}, {"Industrials": 12})
        self.assertEqual(result.confidence, "High confidence")
        self.assertGreaterEqual(result.valid_models, 4)
        self.assertLessEqual(result.dispersion_pct, 50)
        self.assertEqual(evaluate(stock(**{**healthy.__dict__, "price_status":"unverified"}),
                                  {"Industrials":15}, {"Industrials":12}).confidence, "Review only")

    def test_consensus_is_separate_and_report_is_limited_to_50(self):
        s = stock(analyst_target=123.5, analyst_count=15, target_source="Yahoo Finance / yfinance")
        result = evaluate(s, {"Industrials":15}, {"Industrials":12})
        self.assertEqual(result.analyst_target, 123.5)
        self.assertAlmostEqual(result.analyst_upside_pct, 23.5)
        page, _ = build_report([result] * 65, 500, 500, [result]*65)
        self.assertEqual(page.count('<tr data-industry='), 50)
        self.assertIn("12M analyst consensus", page)
        self.assertIn("Group/filter by industry", page)
        self.assertIn("$123.50", page)

    def test_report_safe_escaping(self):
        result = evaluate(stock(name="<script>alert(1)</script>"), {"Industrials": 15}, {"Industrials": 12})
        self.assertIsNotNone(result)
        page, text = build_report([result], 1, 1, [result])
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("EXAMPLE", text)

if __name__ == "__main__":
    unittest.main()
