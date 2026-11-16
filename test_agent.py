import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Stock, dcf_per_share, evaluate, filter_models, get_models, reverse_dcf, sector_medians, build_report


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

    def test_report_safe_escaping(self):
        result = evaluate(stock(name="<script>alert(1)</script>"), {"Industrials": 15}, {"Industrials": 12})
        self.assertIsNotNone(result)
        page, text = build_report([result], 1, 1, [result])
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("EXAMPLE", text)

if __name__ == "__main__":
    unittest.main()
