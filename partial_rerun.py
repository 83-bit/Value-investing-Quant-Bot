"""
Partial rerun: 保留 results_*.csv 內 quant 分數，只重跑 LLM 失敗列。

用法:
  python partial_rerun.py                    # compounder + growth，全部失敗列
  python partial_rerun.py --mode compounder    # 只 compounder
  python partial_rerun.py --limit 10           # 試跑 10 支
  python partial_rerun.py --dry-run            # 只統計，不呼叫 API
  python partial_rerun.py --one-pagers         # 完成後重做 S/A One-Pager
  python partial_rerun.py --fetch-fmp          # 強制重打 FMP（易 403，一般不建議）
"""
import argparse
import pandas as pd

from global_screener import needs_llm_rerun, partial_rerun_csv, run_backtest


def _count(path: str) -> int:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return int(df.apply(needs_llm_rerun, axis=1).sum())


def main():
    parser = argparse.ArgumentParser(description="Rerun LLM only for failed rows in results CSV")
    parser.add_argument(
        "--mode",
        choices=["compounder", "growth", "both"],
        default="both",
        help="Which results file to process",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max tickers to rerun (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Count failed rows only")
    parser.add_argument("--one-pagers", action="store_true", help="Regenerate one-pagers after rerun")
    parser.add_argument("--backtest", action="store_true", help="Run backtest after rerun")
    parser.add_argument(
        "--fetch-fmp",
        action="store_true",
        help="Re-fetch FMP data (default: CSV + yfinance only, no FMP)",
    )
    args = parser.parse_args()

    modes = ["compounder", "growth"] if args.mode == "both" else [args.mode]

    if args.dry_run:
        for m in modes:
            path = f"results_{m}.csv"
            n = _count(path)
            print(f"{path}: {n} rows need LLM partial rerun")
        return

    for m in modes:
        path = f"results_{m}.csv"
        partial_rerun_csv(
            path,
            limit=args.limit,
            regenerate_one_pagers=args.one_pagers,
            use_fmp=args.fetch_fmp,
        )
        if args.backtest:
            run_backtest(path, backtest_output=f"backtest_{m}.csv")


if __name__ == "__main__":
    main()
