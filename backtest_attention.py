"""
S/A tier × attention_signal 分組回測（珍珠 / 共識轉折 / neutral）。

用法:
  python backtest_attention.py
  python backtest_attention.py --mode compounder
  python backtest_attention.py --mode both --years 2
"""
import argparse

from global_screener import run_backtest_attention_groups


def main():
    parser = argparse.ArgumentParser(description="Backtest S/A by attention_signal groups")
    parser.add_argument("--mode", choices=["compounder", "growth", "both"], default="compounder")
    parser.add_argument("--years", type=int, default=2, help="Lookback years")
    args = parser.parse_args()

    modes = ["compounder", "growth"] if args.mode == "both" else [args.mode]
    for mode in modes:
        csv_path = f"results_{mode}.csv"
        out_path = f"backtest_{mode}_attention.csv"
        run_backtest_attention_groups(
            results_csv_path=csv_path,
            lookback_years=args.years,
            backtest_output=out_path,
        )


if __name__ == "__main__":
    main()
