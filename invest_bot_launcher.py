"""
Invest Bot 主選單（PyInstaller 打包入口）。
執行檔會在同目錄讀寫 CSV / .env / checkpoint。
"""
from __future__ import annotations

import argparse
import sys


def _pause():
    if getattr(sys, "frozen", False):
        input("\n按 Enter 結束...")


def _menu():
    print(
        """
╔══════════════════════════════════════╗
║         Invest Bot — Quant Screener    ║
╚══════════════════════════════════════╝
  1  完整掃描 (global_screener)
  2  只重跑 LLM (partial_rerun)
  3  重算 Volume Attention (refresh_volume_attention)
  4  Tier 回測 (backtest)
  5  Attention 分組回測 (backtest_attention)
  6  離開
"""
    )
    return input("請選擇 [1-6]: ").strip()


def _run_full_scan():
    from global_screener import main

    main()


def _run_partial_rerun():
    import partial_rerun

    partial_rerun.main()


def _run_volume_refresh():
    import refresh_volume_attention

    refresh_volume_attention.main()


def _run_backtest():
    from global_screener import run_backtest

    mode = input("模式 compounder / growth / both [both]: ").strip() or "both"
    modes = ["compounder", "growth"] if mode == "both" else [mode]
    for m in modes:
        run_backtest(f"results_{m}.csv", backtest_output=f"backtest_{m}.csv")


def _run_attention_backtest():
    import backtest_attention

    backtest_attention.main()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Invest Bot launcher")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["scan", "partial", "volume", "backtest", "attention", "menu"],
        default="menu",
        help="Direct command (default: interactive menu)",
    )
    args = parser.parse_args(argv)

    cmd_map = {
        "scan": _run_full_scan,
        "partial": _run_partial_rerun,
        "volume": _run_volume_refresh,
        "backtest": _run_backtest,
        "attention": _run_attention_backtest,
    }

    if args.command != "menu":
        try:
            cmd_map[args.command]()
        except KeyboardInterrupt:
            print("\n已中斷。")
        except Exception as e:
            print(f"\n錯誤: {e}")
            _pause()
            sys.exit(1)
        _pause()
        return

    while True:
        choice = _menu()
        if choice == "1":
            try:
                _run_full_scan()
            except KeyboardInterrupt:
                print("\n已中斷。")
            _pause()
        elif choice == "2":
            sys.argv = ["partial_rerun.py"]
            try:
                _run_partial_rerun()
            except KeyboardInterrupt:
                print("\n已中斷。")
            _pause()
        elif choice == "3":
            sys.argv = ["refresh_volume_attention.py"]
            try:
                _run_volume_refresh()
            except KeyboardInterrupt:
                print("\n已中斷。")
            _pause()
        elif choice == "4":
            try:
                _run_backtest()
            except KeyboardInterrupt:
                print("\n已中斷。")
            _pause()
        elif choice == "5":
            sys.argv = ["backtest_attention.py"]
            try:
                _run_attention_backtest()
            except KeyboardInterrupt:
                print("\n已中斷。")
            _pause()
        elif choice == "6":
            break
        else:
            print("無效選項。")


if __name__ == "__main__":
    main()
