================================================================================
  Invest Bot — Quantamental Global Stock Screener
================================================================================

Overview
--------
Invest Bot is a Python quantamental screener that scans a global stock universe
across US, Europe, Japan, Hong Kong, and Taiwan. It combines:

  - Quantitative scoring: ROIC, DuPont, FCF, ROIC momentum, valuation bands, etc.
  - Volume Attention: volume Z-Score to flag "hidden pearls" and "consensus shifts"
  - LLM qualitative analysis: DeepSeek three-stage debate (Grit / Culture / Catalyst)
  - Optional SEC 10-K RAG for US equities
  - Automatic tiering (S / A / B / C), One-Pager reports, and tier backtests

Two screening modes are supported:

  compounder  Compounding machines — high ROIC, durable capital allocation
  growth      Growth breakouts — growth-oriented quantitative thresholds

You can run both modes in one pass (SCREEN_MODE = "both") and get two result lists.


Requirements
------------
  - Python 3.10 or later (3.11+ recommended)
  - Windows (GUI and .exe build script use PowerShell)
  - Internet access (FMP, DeepSeek, yfinance)
  - API keys:
      DEEPSEEK_API_KEY  — DeepSeek LLM
      FMP_API_KEY       — Financial Modeling Prep market data


Installation
------------
1. Clone or extract the project to a local folder.

2. Install dependencies:

     pip install -r requirements.txt

   Main packages: python-dotenv, yfinance, pandas, numpy, requests, openai, tenacity

3. Configure environment variables:

     Copy .env.example to .env
     Set DEEPSEEK_API_KEY and FMP_API_KEY

   Note: .env contains secrets — do not commit it to version control.


Quick Start
-----------

[GUI — recommended]

     python invest_bot_gui.py

  Or run the packaged dist\InvestBot.exe (place .env in the same folder).

  The GUI supports: full scan, LLM partial rerun, volume refresh, tier backtest,
  and attention-group backtest, with live logs in the window.

[CLI — launcher menu]

     python invest_bot_launcher.py

  Interactive menu 1–6, or pass a subcommand directly:

     python invest_bot_launcher.py scan        # full scan
     python invest_bot_launcher.py partial     # rerun LLM only
     python invest_bot_launcher.py volume      # refresh Volume Attention
     python invest_bot_launcher.py backtest    # tier backtest
     python invest_bot_launcher.py attention   # attention-group backtest

[CLI — run core script directly]

     python global_screener.py                 # full scan (same as launcher scan)


Scripts
-------

global_screener.py
  Core engine. Loads global universe → quant scoring → LLM debate → tiering →
  CSV output, One-Pagers, optional auto backtest. Config block is at the top
  of the file (roughly lines 1–100).

invest_bot_gui.py
  tkinter GUI. Long-running tasks run in background threads; logs stream to the window.

invest_bot_launcher.py
  CLI menu; can also serve as a PyInstaller entry point.

partial_rerun.py
  Keeps existing quant scores in results_*.csv and reruns only failed LLM rows.

  Examples:
    python partial_rerun.py
    python partial_rerun.py --mode compounder
    python partial_rerun.py --limit 10
    python partial_rerun.py --dry-run
    python partial_rerun.py --one-pagers --backtest

refresh_volume_attention.py
  Recomputes Volume Attention (Momentum-Attention Score) and writes back to CSV.
  Does not call the LLM or recompute ROIC/FCF.

  Examples:
    python refresh_volume_attention.py
    python refresh_volume_attention.py --dry-run
    python refresh_volume_attention.py --mode compounder --limit 20

  After a volume refresh, run backtest_attention.py to review grouped backtests.

backtest_attention.py
  Backtests S/A tiers by attention_signal (pearl / surge / neutral).

  Examples:
    python backtest_attention.py
    python backtest_attention.py --mode both --years 2

audit_thesis.py
  Audits investment theses against recent price action; writes audit_failures.json
  for Judge failure-context injection (USE_FAILURE_CONTEXT).

  Examples:
    python audit_thesis.py
    python audit_thesis.py --csv results_compounder.csv --lookback-days 90


Output Files
------------

results_compounder.csv      full compounder results
results_growth.csv            full growth results

backtest_compounder.csv       compounder tier backtest
backtest_growth.csv           growth tier backtest
backtest_*_attention.csv      attention-group backtest

one_pagers_compounder/        S/A One-Pager Markdown (compounder)
one_pagers_growth/            S/A One-Pager Markdown (growth)

checkpoint.db                 SQLite checkpoint (resume after interruption)
audit_failures.json           thesis audit failure cases

Key CSV columns include: ticker, region, tier, total_score, quant_score,
qual_score, valuation, roic, investment_thesis, attention_signal, volume_z, etc.


Key Settings (top of global_screener.py)
------------------------------------------
  SCREEN_MODE          "compounder" | "growth" | "both"
  USE_LLM              enable LLM qualitative analysis
  USE_DEBATE           True = three-stage debate; False = single LLM pass
  USE_SEC_RAG          US SEC 10-K deep retrieval (off when FAST_MODE=True)
  USE_VOLUME_ATTENTION volume Z-Score bonus scoring
  MAX_WORKERS          thread pool size (default 5)
  USE_CHECKPOINT       resume from checkpoint
  GENERATE_ONE_PAGERS  generate One-Pagers after scan
  RUN_BACKTEST         run tier backtest when main() finishes

  Tier thresholds: S >= 85, A >= 70, B >= 50

  Full scan defaults to regions US, EU, JP, HK, TW; adjust in main().
  For testing, set max_stocks to cap universe size.


Suggested Workflow
------------------
1. Configure .env → run full scan (GUI or global_screener.py)
2. If LLM rows failed → partial_rerun.py
3. Refresh volume metrics → refresh_volume_attention.py
4. Review strategy performance → backtest_attention.py
5. Periodically audit theses → audit_thesis.py

A full scan can take hours and consume API quota; test with max_stocks or --limit first.


Build Windows Executable
------------------------
From the project root:

     .\build_exe.ps1

Output: dist\InvestBot.exe (GUI, no console window)

The script installs pyinstaller and pillow, then generates assets\invest_bot.ico.
Place .env next to the exe (or in dist\). The app reads and writes CSV, checkpoint,
and .env files from the executable's directory.


Project Layout (summary)
------------------------
  global_screener.py       core screener engine
  invest_bot_gui.py        graphical interface
  invest_bot_launcher.py   CLI menu
  partial_rerun.py         partial LLM rerun
  refresh_volume_attention.py
  backtest_attention.py
  audit_thesis.py
  build_exe.ps1            build script
  invest_bot.spec          PyInstaller spec
  requirements.txt         Python dependencies
  .env.example             environment template
  assets/                  icon and generate_icon.py


Disclaimer
----------
This tool is for research and education only. Output is not investment advice.
Evaluate API costs, data accuracy, and local regulations yourself. Investing
involves risk; you are responsible for your own decisions.

================================================================================
