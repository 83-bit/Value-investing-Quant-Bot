# Invest Bot — Quantamental Global Stock Screener

A Python screener that runs a **quantitative first pass** over a global stock
universe (US · Europe · Japan · Hong Kong · Taiwan), then hands the survivors to
an **LLM debate**, then **backtests** whether the tiers it produced actually
worked.

Built because screening thousands of companies by hand isn't possible, and
because a process you can't backtest is just a story.

---

## What it does

| Stage | What happens |
|---|---|
| **1. Quant** | ROIC, DuPont breakdown, FCF, ROIC momentum, valuation bands |
| **2. Attention** | Volume Z-score flags *hidden pearls* and *consensus shifts* |
| **3. LLM debate** | DeepSeek three-stage debate, one stage per question: **Grit / Culture / Catalyst** |
| **4. Tiering** | Auto S ≥ 85 · A ≥ 70 · B ≥ 50, with One-Pager reports for S/A |
| **5. Backtest** | Tier backtests and attention-group backtests, so the ranking is testable |

Optional **SEC 10-K RAG** for US equities, when you want the qualitative layer to
argue from the filing instead of the headline.

Two screening modes:

- `compounder` — compounding machines: high ROIC, durable capital allocation
- `growth` — growth breakouts: growth-oriented quantitative thresholds

Set `SCREEN_MODE = "both"` to run both in one pass.

---

## Design notes

Three decisions that shape everything else:

**Quant first, narrative second.** The LLM never sees a company until the
quantitative screen has passed it. The model is there to *argue with* the
numbers, not to replace them.

**The debate is staged, not monolithic.** Grit, culture and catalyst are asked
separately, because a single blended prompt reliably produces a single blended
compliment. Splitting them forces the model to take a position on each.

**Everything is backtestable.** Tiers, attention signals, and theses all get
scored against later price action — including `audit_thesis.py`, which exists
purely to find the cases where the thesis was wrong.

---

## Requirements

- Python 3.10+ (3.11+ recommended)
- Windows for the GUI and the `.exe` build (both use PowerShell)
- Internet access (FMP, DeepSeek, yfinance)
- API keys:

| Key | For |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek LLM |
| `FMP_API_KEY` | Financial Modeling Prep market data |

---

## Installation

```bash
git clone https://github.com/83-bit/Value-investing-Quant-Bot.git
cd Value-investing-Quant-Bot
pip install -r requirements.txt
```

Dependencies: `python-dotenv`, `yfinance`, `pandas`, `numpy`, `requests`,
`openai`, `tenacity`.

Then configure secrets:

```bash
cp .env.example .env
# set DEEPSEEK_API_KEY and FMP_API_KEY
```

> `.env` holds secrets — never commit it.

---

## Quick start

**GUI (recommended)**

```bash
python invest_bot_gui.py
```

Supports full scan, LLM partial rerun, volume refresh, tier backtest and
attention-group backtest, with live logs streamed into the window.

**CLI — interactive menu**

```bash
python invest_bot_launcher.py
```

Or call a stage directly:

```bash
python invest_bot_launcher.py scan        # full scan
python invest_bot_launcher.py partial     # rerun failed LLM rows only
python invest_bot_launcher.py volume      # refresh Volume Attention
python invest_bot_launcher.py backtest    # tier backtest
python invest_bot_launcher.py attention   # attention-group backtest
```

**CLI — core engine directly**

```bash
python global_screener.py
```

> A full scan can take hours and burn API quota. Test with `max_stocks` or
> `--limit` first.

---

## Scripts

| Script | Role |
|---|---|
| `global_screener.py` | Core engine. Universe → quant scoring → LLM debate → tiering → CSV, One-Pagers, optional auto backtest. Config block sits at the top (~lines 1–100). |
| `invest_bot_gui.py` | tkinter GUI. Long tasks run in background threads; logs stream to the window. |
| `invest_bot_launcher.py` | CLI menu; also usable as a PyInstaller entry point. |
| `partial_rerun.py` | Keeps existing quant scores in `results_*.csv` and reruns **only failed LLM rows**. |
| `refresh_volume_attention.py` | Recomputes Volume Attention (Momentum-Attention Score) and writes back to CSV. Doesn't call the LLM or recompute ROIC/FCF. |
| `backtest_attention.py` | Backtests S/A tiers grouped by `attention_signal` (pearl / surge / neutral). |
| `audit_thesis.py` | Audits theses against recent price action; writes `audit_failures.json` for Judge failure-context injection. |
| `build_exe.ps1` | Windows build script → `dist/InvestBot.exe`. |

```bash
python partial_rerun.py --mode compounder --limit 10
python refresh_volume_attention.py --mode compounder --limit 20
python backtest_attention.py --mode both --years 2
python audit_thesis.py --csv results_compounder.csv --lookback-days 90
```

---

## Output files

| File | Contents |
|---|---|
| `results_compounder.csv` / `results_growth.csv` | Full screen results |
| `backtest_compounder.csv` / `backtest_growth.csv` | Tier backtests |
| `backtest_*_attention.csv` | Attention-group backtests |
| `one_pagers_compounder/` / `one_pagers_growth/` | S/A One-Pager Markdown |
| `checkpoint.db` | SQLite checkpoint — resume after interruption |
| `audit_failures.json` | Thesis audit failure cases |

Key CSV columns: `ticker`, `region`, `tier`, `total_score`, `quant_score`,
`qual_score`, `valuation`, `roic`, `investment_thesis`, `attention_signal`,
`volume_z`.

---

## Key settings

Top of `global_screener.py`:

| Setting | Meaning |
|---|---|
| `SCREEN_MODE` | `"compounder"` \| `"growth"` \| `"both"` |
| `USE_LLM` | Enable the LLM qualitative layer |
| `USE_DEBATE` | `True` = three-stage debate; `False` = single LLM pass |
| `USE_SEC_RAG` | US SEC 10-K retrieval (off when `FAST_MODE=True`) |
| `USE_VOLUME_ATTENTION` | Volume Z-score bonus scoring |
| `MAX_WORKERS` | Thread pool size (default 5) |
| `USE_CHECKPOINT` | Resume from checkpoint |
| `GENERATE_ONE_PAGERS` | Emit One-Pagers after scan |
| `RUN_BACKTEST` | Run tier backtest when `main()` finishes |

Tier thresholds: `S ≥ 85`, `A ≥ 70`, `B ≥ 50`. Default regions: US, EU, JP, HK, TW.

**Suggested workflow**

1. Configure `.env` → run a full scan
2. If LLM rows failed → `partial_rerun.py`
3. Refresh volume metrics → `refresh_volume_attention.py`
4. Review strategy performance → `backtest_attention.py`
5. Periodically audit theses → `audit_thesis.py`

---

## Build a Windows executable

```powershell
.\build_exe.ps1
```

Output: `dist\InvestBot.exe` (GUI, no console window). The script installs
`pyinstaller` and `pillow`, and generates `assets\invest_bot.ico`. Place `.env`
next to the exe — the app reads and writes CSV, checkpoint and `.env` files from
the executable's directory.

---

## Disclaimer

For research and education only. Nothing here is investment advice. Verify data
accuracy, API costs and your local regulations yourself. Investing involves
risk — you are responsible for your own decisions.
