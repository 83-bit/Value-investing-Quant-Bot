# Invest Bot — Agent Guide

Cursor rules (`.cursor/rules/`):

| Rule | Scope |
|------|--------|
| `fullstack-standards.mdc` | Always — persona, TS standards, Playwright MCP, concise output |
| `invest-bot-context.mdc` | Always — Python screener project context |
| `invest-bot-python.mdc` | `*.py` — backend conventions |
| `nextjs-frontend.mdc` | `*.ts(x)` — Next.js / React when frontend exists |

## Quick start (backend)
1. Copy `.env.example` → `.env` with `DEEPSEEK_API_KEY` and `FMP_API_KEY`
2. GUI: `python invest_bot_gui.py` or `dist/InvestBot.exe`
3. CLI: `python global_screener.py`

## Before editing
- Read config block at top of `global_screener.py`
- Compounder deliverables > growth mode
- After volume refresh, run `python backtest_attention.py`

## Build exe
```powershell
.\build_exe.ps1
```
Output: `dist/InvestBot.exe` (place `.env` beside exe)
