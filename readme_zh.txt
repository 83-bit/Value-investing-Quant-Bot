================================================================================
  Invest Bot — Quantamental 全球選股篩選器
================================================================================

概述
----
Invest Bot 是一套 Python 量化＋質化（Quantamental）選股工具，掃描美、歐、日、
港、台等市場的股票 universe，結合：

  - 量化評分：ROIC、DuPont、FCF、ROIC 動量、估值帶等
  - Volume Attention：量能 Z-Score，辨識「冷門珍珠」與「共識轉折」
  - LLM 質化分析：DeepSeek 三階段辯論（Grit / Culture / Catalyst）
  - 可選 SEC 10-K RAG（美股深度檢索）
  - 自動分級（S / A / B / C）、One-Pager 報告、Tier 回測

支援兩種篩選模式：

  compounder  複利機器 — 高 ROIC、可持續資本配置
  growth      成長爆發 — 成長導向量化門檻

可同時產出兩份榜單（SCREEN_MODE = "both"）。


系統需求
--------
  - Python 3.10 或以上（建議 3.11+）
  - Windows（GUI / .exe 打包腳本為 PowerShell）
  - 網路連線（FMP、DeepSeek、yfinance）
  - API 金鑰：
      DEEPSEEK_API_KEY  — DeepSeek LLM
      FMP_API_KEY       — Financial Modeling Prep 財務數據


安裝
----
1. 克隆或解壓專案至本機資料夾。

2. 安裝依賴：

     pip install -r requirements.txt

   主要套件：python-dotenv, yfinance, pandas, numpy, requests, openai, tenacity

3. 設定環境變數：

     複製 .env.example 為 .env
     填入 DEEPSEEK_API_KEY 與 FMP_API_KEY

   注意：.env 含敏感金鑰，請勿提交至版本控制。


快速開始
--------

【圖形介面 — 推薦】

     python invest_bot_gui.py

  或執行打包後的 dist\InvestBot.exe（需在同目錄放置 .env）。

  GUI 提供：完整掃描、重跑 LLM、重算 Volume、Tier 回測、Attention 回測，
  並在視窗內顯示執行紀錄。

【命令列 — 主選單】

     python invest_bot_launcher.py

  互動選單 1–6，或直接指定子命令：

     python invest_bot_launcher.py scan        # 完整掃描
     python invest_bot_launcher.py partial     # 只重跑 LLM
     python invest_bot_launcher.py volume      # 重算 Volume Attention
     python invest_bot_launcher.py backtest    # Tier 回測
     python invest_bot_launcher.py attention   # Attention 分組回測

【命令列 — 直接執行核心腳本】

     python global_screener.py                 # 完整掃描（等同 launcher scan）


腳本說明
--------

global_screener.py
  核心引擎。載入全球 universe → 量化評分 → LLM 辯論 → 分級 → 輸出 CSV、
  One-Pager、可選自動回測。設定區塊位於檔案開頭（約第 1–100 行）。

invest_bot_gui.py
  tkinter 視窗版。長時間任務在背景執行緒執行，log 即時顯示。

invest_bot_launcher.py
  命令列主選單；PyInstaller 打包時亦可作為無視窗入口。

partial_rerun.py
  保留 results_*.csv 內已有量化分數，只重跑 LLM 失敗列。

  用法範例：
    python partial_rerun.py
    python partial_rerun.py --mode compounder
    python partial_rerun.py --limit 10
    python partial_rerun.py --dry-run
    python partial_rerun.py --one-pagers --backtest

refresh_volume_attention.py
  只重算 Volume Attention（Momentum-Attention Score），寫回現有 CSV。
  不呼叫 LLM、不重算 ROIC/FCF。

  用法範例：
    python refresh_volume_attention.py
    python refresh_volume_attention.py --dry-run
    python refresh_volume_attention.py --mode compounder --limit 20

  重算 Volume 後，建議執行 backtest_attention.py 檢視分組回測。

backtest_attention.py
  依 S/A tier × attention_signal（珍珠 / 共識轉折 / neutral）分組回測。

  用法範例：
    python backtest_attention.py
    python backtest_attention.py --mode both --years 2

audit_thesis.py
  對照近期股價審計 investment thesis，產出 audit_failures.json，
  供後續 Judge 注入失敗案例（USE_FAILURE_CONTEXT）。

  用法範例：
    python audit_thesis.py
    python audit_thesis.py --csv results_compounder.csv --lookback-days 90


輸出檔案
--------

results_compounder.csv      compounder 模式完整結果
results_growth.csv            growth 模式完整結果

backtest_compounder.csv       compounder Tier 回測
backtest_growth.csv           growth Tier 回測
backtest_*_attention.csv      Attention 分組回測

one_pagers_compounder/        S/A 級 One-Pager Markdown（compounder）
one_pagers_growth/            S/A 級 One-Pager Markdown（growth）

checkpoint.db                 SQLite 斷點續傳（中斷後重跑可接續）
audit_failures.json           Thesis 審計失敗案例

CSV 主要欄位含：ticker, region, tier, total_score, quant_score, qual_score,
valuation, roic, investment_thesis, attention_signal, volume_z 等。


主要設定（global_screener.py 開頭）
----------------------------------
  SCREEN_MODE          "compounder" | "growth" | "both"
  USE_LLM              是否啟用 LLM 質化分析
  USE_DEBATE           True = 三階段辯論；False = 單一 LLM
  USE_SEC_RAG          美股 SEC 10-K 深度檢索（FAST_MODE=True 時自動關閉）
  USE_VOLUME_ATTENTION 量能 Z-Score 加分
  MAX_WORKERS          並發執行緒數（預設 5）
  USE_CHECKPOINT       斷點續傳
  GENERATE_ONE_PAGERS  掃描完成後生成 One-Pager
  RUN_BACKTEST         main() 結束後自動跑 Tier 回測

  Tier 門檻：S ≥ 85，A ≥ 70，B ≥ 50

  完整掃描預設 regions = US, EU, JP, HK, TW；可在 main() 內調整。
  測試時可設 max_stocks 限制 universe 大小。


建議工作流程
------------
1. 設定 .env → 執行完整掃描（GUI 或 global_screener.py）
2. 若 LLM 部分失敗 → partial_rerun.py
3. 更新量能指標 → refresh_volume_attention.py
4. 檢視策略效果 → backtest_attention.py
5. 定期審計 thesis → audit_thesis.py

完整掃描可能耗時數小時並消耗 API 配額；建議先用 max_stocks 或 --limit 試跑。


打包 Windows 執行檔
-------------------
在專案根目錄執行：

     .\build_exe.ps1

產出：dist\InvestBot.exe（圖形介面、無主控台視窗）

打包前會自動安裝 pyinstaller、pillow，並生成 assets\invest_bot.ico。
請將 .env 放在 exe 同目錄（或 dist\ 資料夾），程式會在執行檔所在目錄
讀寫 CSV、checkpoint 與 .env。


專案結構（精簡）
----------------
  global_screener.py       核心篩選引擎
  invest_bot_gui.py        圖形介面
  invest_bot_launcher.py   CLI 主選單
  partial_rerun.py         LLM 部分重跑
  refresh_volume_attention.py
  backtest_attention.py
  audit_thesis.py
  build_exe.ps1            打包腳本
  invest_bot.spec          PyInstaller 設定
  requirements.txt         Python 依賴
  .env.example             環境變數範本
  assets/                  圖示與 generate_icon.py


免責聲明
--------
本工具僅供研究與教育用途，輸出結果不構成投資建議。
使用前請自行評估 API 費用、數據準確性與當地法規。投資有風險，決策請自負。

================================================================================
