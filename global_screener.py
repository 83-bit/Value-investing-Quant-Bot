from dotenv import load_dotenv
import os
import sys


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = _app_dir()
os.chdir(APP_DIR)
load_dotenv(os.path.join(APP_DIR, ".env"), override=True)

# =================================================================
# 1. DEEPSEEK API KEY (read from .env file)
# =================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/api/v3"

# =================================================================
# 2. Config
# =================================================================
RISK_FREE_RATE = 4.5               # 10‑year US Treasury proxy (%)
BASE_RISK_FREE = 4.5               # 估值帶锚點；高於此利率 → 壓縮合理乘數
RATE_SENSITIVITY = 0.09            # 利率每高 1% → 合理乘數約 -9%
USE_LLM = True                     # set False to skip LLM parts
LLM_MODEL = "deepseek-chat"        # or deepseek-reasoner
USE_DEBATE = True                  # True=三階段辯論模式; False=單一LLM模式
USE_SEC_RAG = True                 # True=美股啟用SEC 10-K深度檢索
USE_TOOL_CALLING = True            # 保留供未來擴展；Judge 已改為 prefetch，不用 tool calling
JUDGE_USE_TOOLS = False            # Judge 用 tool calling 會破壞 JSON，必須 False
JUDGE_PREFETCH_TOOLS = True        # Judge 前預查 insider/機構/空頭（yfinance），嵌入 prompt

FAST_MODE = True                   # True=跳過 SEC RAG 加速執行（每支省 2~5 秒）
if FAST_MODE:
    USE_SEC_RAG = False

# ----- API 成本控制門檻 -----
DEBATE_MIN_QSCORE   = 15   # 辯論只對 q_score >= 15 的股票啟動（ROIC+DuPont+Footprint 全過）
SEC_RAG_MIN_QSCORE  = 10   # SEC RAG 只對美股 q_score >= 10 執行
TOOL_MAX_ROUNDS     = 2    # Judge Agent 動態工具最多呼叫輪數

# ----- DeepSeek 費率（USD / 1K tokens，2024Q4 定價）-----
DS_COST_INPUT_PER_1K  = 0.00014   # $0.14 / M input tokens
DS_COST_OUTPUT_PER_1K = 0.00028   # $0.28 / M output tokens

# Scoring weights (quantitative max 60, qualitative max 50)
SCORE_ROIC = 15
SCORE_DUPONT = 10
SCORE_FOOTPRINT = 5
SCORE_FCF = 20
SCORE_ROIC_MOMENTUM = 10  # 歷史 ROIC 趨勢動量分
SCORE_GRIT = 16.6
SCORE_CULTURE = 16.6
SCORE_CATALYST = 16.8

# 動態模型映射（根據 sector 自動選擇）
SECTOR_MODEL_MAP = {
    "Technology": "deepseek-reasoner",          # 科技股用推理模型，更嚴謹
    "Healthcare": "deepseek-reasoner",           # 醫療股需要理解複雜管線
    "Financial Services": "deepseek-chat",       # 金融股用快速模型
    "Energy": "deepseek-chat",
    "default": "deepseek-chat",                  # 其餘用快速模型
}
USE_DYNAMIC_MODEL = False   # 預設關閉（deepseek-reasoner 較貴），可手動開啟

USE_EARNINGS_SIGNALS = True        # 啟用 earnings call 訊號分析（+5分）
SCORE_EARNINGS_CONFIDENCE = 5      # earnings 信心分加分上限（量化滿分 60 → 65）
SCORE_CONTRARIAN = 8               # 逆向擁擠度：無人問津的優質股加分
USE_GOOGLE_TRENDS = False          # 預設關閉，pytrends 有時被 rate limit

# ----- 熱度衝擊指標 (Momentum-Attention Score) -----
USE_VOLUME_ATTENTION = True        # FMP quote/historical volume → Volume Z-Score
VOLUME_LOOKBACK_DAYS = 60          # 計算均值/標準差的天數
VOLUME_Z_PEARL_MAX = -0.5          # Z 低於此 → 冷門（潛在珍珠）
VOLUME_Z_BREAKOUT_MIN = 1.5        # Z 高於此 → 量能爆發
VOLUME_Z_SURGE_DELTA = 1.0         # Z 相對 20 日前急升 → 共識轉折
SCORE_VOLUME_PEARL = 6             # 高 quant + 低量能 Z
SCORE_VOLUME_BREAKOUT = 5          # 高 quant + 量能 Z 急升
QUANT_CORE_HIGH_MIN = 35           # 觸發 attention 加分的 quant 門檻（估值調整前）

RUN_BACKTEST = True                # main() 結束後自動跑回測
AUDIT_FAILURES_PATH = "audit_failures.json"
FAILURE_CONTEXT_SAMPLES = 4
USE_FAILURE_CONTEXT = True             # Judge 注入歷史 thesis 失敗案例
GENERATE_ONE_PAGERS = True             # 掃描完成後生成 S/A One-Pager Markdown

# 運行模式: "compounder"（複利機器）, "growth"（成長爆發）, "both"（兩份榜單）
SCREEN_MODE = "both"

# ----- 並發 & 斷點續傳 -----
MAX_WORKERS = 5                   # ThreadPoolExecutor 並發數
CHECKPOINT_DB = "checkpoint.db"   # SQLite 斷點檔案路徑
USE_CHECKPOINT = True              # 啟用斷點續傳（中斷後重跑可接續）

# Tier thresholds
S_TIER_MIN = 85
A_TIER_MIN = 70
B_TIER_MIN = 50

# =================================================================
# 3. Imports & setup
# =================================================================
# 建議安裝: pip install tenacity
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False

import yfinance as yf
import pandas as pd
import numpy as np
import json, time, logging, sys, requests, re, sqlite3, threading
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, Timeout


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
    timeout=Timeout(60.0, connect=15.0)
)


def _http_get_with_backoff(url, headers=None, timeout=15, max_retries=5, base_delay=1.0):
    """HTTP GET with exponential backoff; handles 429 Too Many Requests."""
    headers = headers or {}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and str(retry_after).isdigit() else base_delay * (2 ** attempt)
                wait = min(wait, 90)
                logging.warning(f"429 Too Many Requests — backoff {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            last_err = e
            code = e.response.status_code if e.response is not None else None
            if code == 429 and attempt < max_retries - 1:
                wait = min(base_delay * (2 ** attempt), 90)
                logging.warning(f"429 HTTPError — backoff {wait:.1f}s")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(min(base_delay * (2 ** attempt), 30))
                continue
            raise last_err
    return None


def fmp_get(endpoint, params=None, timeout=15):
    """FMP API GET 請求，自動帶 apikey；429 時指數退避重試"""
    if params is None:
        params = {}
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE}/{endpoint}"
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 60)
                logging.warning(f"FMP 429 on {endpoint} — wait {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "Error Message" in data:
                logging.warning(f"FMP error: {data['Error Message']}")
                return None
            return data
        except Exception as e:
            last_err = e
            msg = str(e)
            if FMP_API_KEY:
                msg = msg.replace(FMP_API_KEY, "***")
            if "429" in msg and attempt < 3:
                time.sleep(min(2 ** attempt * 2, 60))
                continue
            logging.warning(f"FMP request failed ({endpoint}): {msg}")
            return None
    logging.warning(f"FMP request failed ({endpoint}): {last_err}")
    return None


# =================================================================
# 4. Global universe loader — multi-region
# =================================================================

_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "UNH",
    "LLY", "JPM", "V", "XOM", "MA", "AVGO", "PG", "HD", "COST", "MRK",
    "ABBV", "CVX", "KO", "PEP", "ADBE", "WMT", "BAC", "MCD", "CRM", "CSCO",
    "TMO", "ACN", "ABT", "NFLX", "LIN", "DHR", "NEE", "NKE", "QCOM", "AMD",
    "TXN", "PM", "ORCL", "UPS", "RTX", "AMGN", "HON", "SPGI", "IBM", "LOW",
    "ELV", "SBUX", "MS", "GS", "INTU", "CAT", "BLK", "MDT", "AXP", "GE",
    "ADI", "ISRG", "NOW", "VRTX", "AMAT", "PLD", "DE", "T", "MO", "C",
    "GILD", "MDLZ", "MMC", "ZTS", "CB", "SYK", "BMY", "REGN", "SO", "DUK",
    "CI", "PGR", "ITW", "BSX", "BDX", "EMR", "TJX", "SCHW", "ICE", "MCO",
    "USB", "CME", "AON", "NSC", "ETN", "FIS", "WM", "ECL", "ROP", "APD",
    "SHW", "ADP", "EOG", "COF", "EW", "GD", "PSA", "MRNA", "KLAC", "LRCX",
    "MCHP", "AIG", "AFL", "ALL", "AEP", "D", "EXC", "SRE", "PEG", "XEL",
    "WEC", "DTE", "ED", "ES", "ETR", "PPL", "AES", "CMS", "EVRG", "LNT",
    "OTIS", "CARR", "CTRA", "OKE", "TRGP", "WMB", "KMI", "LNG", "DVN", "FANG",
    "MRO", "HAL", "SLB", "BKR", "MRO", "HAL", "WELL", "AMT", "CCI", "EQIX",
    "DLR", "O", "VICI", "SPG", "AVB", "EQR", "MAA", "UDR", "PSA", "ARE",
]

# Europe: STOXX 50 + FTSE 100 large-caps (yfinance suffixes)
_EU_TICKERS = [
    # Switzerland
    "NESN.SW", "NOVN.SW", "ROG.SW", "ABBN.SW", "ZURN.SW", "SREN.SW", "CFR.SW",
    "LONN.SW", "SLHN.SW", "GEBN.SW",
    # France
    "MC.PA", "TTE.PA", "SAN.PA", "OR.PA", "AIR.PA", "BNP.PA", "DG.PA",
    "RI.PA", "AI.PA", "SU.PA", "HO.PA", "CAP.PA", "SGO.PA", "VIV.PA",
    "ATO.PA", "RMS.PA", "KER.PA", "CS.PA", "GLE.PA", "ACA.PA",
    # Germany
    "SAP.DE", "SIE.DE", "ALV.DE", "MBG.DE", "BMW.DE", "DTE.DE", "BAYN.DE",
    "VOW3.DE", "DBK.DE", "ADS.DE", "MUV2.DE", "RWE.DE", "HEN3.DE",
    "BAS.DE", "EOAN.DE", "MTX.DE", "SHL.DE", "MRK.DE", "FRE.DE", "CON.DE",
    # Netherlands
    "ASML.AS", "PHIA.AS", "UNA.AS", "HEIA.AS", "NN.AS", "ABN.AS", "INGA.AS",
    "WKL.AS", "RAND.AS", "AD.AS",
    # UK (FTSE 100)
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L", "BHP.L",
    "GSK.L", "LLOY.L", "BARC.L", "NWG.L", "VOD.L", "BATS.L", "REL.L",
    "DGE.L", "PRU.L", "STAN.L", "NG.L", "WPP.L", "IMB.L", "GLEN.L",
    "AAL.L", "LSEG.L", "CPG.L", "IHG.L", "CNA.L", "EXPN.L", "BT-A.L",
    "BA.L", "RR.L", "SGRO.L", "LAND.L", "BLND.L", "PSN.L", "ABF.L",
    # Italy
    "ENEL.MI", "ENI.MI", "ISP.MI", "UCG.MI", "RACE.MI", "G.MI", "SRG.MI",
    "STM.MI", "TIT.MI", "MONC.MI",
    # Spain
    "IBE.MC", "SAN.MC", "ITX.MC", "BBVA.MC", "TEF.MC", "REP.MC",
    "AMS.MC", "FER.MC", "ACS.MC", "ELE.MC",
    # Sweden
    "ERIC-B.ST", "VOLV-B.ST", "ATCO-A.ST", "SEB-A.ST", "SHB-A.ST",
    "SWED-A.ST", "SKF-B.ST", "SAND.ST", "ALFA.ST", "EVO.ST",
    # Denmark
    "NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "ORSTED.CO", "CARL-B.CO",
    # Finland
    "NOKIA.HE", "NESTE.HE", "SAMPO.HE",
]

# Japan: Nikkei 225 large-caps (yfinance .T suffix)
_JP_TICKERS = [
    "7203.T",  # Toyota
    "6758.T",  # Sony
    "8306.T",  # Mitsubishi UFJ
    "6861.T",  # Keyence
    "8035.T",  # Tokyo Electron
    "9984.T",  # SoftBank
    "6098.T",  # Recruit
    "4063.T",  # Shin-Etsu Chemical
    "8411.T",  # Mizuho
    "9433.T",  # KDDI
    "6954.T",  # Fanuc
    "7267.T",  # Honda
    "9432.T",  # NTT
    "8766.T",  # Tokio Marine
    "4502.T",  # Takeda
    "9983.T",  # Fast Retailing
    "6367.T",  # Daikin
    "8001.T",  # Itochu
    "4519.T",  # Chugai
    "7741.T",  # HOYA
    "6594.T",  # Nidec
    "4568.T",  # Daiichi Sankyo
    "8802.T",  # Mitsubishi Estate
    "3382.T",  # Seven & i
    "9020.T",  # JR East
    "4661.T",  # OLC (Disney Japan)
    "5108.T",  # Bridgestone
    "7751.T",  # Canon
    "6301.T",  # Komatsu
    "8316.T",  # Sumitomo Mitsui
    "4543.T",  # Terumo
    "6702.T",  # Fujitsu
    "6503.T",  # Mitsubishi Electric
    "9022.T",  # JR Central
    "2802.T",  # Ajinomoto
    "4901.T",  # Fujifilm
    "7270.T",  # Subaru
    "6762.T",  # TDK
    "5401.T",  # Nippon Steel
    "8031.T",  # Mitsui & Co
    "6326.T",  # Kubota
    "4307.T",  # Nomura Research
    "6920.T",  # Lasertec
    "9613.T",  # NTT Data
    "2413.T",  # M3
    "7733.T",  # Olympus
    "4911.T",  # Shiseido
    "8750.T",  # Dai-ichi Life
    "7832.T",  # Bandai Namco
    "9735.T",  # Secom
]

# Hong Kong: HSI + major H-shares
_HK_TICKERS = [
    "0700.HK",  # Tencent
    "9988.HK",  # Alibaba
    "0005.HK",  # HSBC
    "0941.HK",  # China Mobile
    "1299.HK",  # AIA
    "2318.HK",  # Ping An
    "0388.HK",  # HKEX
    "1398.HK",  # ICBC
    "3988.HK",  # Bank of China
    "0939.HK",  # CCB
    "2628.HK",  # China Life
    "0001.HK",  # CKH Holdings
    "0016.HK",  # Sun Hung Kai
    "0066.HK",  # MTR
    "0883.HK",  # CNOOC
    "0857.HK",  # PetroChina
    "2388.HK",  # BOC HK
    "0002.HK",  # CLP
    "0003.HK",  # HK Gas
    "0011.HK",  # Hang Seng Bank
    "1113.HK",  # CK Asset
    "0017.HK",  # New World Dev
    "0012.HK",  # Henderson Land
    "0823.HK",  # Link REIT
    "6862.HK",  # Haidilao
    "9999.HK",  # NetEase
    "3690.HK",  # Meituan
    "1810.HK",  # Xiaomi
    "0175.HK",  # Geely
    "2020.HK",  # ANTA Sports
    "9618.HK",  # JD.com
    "0267.HK",  # CITIC
    "1038.HK",  # CKI Holdings
    "0006.HK",  # Power Assets
    "0101.HK",  # Hang Lung Properties
]

# Taiwan: TWSE large-caps (.TW suffix)
_TW_TICKERS = [
    "2330.TW",  # TSMC
    "2317.TW",  # Foxconn
    "2454.TW",  # MediaTek
    "2412.TW",  # Chunghwa Telecom
    "2308.TW",  # Delta Electronics
    "2382.TW",  # Quanta
    "2357.TW",  # ASUS
    "2303.TW",  # UMC
    "3711.TW",  # ASE Tech
    "2379.TW",  # Realtek
    "2395.TW",  # Advantech
    "2881.TW",  # Fubon Financial
    "2882.TW",  # Cathay Financial
    "2886.TW",  # Mega Financial
    "2891.TW",  # CTBC Financial
    "6505.TW",  # Formosa Petro
    "1301.TW",  # Formosa Plastics
    "1303.TW",  # Nan Ya Plastics
    "1326.TW",  # Formosa Chemicals
    "2002.TW",  # China Steel
    "2609.TW",  # Yang Ming Marine
    "2603.TW",  # Evergreen Marine
    "2615.TW",  # Wan Hai Lines
    "3008.TW",  # Largan Precision
    "2408.TW",  # Nanya Tech
    "2353.TW",  # Acer
    "2376.TW",  # Gigabyte
    "2327.TW",  # Innolux
    "5871.TW",  # Chailease
    "2880.TW",  # Hua Nan Financial
]

_REGION_TICKERS = {
    "EU": _EU_TICKERS,
    "JP": _JP_TICKERS,
    "HK": _HK_TICKERS,
    "TW": _TW_TICKERS,
}


def _fetch_us_tickers():
    """Fetch S&P 500 list from Wikipedia; fall back to hardcoded list."""
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(wiki_url, headers=_WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        tickers = pd.read_html(StringIO(resp.text))[0]["Symbol"].tolist()
        if len(tickers) >= 100:
            logging.info(f"S&P 500 loaded from Wikipedia: {len(tickers)} stocks")
            return tickers
        logging.warning(f"Wikipedia returned only {len(tickers)} tickers, using fallback")
    except Exception as e:
        logging.warning(f"Wikipedia fetch failed ({e}), using US fallback")
    return list(_SP500_FALLBACK)


def get_global_universe(max_stocks=None, regions=None):
    """
    Build a global stock universe across developed markets.

    Parameters
    ----------
    max_stocks : int or None
        Cap total tickers (applied after merging all regions). None = no cap.
    regions : list[str] or None
        Subset of ["US","EU","JP","HK","TW"]. None = all regions.

    Returns
    -------
    tickers : list[str]
    region_map : dict[str, str]   ticker -> region label
    """
    if regions is None:
        regions = ["US", "EU", "JP", "HK", "TW"]

    region_map = {}
    tickers_ordered = []

    for region in regions:
        if region == "US":
            us_list = _fetch_us_tickers()
            for t in us_list:
                if t not in region_map:
                    region_map[t] = "US"
                    tickers_ordered.append(t)
        elif region in _REGION_TICKERS:
            for t in _REGION_TICKERS[region]:
                if t not in region_map:
                    region_map[t] = region
                    tickers_ordered.append(t)

    if max_stocks:
        tickers_ordered = tickers_ordered[:max_stocks]
        region_map = {t: region_map[t] for t in tickers_ordered}

    total = len(tickers_ordered)
    breakdown = {r: sum(1 for v in region_map.values() if v == r) for r in regions}
    logging.info(f"Global universe: {total} stocks — {breakdown}")
    return tickers_ordered, region_map

# =================================================================
# 4b. 動態模型選擇
# =================================================================
def get_model_for_sector(sector: str) -> str:
    if not USE_DYNAMIC_MODEL:
        return LLM_MODEL
    return SECTOR_MODEL_MAP.get(sector, SECTOR_MODEL_MAP["default"])


# =================================================================
# 4c. 估值訊號（含利率敏感度）
# =================================================================
def _rate_multiple_scale(risk_free=None):
    """無風險利率越高，可接受的估值乘數越低（相對 BASE_RISK_FREE）。"""
    rf = RISK_FREE_RATE if risk_free is None else float(risk_free)
    spread = rf - BASE_RISK_FREE
    return max(0.55, 1.0 - spread * RATE_SENSITIVITY)


def _apply_rate_band(cheap, fair, expensive, risk_free=None):
    scale = _rate_multiple_scale(risk_free)
    return cheap * scale, fair * scale, expensive * scale, scale


def compute_valuation_signal(d: dict):
    """
    Returns ("便宜", reason), ("合理", reason), ("昂貴", reason), or ("無法判斷", reason)
    估值帶隨 RISK_FREE_RATE 動態壓縮/放寬（相對 BASE_RISK_FREE）。
    """
    sector = d.get("sector", "")
    pe = d.get("pe_ratio")
    pb = d.get("pb_ratio")
    ps = d.get("ps_ratio")
    peg = d.get("peg_ratio")
    ev_eb = d.get("ev_ebitda")
    rf_tag = f"RF={RISK_FREE_RATE:.1f}%"

    # 金融股用 P/B
    if sector in ("Financial Services", "Banks", "Insurance"):
        if pb is None:
            return ("無法判斷", "缺乏P/B數據")
        pb_cheap, pb_fair, _, scale = _apply_rate_band(1.0, 2.0, 3.0)
        if pb < pb_cheap:
            return ("便宜", f"P/B={pb:.1f}x < {pb_cheap:.1f}x（{rf_tag}, adj={scale:.2f}x）")
        if pb < pb_fair:
            return ("合理", f"P/B={pb:.1f}x（合理帶 {pb_cheap:.1f}–{pb_fair:.1f}x, {rf_tag}）")
        return ("昂貴", f"P/B={pb:.1f}x > {pb_fair:.1f}x（{rf_tag}, adj={scale:.2f}x）")

    # 科技/成長股用 PEG
    if sector in ("Technology", "Communication Services") and peg is not None:
        peg_cheap, peg_fair, _, scale = _apply_rate_band(1.0, 2.0, 3.0)
        if peg < peg_cheap:
            return ("便宜", f"PEG={peg:.1f} < {peg_cheap:.1f}（{rf_tag}, adj={scale:.2f}x）")
        if peg < peg_fair:
            return ("合理", f"PEG={peg:.1f}（合理帶 {peg_cheap:.1f}–{peg_fair:.1f}x, {rf_tag}）")
        return ("昂貴", f"PEG={peg:.1f} ≥ {peg_fair:.1f}（{rf_tag}）")

    # 一般股用 EV/EBITDA + PE 組合
    signals = []
    ev_detail = ""
    rate_scale = _rate_multiple_scale()
    if ev_eb is not None:
        roic = d.get("roic")
        if roic is not None and roic > 25:
            ev_cheap, ev_fair, ev_expensive = 12, 25, 40
        elif roic is not None and roic > 15:
            ev_cheap, ev_fair, ev_expensive = 10, 18, 30
        elif roic is not None and roic < 10:
            ev_cheap, ev_fair, ev_expensive = 6, 12, 20
        else:
            ev_cheap, ev_fair, ev_expensive = 10, 20, 30
        ev_cheap, ev_fair, ev_expensive, rate_scale = _apply_rate_band(
            ev_cheap, ev_fair, ev_expensive
        )

        if ev_eb < ev_cheap:
            signals.append("便宜")
        elif ev_eb < ev_fair:
            signals.append("合理")
        else:
            signals.append("昂貴")
        ev_detail = (
            f"EV/EBITDA={ev_eb:.0f}x（動態上限={ev_fair:.1f}x, ROIC={roic:.0f}%, {rf_tag}, adj={rate_scale:.2f}x）"
            if roic is not None else f"EV/EBITDA={ev_eb:.0f}x（{rf_tag}, adj={rate_scale:.2f}x）"
        )
    pe_cheap, pe_fair, _, _ = _apply_rate_band(15, 25, 35)
    if pe is not None:
        if pe < pe_cheap:
            signals.append("便宜")
        elif pe < pe_fair:
            signals.append("合理")
        else:
            signals.append("昂貴")

    if not signals:
        return ("無法判斷", "估值數據不足")

    cheap = signals.count("便宜")
    expensive = signals.count("昂貴")
    if pe and ev_eb:
        detail = f"PE={pe:.0f}x（帶 {pe_cheap:.0f}–{pe_fair:.0f}x）, {ev_detail}" if ev_detail else f"PE={pe:.0f}x, EV/EBITDA={ev_eb:.0f}x"
    elif pe:
        detail = f"PE={pe:.0f}x（帶 {pe_cheap:.0f}–{pe_fair:.0f}x, {rf_tag}）"
    else:
        detail = ev_detail if ev_detail else f"EV/EBITDA={ev_eb:.0f}x"

    if cheap > expensive:
        return ("便宜", detail)
    if expensive > cheap:
        val_signal = "昂貴"
        # 高 ROIC（>30%）的公司給予估值豁免，不扣昂貴分
        if d.get("roic") and d["roic"] > 30:
            return ("合理（壟斷溢價）", f"ROIC={d['roic']:.0f}%，高護城河支撐溢價")
        return (val_signal, detail)
    return ("合理", detail)


# =================================================================
# 5. Helper – safe data extraction
# =================================================================
def safe_get(data, key, default=None):
    try:
        if isinstance(data, dict):
            return data.get(key, default)
        elif isinstance(data, pd.DataFrame):
            if key in data.index and not data.empty:
                return data.loc[key].iloc[0]
            return default
    except:
        return default
    return default

# =================================================================
# 5b. Earnings Call 逐字稿分析
# =================================================================
def fetch_earnings_call_signals(ticker):
    """
    抓取最新 earnings call 的管理層發言，分析說謊/迴避訊號。
    回傳 dict: {
        "vague_word_count": int,      # 模糊詞數量（meaningful, transformative, excited about...）
        "dodge_signals": int,          # 迴避訊號（let me rephrase, moving on, next question...）
        "confidence_score": float,     # 0-10，越高越坦誠
        "raw_excerpt": str             # 原文摘錄供 LLM 用
    }
    """
    vague_words = ["meaningful", "transformative", "excited about", "encouraged by",
                   "we believe", "going forward", "in due course", "as appropriate",
                   "we are confident", "headwinds", "challenging environment"]
    dodge_phrases = ["let me rephrase", "i'll take that offline", "next question",
                     "we don't guide on", "i wouldn't characterize it that way",
                     "that's a great question", "it's complicated"]

    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []

        earnings_text = ""
        for n in news[:20]:
            title = n.get("title", "").lower()
            if any(kw in title for kw in ["earnings", "quarter", "q1", "q2", "q3", "q4", "revenue", "guidance"]):
                earnings_text += n.get("title", "") + " " + (n.get("summary", "") or "")

        if not earnings_text:
            return {"vague_word_count": 0, "dodge_signals": 0, "confidence_score": 5.0, "raw_excerpt": ""}

        text_lower = earnings_text.lower()
        vague_count = sum(text_lower.count(w) for w in vague_words)
        dodge_count = sum(text_lower.count(p) for p in dodge_phrases)

        confidence = max(0, min(10, 8 - vague_count * 0.5 - dodge_count * 1.5))

        return {
            "vague_word_count": vague_count,
            "dodge_signals": dodge_count,
            "confidence_score": round(confidence, 1),
            "raw_excerpt": earnings_text[:500]
        }
    except Exception:
        return {"vague_word_count": 0, "dodge_signals": 0, "confidence_score": 5.0, "raw_excerpt": ""}


# =================================================================
# 5c. Google Trends 搜尋熱度
# =================================================================
def fetch_google_trends_score(ticker, company_name):
    """
    用 pytrends 查詢過去 12 個月的搜尋趨勢。
    回傳 {
        "trend_direction": "上升"/"下降"/"持平"/"無數據"/"未安裝pytrends",
        "trend_score": float 0-10,
        "is_low_attention": bool,   # 近期均值低於歷史中位數 80%（無人問津）
        "current_vs_peak": float,   # 當前熱度 / 歷史峰值，越低越冷門
    }
    """
    _no_data = {"trend_direction": "無數據", "trend_score": 5.0, "is_low_attention": False, "current_vs_peak": 1.0}
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        kw = company_name[:30] if company_name else ticker
        pytrends.build_payload([kw], timeframe='today 12-m')
        df = pytrends.interest_over_time()
        if df.empty or kw not in df.columns:
            return _no_data

        values = df[kw].values
        recent_avg = values[-4:].mean()   # 近 4 週均值
        past = values[:4].mean()           # 最初 4 週均值
        historical_median = np.median(values)
        peak = values.max() if values.max() > 0 else 1
        current_vs_peak = recent_avg / peak

        is_low_attention = recent_avg < historical_median * 0.8  # 比歷史中位數低 20% 以上

        if past == 0:
            return {**_no_data, "is_low_attention": is_low_attention, "current_vs_peak": float(current_vs_peak)}

        change = (recent_avg - past) / past
        if change > 0.2:
            direction, tscore = "上升", 8.0
        elif change < -0.2:
            direction, tscore = "下降", 3.0
        else:
            direction, tscore = "持平", 5.0

        return {
            "trend_direction": direction,
            "trend_score": tscore,
            "is_low_attention": bool(is_low_attention),
            "current_vs_peak": float(current_vs_peak),
        }
    except ImportError:
        return {"trend_direction": "未安裝pytrends", "trend_score": 5.0, "is_low_attention": False, "current_vs_peak": 1.0}
    except Exception:
        return _no_data


def compute_volume_attention_metrics(volumes, source="unknown"):
    """
    Volume_Z = (Current_Volume - Avg_Volume) / Std_Dev_Volume
    volumes[0] = 最新交易日（FMP historical 為新→舊排序）
    """
    if not volumes or len(volumes) < 20:
        return None

    arr = np.array(volumes[:VOLUME_LOOKBACK_DAYS], dtype=float)
    current = arr[0]
    hist = arr[1:]
    avg = hist.mean()
    std = hist.std()
    if std < 1:
        return None

    z_score = (current - avg) / std

    z_delta = 0.0
    if len(arr) >= 21:
        past_current = arr[20]
        past_hist = arr[21:]
        past_std = past_hist.std()
        if past_std >= 1:
            past_z = (past_current - past_hist.mean()) / past_std
            z_delta = z_score - past_z

    surge_z = 0.0
    if len(arr) >= 25:
        recent_5 = arr[:5].mean()
        prior = arr[5:25]
        if prior.std() >= 1:
            surge_z = (recent_5 - prior.mean()) / prior.std()

    if z_score <= VOLUME_Z_PEARL_MAX:
        signal = "潛在珍珠"
    elif z_score >= VOLUME_Z_BREAKOUT_MIN or z_delta >= VOLUME_Z_SURGE_DELTA or surge_z >= VOLUME_Z_BREAKOUT_MIN:
        signal = "共識轉折"
    else:
        signal = "neutral"

    return {
        "volume_z": round(float(z_score), 2),
        "volume_z_delta": round(float(z_delta), 2),
        "volume_surge_z": round(float(surge_z), 2),
        "current_volume": int(current),
        "avg_volume": int(avg),
        "attention_signal": signal,
        "source": source,
    }


def fetch_volume_attention(ticker, fmp_ticker=None, fmp_first=True):
    """FMP historical-price-full 取 volume；失敗時 fallback yfinance。"""
    if not USE_VOLUME_ATTENTION:
        return None

    fmp_ticker = fmp_ticker or ticker
    volumes = []
    source = None

    if fmp_first:
        data = fmp_get(f"historical-price-full/{fmp_ticker}", {"timeseries": VOLUME_LOOKBACK_DAYS})
        if data:
            hist = data.get("historical") if isinstance(data, dict) else data
            if hist and isinstance(hist, list):
                volumes = [h.get("volume") for h in hist if h.get("volume") is not None]
                if len(volumes) >= 20:
                    source = "fmp"

    if len(volumes) < 20:
        try:
            df = yf.Ticker(ticker).history(period="3mo")
            if not df.empty and "Volume" in df.columns:
                vols = df["Volume"].dropna().astype(float).tolist()
                volumes = list(reversed(vols[-VOLUME_LOOKBACK_DAYS:]))
                source = "yfinance"
        except Exception:
            pass

    return compute_volume_attention_metrics(volumes, source or "unknown")


def apply_volume_attention_bonus(d, score, details, golden_zone=False):
    """高 quant + 量能 Z → 珍珠 / 共識轉折加分。"""
    attn = d.get("volume_attention")
    if not USE_VOLUME_ATTENTION or not attn:
        details.setdefault("volume_attention_score", 0)
        details.setdefault("attention_signal", "N/A")
        details.setdefault("volume_z", None)
        details.setdefault("volume_z_delta", None)
        return score, details, golden_zone

    details["volume_z"] = attn.get("volume_z")
    details["volume_z_delta"] = attn.get("volume_z_delta")
    details["volume_surge_z"] = attn.get("volume_surge_z")

    if score < QUANT_CORE_HIGH_MIN:
        details["volume_attention_score"] = 0
        details["attention_signal"] = attn.get("attention_signal", "neutral")
        return score, details, golden_zone

    sig = attn.get("attention_signal", "neutral")
    ticker = d.get("ticker", "?")

    if sig == "潛在珍珠":
        score += SCORE_VOLUME_PEARL
        details["volume_attention_score"] = SCORE_VOLUME_PEARL
        details["attention_signal"] = "潛在珍珠"
        if not golden_zone:
            details["golden_zone"] = True
            golden_zone = True
        logging.info(
            f"[潛在珍珠] {ticker}: quant={score - SCORE_VOLUME_PEARL:.0f}, "
            f"Volume_Z={attn['volume_z']:.2f}（冷門優質，等待被發現）"
        )
    elif sig == "共識轉折":
        score += SCORE_VOLUME_BREAKOUT
        details["volume_attention_score"] = SCORE_VOLUME_BREAKOUT
        details["attention_signal"] = "共識轉折"
        logging.info(
            f"[共識轉折] {ticker}: quant={score - SCORE_VOLUME_BREAKOUT:.0f}, "
            f"Volume_Z={attn['volume_z']:.2f}, ΔZ={attn.get('volume_z_delta', 0):.2f}（量能急升轉折）"
        )
    else:
        details["volume_attention_score"] = 0
        details["attention_signal"] = "neutral"

    return score, details, golden_zone


# =================================================================
# 6. Financial data fetch
# =================================================================
def _fetch_data_inner(ticker, region="US"):
    fmp_ticker = ticker

    # --- 1. Profile（市值、描述、sector、分析師評級）---
    profile_data = fmp_get(f"profile/{fmp_ticker}")
    profile = profile_data[0] if profile_data and len(profile_data) > 0 else {}

    mkt_cap = profile.get("mktCap")
    desc = profile.get("description", "")
    sector = profile.get("sector", "")
    price_now = profile.get("price")

    # --- 2. 財務報表（最新年報）---
    income_data = fmp_get(f"income-statement/{fmp_ticker}", {"limit": 4, "period": "annual"})
    income = income_data[0] if income_data and len(income_data) > 0 else {}

    bs_data = fmp_get(f"balance-sheet-statement/{fmp_ticker}", {"limit": 4, "period": "annual"})
    bs = bs_data[0] if bs_data and len(bs_data) > 0 else {}

    cf_data = fmp_get(f"cash-flow-statement/{fmp_ticker}", {"limit": 4, "period": "annual"})
    cf = cf_data[0] if cf_data and len(cf_data) > 0 else {}

    # --- 3. 關鍵財務數據映射 ---
    total_revenue = income.get("revenue")
    net_income = income.get("netIncome")
    ebit = income.get("operatingIncome")

    tax_provision = income.get("incomeTaxExpense")
    pretax = income.get("incomeBeforeTax")
    if pretax and tax_provision and pretax != 0:
        tax_rate = tax_provision / pretax
    else:
        tax_rate = 0.21

    total_assets = bs.get("totalAssets")
    total_debt = bs.get("totalDebt") or (
        (bs.get("longTermDebt") or 0) + (bs.get("shortTermDebt") or 0)
    )
    cash = bs.get("cashAndCashEquivalents") or bs.get("cash")
    total_equity = bs.get("totalStockholdersEquity") or bs.get("totalEquity")

    op_cf = cf.get("operatingCashFlow") or cf.get("netCashProvidedByOperatingActivities")
    capex = cf.get("capitalExpenditure") or cf.get("investmentsInPropertyPlantAndEquipment")
    if capex is not None and capex > 0:
        capex = -capex

    sbc = cf.get("stockBasedCompensation") or 0
    rnd = income.get("researchAndDevelopmentExpenses") or 0

    employees = profile.get("fullTimeEmployees") or profile.get("employees")
    high52 = profile.get("range", "").split("-")[-1].strip() if profile.get("range") else None
    try:
        high52 = float(high52) if high52 else None
    except Exception:
        high52 = None

    # --- 4. 估值數據 ---
    key_metrics = fmp_get(f"key-metrics/{fmp_ticker}", {"limit": 1, "period": "annual"})
    km = key_metrics[0] if key_metrics and len(key_metrics) > 0 else {}

    pe_ratio = km.get("peRatio") or profile.get("pe")
    pb_ratio = km.get("pbRatio") or profile.get("priceToBook")
    ps_ratio = km.get("priceToSalesRatio")
    peg_ratio = km.get("pegRatio")
    ev_ebitda = km.get("enterpriseValueOverEBITDA")

    # --- 5. ROIC 趨勢（多年）---
    roic_history = []
    if income_data and bs_data and cf_data:
        for i in range(min(len(income_data), len(bs_data), 4)):
            try:
                _ebit = income_data[i].get("operatingIncome")
                _eq = bs_data[i].get("totalStockholdersEquity") or bs_data[i].get("totalEquity")
                _debt = bs_data[i].get("totalDebt", 0) or 0
                _cash = bs_data[i].get("cashAndCashEquivalents", 0) or 0
                if all(v is not None for v in [_ebit, _eq]):
                    _ic = (_eq or 0) + _debt - _cash
                    if _ic > 0:
                        roic_history.append((_ebit * 0.79) / _ic * 100)
                    else:
                        roic_history.append(None)
            except Exception:
                pass

    # --- 6. 新聞（FMP stock news endpoint）---
    news_data = fmp_get(f"stock_news", {"tickers": fmp_ticker, "limit": 10}) or []
    news = [{"title": n.get("title", ""), "summary": n.get("text", "")} for n in news_data]

    # --- 7. 分析師評級 ---
    rating_data = fmp_get(f"analyst-stock-recommendations/{fmp_ticker}", {"limit": 1})
    rating = rating_data[0] if rating_data and len(rating_data) > 0 else {}
    analyst_rating = rating.get("analystRatingsStrongBuy", "")
    analyst_target = None
    analyst_count = 0

    price_target_data = fmp_get(f"price-target-consensus/{fmp_ticker}")
    if price_target_data and len(price_target_data) > 0:
        analyst_target = price_target_data[0].get("targetConsensus")

    # --- 8. Earnings signals ---
    earnings_signals = fetch_earnings_call_signals(ticker) if USE_EARNINGS_SIGNALS else {}

    return {
        "ticker": ticker,
        "region": region,
        "sector": sector,
        "mkt_cap": mkt_cap,
        "total_revenue": total_revenue,
        "net_income": net_income,
        "ebit": ebit,
        "tax_rate": tax_rate,
        "total_assets": total_assets,
        "total_debt": total_debt,
        "cash": cash,
        "total_equity": total_equity,
        "op_cf": op_cf,
        "capex": capex,
        "sbc": sbc,
        "employees": employees,
        "rnd": rnd,
        "price": price_now,
        "high52": high52,
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "ps_ratio": ps_ratio,
        "peg_ratio": peg_ratio,
        "ev_ebitda": ev_ebitda,
        "news": news,
        "desc": desc,
        "roic_history": roic_history,
        "analyst_rating": analyst_rating,
        "analyst_target": analyst_target,
        "analyst_count": analyst_count,
        "earnings_signals": earnings_signals,
        "trends": fetch_google_trends_score(ticker, profile.get("companyName", "")) if USE_GOOGLE_TRENDS else {"trend_direction": "未啟用", "trend_score": 5.0, "is_low_attention": False, "current_vs_peak": 1.0},
        "volume_attention": fetch_volume_attention(ticker, fmp_ticker),
    }


if HAS_TENACITY:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def fetch_data(ticker, region="US"):
        return _fetch_data_inner(ticker, region)
else:
    def fetch_data(ticker, region="US"):
        for attempt in range(3):
            try:
                return _fetch_data_inner(ticker, region)
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt * 2
                logging.warning(f"Fetch retry {attempt+1}/3 for {ticker}, wait {wait}s: {e}")
                time.sleep(wait)


# =================================================================
# 7. Quantitative metrics & scoring
# =================================================================
def compute_roic_momentum(roic_history):
    """
    Returns (momentum_score, trend_label).
    roic_history: list of ROIC values, newest first.
    - 連續 3 年上升 (oldest→newest) → score=10, "加速成長"
    - 近 2 年上升但第 3 年不確定   → score=5,  "穩定改善"
    - 持平或下滑                   → score=0,  "趨勢疲軟"
    - 資料不足（< 2 筆）           → score=0,  "資料不足"
    """
    if not roic_history or len(roic_history) < 2:
        return 0, "資料不足"
    # 序列為 newest-first，翻轉成 oldest-first 方便比較
    seq = list(reversed(roic_history))
    if len(seq) >= 3 and seq[-1] > seq[-2] > seq[-3]:
        return 10, "加速成長"
    if seq[-1] > seq[-2]:
        return 5, "穩定改善"
    return 0, "趨勢疲軟"


def compute_roic(d):
    if not (d["ebit"] and d["tax_rate"] is not None and d["total_equity"] is not None and d["total_debt"] is not None and d["cash"] is not None):
        return None
    nopat = d["ebit"] * (1 - d["tax_rate"])
    ic = d["total_equity"] + d["total_debt"] - d["cash"]
    if ic <= 0:
        return None
    if d["rnd"] and d["rnd"] > 0:
        cap_rnd = d["rnd"] * 0.8
        nopat += cap_rnd * 0.2
        ic += cap_rnd
    return (nopat / ic) * 100

def dupont_drive(d):
    if not (d["net_income"] and d["total_revenue"] and d["total_assets"]):
        return None
    total_equity = d["total_equity"]
    total_assets = d["total_assets"]

    # 負股東權益：可能是庫藏股回購導致（如 AAPL、MCD、HD）
    if total_equity is not None and total_equity <= 0:
        op_cf = d.get("op_cf") or 0
        mkt_cap = d.get("mkt_cap") or 1
        cf_yield = (op_cf / mkt_cap) * 100 if mkt_cap > 0 else 0
        roic_hist = d.get("roic_history", [])
        latest_roic = roic_hist[0] if roic_hist else None

        if cf_yield > 3 and latest_roic is not None and latest_roic > 15:
            logging.info(
                f"[庫藏股複利機器] {d.get('ticker', '?')}: 負股東權益但 "
                f"CF Yield={cf_yield:.1f}%, ROIC={latest_roic:.1f}%"
            )
            return "buyback_compounder"
        return "negative_equity_risk"

    if total_equity is None or total_equity == 0:
        return None

    leverage = total_assets / total_equity if total_equity > 0 else 999
    financial_sectors = ("Financial Services", "Banks", "Insurance", "Real Estate")
    sector = d.get("sector", "")
    if leverage > 5:
        if sector in financial_sectors:
            return "balanced"  # 金融股高槓桿是正常商業模式
        return "leverage"  # 其他行業高槓桿才是風險
    net_margin = d["net_income"] / d["total_revenue"]
    turnover = d["total_revenue"] / total_assets
    if net_margin and turnover:
        if net_margin > 0.15 and turnover < 1.0:
            return "margin"
        elif turnover > 1.2 and net_margin > 0.05:
            return "turnover"
        else:
            return "balanced"
    return "unknown"

def footprint_ok(d):
    if d["total_revenue"] and d["employees"] and d["employees"] > 0:
        rev_per_emp = d["total_revenue"] / d["employees"]
        if rev_per_emp < 100000:
            return False
    return True

def excess_fcf_yield(d):
    if not (d["op_cf"] and d["capex"] is not None and d["mkt_cap"] and d["mkt_cap"] > 0):
        return None
    # SBC 調整：股票薪酬是真實的股東成本，需從 FCF 中扣除
    sbc = d.get("sbc") or 0
    # op_cf 中已含 SBC（非現金項目加回），需扣除以還原真實股東現金流
    fcf = d["op_cf"] + d["capex"] - abs(sbc)
    fcf_yield = (fcf / d["mkt_cap"]) * 100
    return fcf_yield - RISK_FREE_RATE

def quant_score(d):
    score = 0
    details = {}

    roic = compute_roic(d)
    if roic is not None and roic > 15:
        score += SCORE_ROIC
        details["roic_score"] = SCORE_ROIC
    elif roic is not None and roic > 10:
        score += SCORE_ROIC * 0.5
        details["roic_score"] = SCORE_ROIC * 0.5
    else:
        details["roic_score"] = 0

    drive = dupont_drive(d)
    if drive in ("margin", "turnover", "balanced", "buyback_compounder"):
        score += SCORE_DUPONT
        details["dupont_score"] = SCORE_DUPONT
    else:
        details["dupont_score"] = 0
    details["dupont_label"] = drive

    if footprint_ok(d):
        score += SCORE_FOOTPRINT
        details["footprint_score"] = SCORE_FOOTPRINT
    else:
        details["footprint_score"] = 0

    ex_yield = excess_fcf_yield(d)
    grey_zone = False
    if ex_yield is not None and ex_yield >= 4:
        score += SCORE_FCF
        details["fcf_score"] = SCORE_FCF
    elif ex_yield is not None and ex_yield >= 2:
        score += SCORE_FCF * 0.5
        details["fcf_score"] = SCORE_FCF * 0.5
        grey_zone = True
    else:
        details["fcf_score"] = 0
    details["grey_zone"] = grey_zone
    details["sbc_adjusted"] = bool(d.get("sbc") and d.get("sbc") != 0)

    momentum_score, trend_label = compute_roic_momentum(d.get("roic_history", []))
    score += momentum_score
    details["roic_momentum_score"] = momentum_score
    details["roic_trend"] = trend_label

    if USE_EARNINGS_SIGNALS and d.get("earnings_signals"):
        conf = d["earnings_signals"].get("confidence_score", 5.0)
        if conf >= 7:
            score += SCORE_EARNINGS_CONFIDENCE
            details["earnings_score"] = SCORE_EARNINGS_CONFIDENCE
        elif conf >= 5:
            score += SCORE_EARNINGS_CONFIDENCE * 0.5
            details["earnings_score"] = SCORE_EARNINGS_CONFIDENCE * 0.5
        else:
            details["earnings_score"] = 0
        details["earnings_confidence"] = conf

    # 逆向擁擠度指標：無人問津的優質股
    trends = d.get("trends", {})
    trend_dir = trends.get("trend_direction", "未啟用")
    is_low_attention = trends.get("is_low_attention", False)
    roic_for_contrarian = roic or 0
    golden_zone = False
    if USE_GOOGLE_TRENDS and is_low_attention and roic_for_contrarian > 15:
        score += SCORE_CONTRARIAN
        details["contrarian_bonus"] = SCORE_CONTRARIAN
        details["golden_zone"] = True
        golden_zone = True
        logging.info(
            f"[黃金擊球區] {d['ticker']}: ROIC={roic_for_contrarian:.1f}%, "
            f"熱度低迷（current_vs_peak={trends.get('current_vs_peak', 0):.2f}），無人問津的優質股"
        )
    else:
        details["contrarian_bonus"] = 0
        details["golden_zone"] = False

    score, details, golden_zone = apply_volume_attention_bonus(d, score, details, golden_zone)
    details["golden_zone"] = golden_zone

    # 把本次計算的 roic 暫存到 d，讓 compute_valuation_signal 可讀取壟斷溢價豁免
    d["roic"] = roic
    val_signal, val_reason = compute_valuation_signal(d)
    details["valuation"] = val_signal
    details["valuation_reason"] = val_reason
    if val_signal == "便宜":
        score += 5
    elif val_signal == "昂貴":
        score -= 5

    details["total_quant"] = score
    return score, details, roic, ex_yield

# =================================================================
# 7b. 成長爆發模式量化評分
# =================================================================
def quant_score_growth(d):
    """
    成長股評分邏輯：獎勵高速擴張，容忍短期虧損。
    滿分 60 分（與複利模式相同，但權重不同）
    """
    score = 0
    details = {}

    # 1. 營收成長率代理（25分）：PS ratio 作為估值/成長代理
    ps = d.get("ps_ratio")
    if ps is not None:
        if ps < 5:
            score += 25
            details["rev_score"] = 25
        elif ps < 15:
            score += 15
            details["rev_score"] = 15
        elif ps < 30:
            score += 8
            details["rev_score"] = 8
        else:
            details["rev_score"] = 0  # 泡沫估值
    else:
        details["rev_score"] = 0

    # 2. R&D 強度（15分）：是否在建護城河
    rnd = d.get("rnd")
    rev_val = d.get("total_revenue")
    if rnd and rev_val and rev_val > 0:
        rnd_intensity = rnd / rev_val
        if rnd_intensity > 0.15:
            score += 15
            details["rnd_score"] = 15
        elif rnd_intensity > 0.05:
            score += 8
            details["rnd_score"] = 8
        else:
            details["rnd_score"] = 0
    else:
        details["rnd_score"] = 0

    # 3. 市場規模代理（10分）：市值是否仍在成長期
    mkt_cap = d.get("mkt_cap")
    if mkt_cap:
        if mkt_cap < 10e9:       # 小於100億（小型成長股）
            score += 10
            details["size_score"] = 10
        elif mkt_cap < 100e9:    # 100~1000億（中型）
            score += 6
            details["size_score"] = 6
        else:
            score += 2
            details["size_score"] = 2  # 超大型成長已放緩
    else:
        details["size_score"] = 0

    # 4. 毛利率代理（10分）：高淨利率=產品壁壘（成長股容忍虧損但高毛利加分）
    net_income = d.get("net_income")
    rev_val = d.get("total_revenue")
    if net_income and rev_val and rev_val > 0:
        margin = net_income / rev_val
        if margin > 0.20:
            score += 10
            details["margin_score"] = 10
        elif margin > 0.05:
            score += 5
            details["margin_score"] = 5
        else:
            details["margin_score"] = 0  # 虧損可以接受但不加分
    else:
        details["margin_score"] = 0

    # 5. Earnings Confidence（5分）：管理層誠信
    if d.get("earnings_signals"):
        conf = d["earnings_signals"].get("confidence_score", 5.0)
        if conf >= 7:
            score += 5
            details["earnings_score"] = 5
        elif conf >= 5:
            score += 2
            details["earnings_score"] = 2
        else:
            details["earnings_score"] = 0
        details["earnings_confidence"] = conf
    else:
        details["earnings_score"] = 0
        details["earnings_confidence"] = 5.0

    score, details, golden_zone = apply_volume_attention_bonus(d, score, details, golden_zone=False)
    details["golden_zone"] = golden_zone

    roic = compute_roic(d)
    ex_yield = excess_fcf_yield(d)

    # 成長模式沒有 FCF 懲罰（容忍負現金流）
    details["total_quant"] = score
    details["grey_zone"] = False
    details["roic_trend"] = "N/A"
    details["valuation"] = "成長模式"
    details["valuation_reason"] = "成長評分不依賴估值乘數"
    return score, details, roic, ex_yield


# =================================================================
# 8. LLM functions (kill switches & qualitative scoring)
# =================================================================
def extract_news_headlines(news_data, max_items=5):
    keywords = ["earnings", "CEO", "acquisition", "layoff", "regulatory", "lawsuit", "investment", "restructure"]
    filtered = []
    if not news_data:
        return "無重大動向新聞"
    for n in news_data[:10]:
        title = n.get("title", "")
        summary = n.get("summary", "") or n.get("description", "")
        combined = (title + " " + summary).lower()
        if any(kw in combined for kw in keywords):
            filtered.append(f"- {title} | {summary[:150]}")
    return "\n".join(filtered[:max_items]) if filtered else "無重大動向新聞"

def llm_kill_switches(d):
    if not USE_LLM or not client or not d.get("desc"):
        return False, ""
    news_text = extract_news_headlines(d.get("news", []))
    prompt = f"""根據以下資訊，判斷此公司是否觸發任何一票否決閘門：
1. 誠信閘門：業績會/新聞中是否出現大量模糊詞或重大會計不一致？
2. 靈魂流失閘門：核心高管或技術靈魂人物大規模流失？
3. 監管與地緣暴擊：營收嚴重依賴單一高政治風險國家且無本地產能，或面臨重大訴訟/制裁。

公司：{d['ticker']}
業務：{d['desc'][:300]}
最新新聞：{news_text}

輸出 JSON：{{"kill_triggered": true/false, "reason": "簡述"}}"""
    try:
        resp = client.chat.completions.create(
            model=get_model_for_sector(d.get("sector", "")),
            messages=[{"role": "system", "content": "你是嚴格的投資審計師，負責輸出JSON格式結果。"},
                      {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            res = json.loads(match.group())
        else:
            res = {"kill_triggered": False, "reason": "JSON parse failed"}
        return res.get("kill_triggered", False), res.get("reason", "")
    except Exception as e:
        logging.warning(f"LLM kill switch failed for {d['ticker']}: {e}")
        return False, "LLM error"

def llm_qualitative_score(d):
    if not USE_LLM or not client:
        return {"grit": 0, "culture": 0, "catalyst": 0}
    news_text = extract_news_headlines(d.get("news", []), max_items=8)
    prompt = f"""根據以下公司資訊及近期新聞，嚴格按標準評分（僅能給出 0、10 或最高分 16.6/16.8）：

- 毅力分數（16.6滿分）：逆週期資本開支（行業谷底擴張）→ 16.6；穩定 R&D 投入 → 10；跟風炒作 → 0。
- 文化分數（16.6滿分）：CEO 好評 >85% 且品牌具備文化模因壟斷 → 16.6；平穩 → 10；壓榨員工或罷工醜聞 → 0。
- 催化劑分數（16.8滿分）：機構持股 <20% 且內部人增持，或頂級家辦建倉 → 16.8；機構低但無動靜 → 10；擁擠且無邊際變化 → 0。

公司：{d['ticker']}
業務：{d['desc'][:200]}
新聞摘要：{news_text}
財務：FCF Yield={d.get('ex_fcf_yield', 'N/A')}, ROIC={d.get('roic', 'N/A')}, 員工數={d.get('employees', 'N/A')}

輸出純JSON，不要其他文字：{{"grit": <分數>, "grit_reason": "...", "culture": <分數>, "culture_reason": "...", "catalyst": <分數>, "catalyst_reason": "..."}}"""
    try:
        resp = client.chat.completions.create(
            model=get_model_for_sector(d.get("sector", "")),
            messages=[{"role": "system", "content": "你是結合巴菲特、馬斯克、塔雷伯的投資評分官，負責輸出JSON格式結果。"},
                      {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        parsed = _parse_llm_json(content)
        if parsed:
            parsed.setdefault("debate_summary", "")
            parsed.setdefault("investment_thesis", "")
            return parsed
        return {"grit": 0, "culture": 0, "catalyst": 0, "debate_summary": "", "investment_thesis": ""}
    except Exception as e:
        logging.warning(f"LLM qual failed for {d['ticker']}: {e}")
        return {"grit": 0, "culture": 0, "catalyst": 0}

# =================================================================
# 9. API 成本追蹤器
# =================================================================
_cost_tracker: dict = {
    "total_tokens":   0,
    "total_cost_usd": 0.0,
    "stock_count":    0,
}
_cost_lock = threading.Lock()
_checkpoint_lock = threading.Lock()


def _update_cost_tracker(ticker: str, tokens_used: int, log_interval: int = 50) -> None:
    """每支股票記錄預估 token 用量；每 log_interval 支印一次累計費用。執行緒安全。"""
    est_input  = int(tokens_used * 0.6)
    est_output = int(tokens_used * 0.4)
    est_cost   = (est_input  / 1000 * DS_COST_INPUT_PER_1K +
                  est_output / 1000 * DS_COST_OUTPUT_PER_1K)

    with _cost_lock:
        _cost_tracker["total_tokens"]   += tokens_used
        _cost_tracker["total_cost_usd"] += est_cost
        _cost_tracker["stock_count"]    += 1
        tokens_total = _cost_tracker["total_tokens"]
        cost_total   = _cost_tracker["total_cost_usd"]
        count_total  = _cost_tracker["stock_count"]

    logging.info(
        f"[COST] {ticker}: ~{tokens_used:,} tokens, ~${est_cost:.5f} | "
        f"累計: {tokens_total:,} tokens, ${cost_total:.4f}"
    )
    if count_total % log_interval == 0:
        avg = cost_total / count_total
        logging.info(
            f"[COST SUMMARY ✦] 已處理 {count_total} 支股票 | "
            f"累計 tokens: {tokens_total:,} | "
            f"累計費用: ${cost_total:.4f} | "
            f"平均每股: ${avg:.5f}"
        )


# =================================================================
# 10. SEC RAG – 美股 10-K 深度檢索（MD&A 段落檢索）
# =================================================================
SEC_RAG_TOP_CHUNKS = 8
SEC_MDA_KEYWORDS = [
    "management", "outlook", "guidance", "liquidity", "capital allocation",
    "restructuring", "impairment", "acquisition", "divestiture", "capex",
    "share repurchase", "cost reduction", "margin", "demand", "competition",
    "regulatory", "strategy", "CEO", "CFO", "turnaround", "headwind", "tailwind",
]
SEC_LATENT_KEYWORDS = [
    "new product", "pipeline", "intellectual property", "patent", "contract",
    "customer concentration", "supplier", "geographic expansion", "workforce",
    "insider", "related party", "goodwill", "intangible", "write-down",
]


def _strip_html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_sec_item_section(text: str, item_label: str, next_items: list) -> str:
    """Extract SEC filing section between Item X and next Item Y (case-insensitive)."""
    start_pat = rf"ITEM\s*{re.escape(item_label)}\s*[\.:\-]"
    m = re.search(start_pat, text, re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    end = len(text)
    for nxt in next_items:
        nm = re.search(rf"ITEM\s*{re.escape(nxt)}\s*[\.:\-]", text[start:], re.IGNORECASE)
        if nm:
            end = min(end, start + nm.start())
    section = text[start:end].strip()
    return section[:120000]


def _chunk_paragraphs(text: str, min_len=120, max_len=1200):
    """Split long SEC text into paragraph-ish chunks."""
    parts = re.split(r"(?<=[\.!?])\s+(?=[A-Z0-9\"'])", text)
    chunks, buf = [], ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) + len(part) < max_len:
            buf = f"{buf} {part}".strip()
        else:
            if len(buf) >= min_len:
                chunks.append(buf)
            buf = part
    if len(buf) >= min_len:
        chunks.append(buf)
    if not chunks and text:
        for i in range(0, len(text), max_len):
            chunks.append(text[i:i + max_len])
    return chunks


def _score_sec_chunk(chunk: str, keywords: list) -> float:
    """Lightweight lexical retrieval (TF-style); no embedding API required."""
    lower = chunk.lower()
    words = re.findall(r"[a-zA-Z]{4,}", lower)
    if not words:
        return 0.0
    score = 0.0
    for kw in keywords:
        kw_l = kw.lower()
        score += lower.count(kw_l) * (2.0 if " " in kw else 1.0)
    # Prefer chunks with numbers (often guidance / YoY deltas)
    score += min(len(re.findall(r"\b\d{1,3}(?:\.\d+)?%|\$\d", chunk)), 5) * 0.5
    return score / max(len(words) ** 0.5, 1.0)


def _retrieve_sec_chunks(section_text: str, keywords: list, top_k=SEC_RAG_TOP_CHUNKS) -> list:
    chunks = _chunk_paragraphs(section_text)
    ranked = sorted(
        ((_score_sec_chunk(c, keywords), c) for c in chunks),
        key=lambda x: x[0],
        reverse=True,
    )
    return [c for s, c in ranked if s > 0][:top_k]


def fetch_sec_context(ticker: str) -> str:
    """
    從 SEC EDGAR 搜尋最新 10-K，重點檢索 MD&A (Item 7) 與 Risk Factors (Item 1A)
    的高相關段落（lexical ranking），而非 raw_text[:8000]。
    """
    headers = {"User-Agent": "InvestBot research@example.com"}
    try:
        search_url = (
            "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            "&dateRange=custom&startdt=2024-01-01&forms=10-K"
        ).format(ticker=ticker)
        resp = _http_get_with_backoff(search_url, headers=headers, timeout=15)
        if resp is None:
            return ""
        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            return ""

        doc_url = None
        for hit in hits[:5]:
            src = hit.get("_source", {})
            entity = src.get("entity_name", "").upper()
            if ticker.upper() in entity or entity in ticker.upper():
                doc_url = src.get("file_url") or src.get("period_of_report")
                break
        if not doc_url:
            doc_url = hits[0].get("_source", {}).get("file_url") or ""

        if doc_url and not doc_url.startswith("http"):
            doc_url = "https://www.sec.gov" + doc_url
        if not doc_url:
            return ""

        doc_resp = _http_get_with_backoff(doc_url, headers=headers, timeout=25)
        if doc_resp is None:
            return ""
        raw_text = _strip_html_to_text(doc_resp.text)

        mda = _extract_sec_item_section(raw_text, "7", ["7A", "8", "9"])
        risks = _extract_sec_item_section(raw_text, "1A", ["1B", "2"])
        mda_hits = _retrieve_sec_chunks(mda, SEC_MDA_KEYWORDS)
        latent_hits = _retrieve_sec_chunks(mda + " " + risks, SEC_LATENT_KEYWORDS, top_k=4)
        risk_hits = _retrieve_sec_chunks(risks, SEC_MDA_KEYWORDS[:12], top_k=3)

        if not mda_hits and not risk_hits:
            # fallback: legacy excerpt
            fallback = raw_text[:4000]
            return f"=== SEC 10-K fallback excerpt ({ticker}) ===\n{fallback}"

        parts = [f"=== SEC 10-K 段落檢索 ({ticker}) ==="]
        parts.append("\n[MD&A — 管理層討論與分析 · 高相關段落]")
        for i, ch in enumerate(mda_hits, 1):
            parts.append(f"\n({i}) {ch[:900]}")
        if latent_hits:
            parts.append("\n[隱性資產 / 結構變化線索]")
            for i, ch in enumerate(latent_hits, 1):
                parts.append(f"\n(L{i}) {ch[:700]}")
        if risk_hits:
            parts.append("\n[Risk Factors — 新浮現風險]")
            for i, ch in enumerate(risk_hits, 1):
                parts.append(f"\n(R{i}) {ch[:700]}")
        return "\n".join(parts)[:12000]
    except Exception as e:
        logging.debug(f"SEC RAG failed for {ticker}: {e}")
        return ""

# =================================================================
# 10. 動態工具調用 – Function Calling
# =================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_insider_trading",
            "description": "查詢內部人（高管/董事）最近 6 個月的持股增減，用於評估 catalyst 分數",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "股票代碼，例如 AAPL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_institutional_ownership",
            "description": "查詢機構持股比例，判斷是否為低機構覆蓋率的潛力股",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_short_interest",
            "description": "查詢空頭比率（Short Interest %），高空頭可能暗示市場疑慮",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"}
                },
                "required": ["ticker"]
            }
        }
    }
]


def get_insider_trading(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        transactions = stock.insider_transactions
        if transactions is None or transactions.empty:
            return {"net_shares": 0, "direction": "持平", "transactions": 0}
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=6)
        # 相容不同欄位名稱
        date_col = next((c for c in transactions.columns if "date" in c.lower()), None)
        share_col = next((c for c in transactions.columns if "share" in c.lower() or "shares" in c.lower()), None)
        if date_col is None or share_col is None:
            return {"net_shares": 0, "direction": "持平", "transactions": 0}
        recent = transactions[pd.to_datetime(transactions[date_col], errors="coerce") >= cutoff]
        net = int(recent[share_col].fillna(0).sum())
        direction = "買入" if net > 0 else ("賣出" if net < 0 else "持平")
        return {"net_shares": net, "direction": direction, "transactions": len(recent)}
    except Exception as e:
        logging.debug(f"get_insider_trading({ticker}): {e}")
        return {"net_shares": 0, "direction": "持平", "transactions": 0}


def get_institutional_ownership(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        holders = stock.institutional_holders
        if holders is None or holders.empty:
            return {"institutional_pct": 0.0, "top_holders": []}
        pct_col = next((c for c in holders.columns if "pct" in c.lower() or "%" in c), None)
        name_col = next((c for c in holders.columns if "holder" in c.lower() or "name" in c.lower()), None)
        pct = float(holders[pct_col].sum() * 100) if pct_col else 0.0
        top = holders[name_col].head(5).tolist() if name_col else []
        return {"institutional_pct": round(pct, 2), "top_holders": top}
    except Exception as e:
        logging.debug(f"get_institutional_ownership({ticker}): {e}")
        return {"institutional_pct": 0.0, "top_holders": []}


def get_short_interest(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        short_pct = info.get("shortPercentOfFloat") or 0.0
        return {"short_pct": round(float(short_pct) * 100, 2)}
    except Exception as e:
        logging.debug(f"get_short_interest({ticker}): {e}")
        return {"short_pct": 0.0}


def dispatch_tool(name: str, args: dict) -> dict:
    if name == "get_insider_trading":
        return get_insider_trading(args.get("ticker", ""))
    elif name == "get_institutional_ownership":
        return get_institutional_ownership(args.get("ticker", ""))
    elif name == "get_short_interest":
        return get_short_interest(args.get("ticker", ""))
    return {"error": f"unknown tool: {name}"}


def llm_with_tools(ticker: str, messages: list, tools: list, max_rounds: int = TOOL_MAX_ROUNDS, model: str = None) -> str:
    """
    動態推理迴圈：讓 LLM 在生成回答前自主呼叫工具。
    若 DeepSeek 不支援 tool_calling 則自動降級為普通模式。
    max_rounds 預設使用 TOOL_MAX_ROUNDS（成本控制=2）。
    """
    global USE_TOOL_CALLING
    _model = model or LLM_MODEL
    for _ in range(max_rounds):
        try:
            resp = client.chat.completions.create(
                model=_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=400,   # 成本控制
            )
        except Exception as e:
            err_str = str(e).lower()
            if "tool" in err_str or "function" in err_str or "not support" in err_str:
                logging.warning(f"Tool calling not supported by {_model}, falling back to plain mode.")
                USE_TOOL_CALLING = False
                # 降級：重新不帶 tools 呼叫
                try:
                    resp = client.chat.completions.create(
                        model=_model,
                        messages=messages,
                        temperature=0,
                        max_tokens=400,   # 成本控制
                    )
                    return resp.choices[0].message.content or ""
                except Exception as e2:
                    logging.warning(f"Fallback LLM call failed: {e2}")
                    return ""
            raise

        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})
            for call in msg.tool_calls:
                try:
                    result = dispatch_tool(call.function.name, json.loads(call.function.arguments))
                except Exception:
                    result = {"error": "tool execution failed"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        else:
            return msg.content or ""
    return msg.content or "" if msg else ""

# =================================================================
# 10b. LLM JSON 解析（partial rerun 共用）
# =================================================================
def _parse_llm_json(content: str):
    """從 LLM 回傳文字擷取 JSON object；失敗回傳 None。"""
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _normalize_qual_result(result: dict) -> dict:
    """確保 grit/culture/catalyst 為有效數值。"""
    out = dict(result)
    for key in ("grit", "culture", "catalyst"):
        try:
            out[key] = float(out.get(key, 0) or 0)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def build_debate_json(qual: dict, llm_mode: str = "debate", parse_ok: bool = True) -> str:
    """將辯論/評分邏輯摘要序列化為 JSON 字串，寫入 CSV 供勝率回測。"""
    payload = {
        "llm_mode": llm_mode,
        "parse_ok": bool(parse_ok),
        "risk_free_rate": RISK_FREE_RATE,
        "debate_summary": qual.get("debate_summary", ""),
        "investment_thesis": qual.get("investment_thesis", ""),
        "scores": {
            "grit": qual.get("grit"),
            "culture": qual.get("culture"),
            "catalyst": qual.get("catalyst"),
        },
        "reasons": {
            "grit": qual.get("grit_reason", ""),
            "culture": qual.get("culture_reason", ""),
            "catalyst": qual.get("catalyst_reason", ""),
        },
        "bull_points": qual.get("bull_points"),
        "bear_rebuttals": qual.get("bear_rebuttals"),
        "bull_scores": qual.get("bull_scores"),
        "bear_scores": qual.get("bear_scores"),
    }
    return json.dumps(payload, ensure_ascii=False)


def _qual_parse_ok(qual: dict) -> bool:
    summary = str(qual.get("debate_summary") or "")
    return "JSON解析失敗" not in summary and not summary.startswith("error:")


def _judge_tool_context(ticker: str) -> str:
    """預查 yfinance 工具數據，嵌入 Judge prompt（取代 tool calling）。"""
    if not JUDGE_PREFETCH_TOOLS:
        return ""
    try:
        ctx = {
            "insider_6m": get_insider_trading(ticker),
            "institutional": get_institutional_ownership(ticker),
            "short_interest": get_short_interest(ticker),
        }
        return f"\n補充數據（供催化劑評分）：{json.dumps(ctx, ensure_ascii=False)}\n"
    except Exception as e:
        logging.debug(f"judge tool prefetch for {ticker}: {e}")
        return ""


def _call_judge_llm(ticker: str, judge_messages: list, model: str):
    """Judge 專用：強制 JSON mode，不用 tool calling，最多 3 次。"""
    messages = list(judge_messages)
    for attempt in range(3):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        result = _parse_llm_json(content)
        if result and all(k in result for k in ("grit", "culture", "catalyst")):
            return _normalize_qual_result(result)
        if attempt < 2:
            messages = messages + [
                {"role": "assistant", "content": content or "{}"},
                {
                    "role": "user",
                    "content": (
                        "格式錯誤。請只輸出單一 JSON object，"
                        "必含 grit, culture, catalyst, debate_summary, investment_thesis；"
                        "grit/culture/catalyst 必須是數字。"
                    ),
                },
            ]
            logging.info(f"Judge JSON retry {attempt + 2}/3 for {ticker}")
    return None


def needs_llm_rerun(row) -> bool:
    """判斷 CSV 該列是否需 partial rerun（保留原 quant_score）。"""
    if str(row.get("tier", "")) == "C (Kill)":
        return False
    parse_ok = row.get("llm_parse_ok")
    if parse_ok is False or str(parse_ok).lower() == "false":
        return True
    summary = str(row.get("debate_summary") or "")
    thesis = str(row.get("investment_thesis") or "")
    if "JSON解析失敗" in summary or "json parse failed" in summary.lower():
        return True
    if thesis == "資料不足，無法判斷" and float(row.get("qual_score") or 0) <= 30:
        return True
    if summary.startswith("error:"):
        return True
    return False


def _tier_from_score(total_score: float, grey_zone: bool = False, total_qual: float = 0) -> str:
    score = total_score
    if grey_zone and total_qual >= 40 and score < 50:
        score = 50
    if score >= S_TIER_MIN:
        return "S"
    if score >= A_TIER_MIN:
        return "A"
    if score >= B_TIER_MIN:
        return "B"
    return "C"


def fetch_llm_context_light(ticker: str, region: str, row: dict) -> dict:
    """
    Partial rerun 專用：不呼叫 FMP，量化數字沿用 CSV，僅用 yfinance 補描述/新聞（可選）。
    """
    d = {
        "ticker": ticker,
        "region": region,
        "roic": row.get("roic"),
        "ex_fcf_yield": row.get("ex_fcf_yield"),
        "desc": (
            f"{ticker} ({region}). "
            f"Quant={row.get('quant_score')}, ROIC={row.get('roic')}, "
            f"valuation={row.get('valuation')} ({row.get('valuation_reason')}). "
            f"Trend={row.get('roic_trend', 'N/A')}."
        ),
        "news": [],
        "employees": None,
        "sector": "",
        "sec_context": "",
        "analyst_rating": "",
        "analyst_target": None,
        "analyst_count": 0,
        "screen_mode": row.get("mode", "compounder"),
        "earnings_signals": (
            fetch_earnings_call_signals(ticker) if USE_EARNINGS_SIGNALS else {}
        ),
        "trends": {
            "trend_direction": "未啟用",
            "trend_score": 5.0,
            "is_low_attention": False,
            "current_vs_peak": 1.0,
        },
    }
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        summary = info.get("longBusinessSummary") or info.get("shortName")
        if summary:
            d["desc"] = summary
        if info.get("sector"):
            d["sector"] = info["sector"]
        if info.get("fullTimeEmployees"):
            d["employees"] = info["fullTimeEmployees"]
        d["news"] = [
            {
                "title": n.get("title", ""),
                "summary": n.get("summary") or n.get("description", ""),
            }
            for n in (stock.news or [])[:10]
        ]
    except Exception as e:
        logging.debug(f"yfinance light context for {ticker}: {e}")
    return d


def rerun_llm_for_row(row: dict, mode: str, use_fmp: bool = False) -> dict:
    """
    保留 CSV 內 quant 欄位，只重跑 LLM（debate / single），更新 qual 與 tier。
    預設 use_fmp=False：不重打 FMP（避免 403 / 超額）。
    """
    ticker = row["ticker"]
    region = row.get("region", "US")
    q_score = float(row["quant_score"])
    debate_threshold = DEBATE_MIN_QSCORE if mode == "compounder" else 10

    if use_fmp:
        d = fetch_data(ticker, region)
        time.sleep(0.5)
        roic = row.get("roic")
        if roic is not None and not (isinstance(roic, float) and np.isnan(roic)):
            d["roic"] = float(roic)
        ex_yield = row.get("ex_fcf_yield")
        if ex_yield is not None and not (isinstance(ex_yield, float) and np.isnan(ex_yield)):
            d["ex_fcf_yield"] = float(ex_yield)
        if mode == "compounder" and USE_SEC_RAG and region == "US" and q_score >= SEC_RAG_MIN_QSCORE:
            d["sec_context"] = fetch_sec_context(ticker)
    else:
        d = fetch_llm_context_light(ticker, region, row)
        if mode == "compounder" and USE_SEC_RAG and region == "US" and q_score >= SEC_RAG_MIN_QSCORE:
            sec_ctx = fetch_sec_context(ticker)
            if sec_ctx:
                d["sec_context"] = sec_ctx
        time.sleep(0.3)

    d["screen_mode"] = mode

    if USE_DEBATE and q_score >= debate_threshold:
        logging.info(f"  → [PARTIAL RERUN] {ticker} debate (q_score={q_score})")
        qual = llm_qualitative_debate(d)
    elif USE_LLM:
        logging.info(f"  → [PARTIAL RERUN] {ticker} single LLM (q_score={q_score})")
        qual = llm_qualitative_score(d)
        _update_cost_tracker(ticker, 600)
    else:
        qual = {"grit": 0, "culture": 0, "catalyst": 0}

    grit = float(qual.get("grit", 0) or 0)
    culture = float(qual.get("culture", 0) or 0)
    catalyst = float(qual.get("catalyst", 0) or 0)
    total_qual = grit + culture + catalyst
    total_score = q_score + total_qual

    llm_parse_ok = _qual_parse_ok(qual)
    llm_mode = "debate" if USE_DEBATE and q_score >= debate_threshold else "single"

    updated = dict(row)
    updated.update({
        "qual_score": round(total_qual, 1),
        "total_score": round(total_score, 1),
        "tier": _tier_from_score(total_score, total_qual=total_qual),
        "grit": grit,
        "culture": culture,
        "catalyst": catalyst,
        "investment_thesis": qual.get("investment_thesis", ""),
        "debate_summary": qual.get("debate_summary", ""),
        "debate_json": build_debate_json(qual, llm_mode=llm_mode, parse_ok=llm_parse_ok),
        "llm_parse_ok": llm_parse_ok,
    })
    return updated


def partial_rerun_csv(
    csv_path: str,
    limit=None,
    max_workers=None,
    backup: bool = True,
    regenerate_one_pagers: bool = False,
    use_fmp: bool = False,
):
    """
    讀取現有 results CSV，只對 LLM 失敗列重跑，寫回同檔（可選備份）。
    quant_score / valuation 等欄位沿用原 CSV。
    """
    if max_workers is None:
        max_workers = MAX_WORKERS
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    mode = df["mode"].iloc[0] if "mode" in df.columns and len(df) else "compounder"

    mask = df.apply(needs_llm_rerun, axis=1)
    todo_idx = df.index[mask].tolist()
    if limit is not None:
        todo_idx = todo_idx[: int(limit)]

    total_failed = int(mask.sum())
    logging.info(
        f"[partial rerun] {csv_path}: {total_failed} rows need LLM rerun, "
        f"processing {len(todo_idx)} (workers={max_workers}, use_fmp={use_fmp})"
    )
    if not todo_idx:
        print(f"No LLM rerun needed for {csv_path}")
        return df

    if backup:
        backup_path = csv_path.replace(".csv", ".backup.csv")
        df.to_csv(backup_path, index=False, encoding="utf-8-sig")
        logging.info(f"Backup saved to {backup_path}")

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(rerun_llm_for_row, df.loc[i].to_dict(), mode, use_fmp): i
            for i in todo_idx
        }
        for future in as_completed(futures):
            idx = futures[future]
            ticker = df.at[idx, "ticker"]
            try:
                updated = future.result()
                for col, val in updated.items():
                    df.at[idx, col] = val
                completed += 1
                logging.info(
                    f"[partial rerun] [{completed}/{len(todo_idx)}] {ticker} "
                    f"tier={updated.get('tier')} score={updated.get('total_score')} "
                    f"parse_ok={updated.get('llm_parse_ok')}"
                )
            except Exception as e:
                logging.error(f"[partial rerun] failed {ticker}: {e}")

    if "llm_parse_ok" not in df.columns:
        df["llm_parse_ok"] = ~df.apply(needs_llm_rerun, axis=1)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Updated {csv_path} ({completed}/{len(todo_idx)} rows rerun)")

    if regenerate_one_pagers:
        generate_one_pagers(df, mode=mode, output_dir=f"one_pagers_{mode}")

    return df


# =================================================================
# 11. 多智能體辯論評分
# =================================================================
def llm_qualitative_debate(d: dict) -> dict:
    """
    三階段辯論：Bull → Bear → Judge
    回傳格式與 llm_qualitative_score 相同，額外包含 debate_summary。
    """
    if not USE_LLM or not client:
        return {"grit": 0, "culture": 0, "catalyst": 0, "debate_summary": "LLM disabled"}

    ticker = d["ticker"]
    news_text = extract_news_headlines(d.get("news", []), max_items=8)
    sec_ctx = d.get("sec_context", "")
    sec_block = f"\nSEC 10-K 背景：\n{sec_ctx[:1500]}" if sec_ctx else ""

    # --- 額外訊號區塊 ---
    earnings_sig = d.get("earnings_signals", {})
    earnings_excerpt = earnings_sig.get("raw_excerpt", "")
    earnings_block = (
        f"\nEarnings Call 摘錄（管理層語言信心分={earnings_sig.get('confidence_score', 'N/A')}）：\n"
        f"{earnings_excerpt[:400]}"
    ) if earnings_excerpt else ""

    trend_info = d.get("trends", {})
    trend_block = f"\nGoogle Trends 搜尋熱度：{trend_info.get('trend_direction', 'N/A')}" if trend_info.get("trend_direction", "未啟用") != "未啟用" else ""

    analyst_rating = d.get("analyst_rating", "")
    analyst_target = d.get("analyst_target")
    analyst_count = d.get("analyst_count", 0)
    analyst_block = (
        f"\n分析師共識：評級={analyst_rating}, 目標價={analyst_target}, 覆蓋分析師數={analyst_count}"
    ) if analyst_rating else ""

    base_info = (
        f"公司：{ticker}\n"
        f"業務：{d.get('desc', '')[:250]}\n"
        f"新聞摘要：{news_text}\n"
        f"財務：FCF Yield={d.get('ex_fcf_yield', 'N/A')}, "
        f"ROIC={d.get('roic', 'N/A')}, 員工數={d.get('employees', 'N/A')}"
        f"{sec_block}"
        f"{earnings_block}"
        f"{trend_block}"
        f"{analyst_block}"
    )

    _sector_model = get_model_for_sector(d.get("sector", ""))

    # --- Stage 1: Bull Agent ---
    try:
        bull_resp = client.chat.completions.create(
            model=_sector_model,
            messages=[
                {"role": "system", "content": (
                    "你是極端長期持有者（15年+），只找複利與護城河證據。"
                    "你必須反駁任何短期噪音，假設管理層在逆週期做對的事。"
                    "輸出 JSON，禁止廢話。"
                )},
                {"role": "user", "content": (
                    f"【長期持有者視角】根據以下資訊，找出此公司3個最強的投資亮點，"
                    f"並給出初步評分（grit/culture/catalyst 各滿分為 16.6/16.6/16.8）：\n\n"
                    f"{base_info}\n\n"
                    f"輸出JSON：{{\"bull_points\": [\"亮點1\",\"亮點2\",\"亮點3\"], "
                    f"\"suggested_scores\": {{\"grit\": X, \"culture\": X, \"catalyst\": X}}}}"
                )}
            ],
            temperature=0.3,
            max_tokens=300,   # 成本控制
            response_format={"type": "json_object"},
        )
        bull_content = bull_resp.choices[0].message.content or ""
        bull_data = _parse_llm_json(bull_content) or {}
    except Exception as e:
        logging.warning(f"Bull agent failed for {ticker}: {e}")
        bull_data = {}

    bull_points = bull_data.get("bull_points", ["資料不足"])
    bull_scores = bull_data.get("suggested_scores", {})

    # --- Stage 2: Bear Agent ---
    try:
        bear_resp = client.chat.completions.create(
            model=_sector_model,
            messages=[
                {"role": "system", "content": (
                    "你是激進空頭對沖基金分析師，專找會計地雷、治理漏洞、估值泡沫與結構性衰退。"
                    "你假設管理層在誤導市場；每個多方亮點都必須被拆解。"
                    "輸出 JSON，禁止客套。"
                )},
                {"role": "user", "content": (
                    f"【激進空頭視角】多方分析師對 {ticker} 提出以下亮點：{bull_points}\n"
                    f"初步評分：{bull_scores}\n\n"
                    f"公司基本資訊：\n{base_info}\n\n"
                    f"針對每個亮點提出最致命反駁，挑出尚未被定價的風險：\n"
                    f"輸出JSON：{{\"bear_rebuttals\": [\"反駁1\",\"反駁2\",\"反駁3\"], "
                    f"\"counter_scores\": {{\"grit\": X, \"culture\": X, \"catalyst\": X}}}}"
                )}
            ],
            temperature=0.3,
            max_tokens=300,   # 成本控制
            response_format={"type": "json_object"},
        )
        bear_content = bear_resp.choices[0].message.content or ""
        bear_data = _parse_llm_json(bear_content) or {}
    except Exception as e:
        logging.warning(f"Bear agent failed for {ticker}: {e}")
        bear_data = {}

    bear_rebuttals = bear_data.get("bear_rebuttals", ["資料不足"])
    bear_scores = bear_data.get("counter_scores", {})

    # --- Stage 3: Judge Agent（JSON mode；工具數據預先嵌入，不用 tool calling）---
    tool_block = _judge_tool_context(ticker)
    failure_block = ""
    if USE_FAILURE_CONTEXT:
        try:
            from audit_thesis import load_failure_context_for_prompt
            failure_block = load_failure_context_for_prompt(
                AUDIT_FAILURES_PATH,
                sector=d.get("sector", ""),
                n_samples=FAILURE_CONTEXT_SAMPLES,
            )
        except Exception as e:
            logging.debug(f"failure context load skipped: {e}")
    if failure_block:
        failure_block = f"\n{failure_block}\n"

    judge_prompt = (
        f"你是公正的投資法官，需綜合多空雙方論點後做出最終裁決。\n\n"
        f"公司：{ticker}\n"
        f"【多方亮點】：{bull_points}\n多方建議評分：{bull_scores}\n\n"
        f"【空方反駁】：{bear_rebuttals}\n空方建議評分：{bear_scores}\n\n"
        f"公司基本資訊：\n{base_info}\n"
        f"{tool_block}"
        f"{failure_block}"
        f"評分標準：\n"
        f"- grit（毅力, 滿分16.6）：逆週期資本開支→16.6；穩定R&D→10；跟風→0\n"
        f"- culture（文化, 滿分16.6）：CEO好評>85%且品牌文化壟斷→16.6；平穩→10；罷工醜聞→0\n"
        f"- catalyst（催化劑, 滿分16.8）：機構<20%且內部人增持→16.8；機構低無動靜→10；擁擠無變化→0\n\n"
        f"輸出純JSON：{{\"grit\": X, \"grit_reason\": \"...\", \"culture\": X, \"culture_reason\": \"...\", "
        f"\"catalyst\": X, \"catalyst_reason\": \"...\", \"debate_summary\": \"一句話結論\", "
        f"\"investment_thesis\": \"一句話說明為何這是（或不是）值得投資的公司，包含核心競爭優勢或風險\"}}"
    )

    try:
        judge_messages = [
            {"role": "system", "content": (
                "你是泰爾式風格的公正法官：懷疑後視鏡指標，但重視非結構化證據"
                "（10-K 段落、管理層語氣轉變、隱性資產線索）。"
                "只輸出 JSON object，不要 markdown。"
            )},
            {"role": "user", "content": judge_prompt},
        ]

        if JUDGE_USE_TOOLS and USE_TOOL_CALLING:
            judge_content = llm_with_tools(
                ticker, judge_messages, TOOLS, max_rounds=TOOL_MAX_ROUNDS, model=_sector_model
            )
            result = _parse_llm_json(judge_content)
            if result and all(k in result for k in ("grit", "culture", "catalyst")):
                result = _normalize_qual_result(result)
            else:
                result = None
        else:
            result = _call_judge_llm(ticker, judge_messages, _sector_model)

        if result is not None:
            result.setdefault("debate_summary", "")
            result["bull_points"] = bull_points
            result["bear_rebuttals"] = bear_rebuttals
            result["bull_scores"] = bull_scores
            result["bear_scores"] = bear_scores
            _update_cost_tracker(ticker, 1150)
            logging.info(f"  → [DEBATE OK] {ticker}: {result.get('debate_summary', '')[:80]}")
            return result
        else:
            logging.warning(f"Judge agent JSON parse failed for {ticker}")
            _update_cost_tracker(ticker, 750)
            return {
                "grit": 10, "culture": 10, "catalyst": 10,
                "debate_summary": "JSON解析失敗，使用預設中性分數",
                "investment_thesis": "資料不足，無法判斷",
                "bull_points": bull_points,
                "bear_rebuttals": bear_rebuttals,
                "bull_scores": bull_scores,
                "bear_scores": bear_scores,
            }
    except Exception as e:
        logging.warning(f"Judge agent failed for {ticker}: {e}")
        _update_cost_tracker(ticker, 750)
        return {"grit": 0, "culture": 0, "catalyst": 0, "debate_summary": f"error: {e}", "investment_thesis": "資料不足，無法判斷"}

# =================================================================
# 11b. 機構級 One-Pager 投資備忘錄（Markdown）
# =================================================================
def generate_one_pagers(results_df, mode="compounder", output_dir="one_pagers"):
    """為 S-Tier（和 A-Tier）股票生成 Markdown 格式的機構級投資備忘錄"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        top_stocks = results_df[results_df["tier"].isin(["S", "A"])].copy()
        top_stocks = top_stocks.sort_values("total_score", ascending=False)

        generated = []
        for _, row in top_stocks.iterrows():
            ticker = row["ticker"]
            tier = row["tier"]
            region = row.get("region", "US")

            roic_val = row.get("roic")
            roic_str = f"{roic_val:.1f}%" if roic_val and str(roic_val) != "nan" else "N/A"
            fcf_val = row.get("ex_fcf_yield")
            fcf_str = f"{fcf_val:.2f}%" if fcf_val and str(fcf_val) != "nan" else "N/A"
            val_str = row.get("valuation", "N/A")
            val_reason = row.get("valuation_reason", "")
            roic_trend = row.get("roic_trend", "N/A")
            golden = "🎯 黃金擊球區（無人問津的優質股）" if row.get("golden_zone") else ""
            sbc_note = "⚠️ SBC 已調整（真實 FCF 低於帳面）" if row.get("sbc_adjusted") else ""
            buyback = "💰 庫藏股複利機器（負股東權益為回購所致）" if row.get("dupont_label") == "buyback_compounder" else ""

            md = f"""# {ticker} [{tier}-Tier] — {region} 投資備忘錄
**模式：{mode.upper()} | 總分：{row['total_score']}/110 | 估值：{val_str}（{val_reason}）**

{golden}
{sbc_note}
{buyback}

---

## 核心量化數據

| 指標 | 數值 |
|------|------|
| ROIC | {roic_str} |
| ROIC 趨勢 | {roic_trend} |
| 超額 FCF 殖利率 | {fcf_str} |
| 量化分 | {row['quant_score']}/60 |
| 定性分（AI 辯論） | {row.get('qual_score', 0)}/50 |
| 估值訊號 | {val_str} — {val_reason} |

---

## 投資論點

{row.get('investment_thesis', '（無資料）')}

---

## AI 辯論摘要

{row.get('debate_summary', '（未觸發辯論模式）')}

---

## Bear Case 最強反駁點

*（參考 AI 辯論的空方論點，以下為系統標記的最高風險點）*

{row.get('kill_reason') or '無重大風險標記'}

---

## 人腦最終定價決策（留白）

**當前市價：** ___________

**我的合理估值：** ___________

**安全邊際：** ___________

**部位規模決策：** ___________

**觀察觸發點：** ___________

---
*報告生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | 資料來源：yfinance + DeepSeek AI*
"""
            filename = f"{output_dir}/{mode}_{tier}_{ticker.replace('.', '_')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md)
            generated.append(filename)

        print(f"\n📄 已生成 {len(generated)} 份 One-Pager 備忘錄至 {output_dir}/ 資料夾")
        return generated
    except Exception as e:
        logging.warning(f"One-Pager 生成失敗：{e}")
        return []


# =================================================================
# 12. 回測引擎
# =================================================================
def _extract_close_series(hist):
    """相容 yfinance 單檔 / 多檔 MultiIndex 欄位。"""
    if hist is None or getattr(hist, "empty", True):
        return None
    cols = hist.columns
    if isinstance(cols, pd.MultiIndex):
        if "Close" not in cols.get_level_values(0):
            return None
        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            return close.iloc[:, 0]
        return close
    if "Close" in cols:
        return hist["Close"]
    return None


def _hist_total_return_pct(hist) -> float | None:
    """從 yfinance history 計算區間報酬 %；略過 Close 首尾 NaN。"""
    try:
        close = _extract_close_series(hist)
        if close is None:
            return None
        close = close.dropna()
        if close.empty or len(close) < 10:
            return None
        ret = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100
        if np.isnan(ret) or np.isinf(ret):
            return None
        return ret
    except Exception:
        return None


def _summarize_returns(returns):
    if not returns:
        return None
    sorted_r = sorted(returns)
    return {
        "avg_return": round(float(np.mean(returns)), 1),
        "median_return": round(sorted_r[len(sorted_r) // 2], 1),
        "stocks_tested": len(returns),
        "best": round(max(returns), 1),
        "worst": round(min(returns), 1),
    }


def _fetch_ticker_returns(tickers, start_date, end_date, batch_size=40):
    """批量下載 yfinance 歷史價，回傳 {ticker: return_pct}。"""
    tickers = list(dict.fromkeys(tickers))
    out = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            hist_all = yf.download(
                batch, start=str(start_date), end=str(end_date),
                progress=False, auto_adjust=True, group_by="ticker",
            )
            if len(batch) == 1:
                ret = _hist_total_return_pct(hist_all)
                if ret is not None:
                    out[batch[0]] = ret
            else:
                for t in batch:
                    try:
                        ret = _hist_total_return_pct(hist_all[t])
                        if ret is not None:
                            out[t] = ret
                    except Exception:
                        continue
        except Exception as e:
            logging.warning(f"批量回測下載失敗 ({batch[:3]}...): {e}")
    return out


def _spy_benchmark(start_date, end_date):
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False, auto_adjust=True)
        spy_ret = _hist_total_return_pct(spy)
        return round(spy_ret, 1) if spy_ret is not None else None
    except Exception:
        return None


ATTENTION_SIGNAL_GROUPS = ["潛在珍珠", "共識轉折", "neutral"]


def run_backtest_attention_groups(
    results_csv_path="results_compounder.csv",
    lookback_years=2,
    backtest_output="backtest_compounder_attention.csv",
    tiers=("S", "A"),
):
    """
    S/A tier × attention_signal 分組回測（僅美股）。
    輸出 CSV：tier, attention_signal, avg/median/best/worst, alpha_vs_spy, universe_count
    """
    try:
        import datetime

        df = pd.read_csv(results_csv_path, encoding="utf-8-sig")
        if "attention_signal" not in df.columns:
            logging.warning(f"{results_csv_path} 缺少 attention_signal，請先跑 refresh_volume_attention.py")
            return

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * lookback_years)

        subset = df[df["region"].eq("US") & df["tier"].isin(tiers)].copy()
        subset["attention_signal"] = subset["attention_signal"].fillna("N/A")

        ret_map = _fetch_ticker_returns(subset["ticker"].tolist(), start_date, end_date)
        benchmark = _spy_benchmark(start_date, end_date)

        rows = []
        tier_labels = list(tiers) + ["S+A"]

        for tier_label in tier_labels:
            for sig in ATTENTION_SIGNAL_GROUPS:
                if tier_label == "S+A":
                    grp = subset[subset["attention_signal"] == sig]
                else:
                    grp = subset[(subset["tier"] == tier_label) & (subset["attention_signal"] == sig)]

                returns = [ret_map[t] for t in grp["ticker"] if t in ret_map]
                stats = _summarize_returns(returns)
                if not stats:
                    continue

                alpha = round(stats["avg_return"] - benchmark, 1) if benchmark is not None else None
                med_alpha = (
                    round(stats["median_return"] - benchmark, 1) if benchmark is not None else None
                )
                rows.append({
                    "tier": tier_label,
                    "attention_signal": sig,
                    "universe_count": len(grp),
                    **stats,
                    "alpha_vs_spy": alpha,
                    "median_alpha_vs_spy": med_alpha,
                    "benchmark_spy": benchmark,
                })

        if not rows:
            logging.warning("attention 分組回測無有效樣本")
            return

        backtest_df = pd.DataFrame(rows)
        backtest_df.to_csv(backtest_output, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 80)
        print(f"📈 Attention 分組回測（{tiers} tier × 珍珠/轉折/neutral，過去 {lookback_years} 年，美股）")
        print("=" * 80)
        if benchmark is not None:
            print(f"S&P 500 基準報酬：{benchmark:+.1f}%")
        print(f"有效報酬樣本：{len(ret_map)}/{len(subset)} 檔")

        for tier_label in tier_labels:
            block = backtest_df[backtest_df["tier"] == tier_label]
            if block.empty:
                continue
            print(f"\n--- {tier_label} ---")
            for _, r in block.iterrows():
                print(
                    f"  {r['attention_signal']:6s}  n={int(r['stocks_tested']):2d}/{int(r['universe_count']):2d}  "
                    f"avg={r['avg_return']:+.1f}%  med={r['median_return']:+.1f}%  "
                    f"α={r['alpha_vs_spy']:+.1f}%  "
                    f"[{r['worst']:+.1f}% ~ {r['best']:+.1f}%]"
                )

        print(f"\n✅ 分組回測已存至 {backtest_output}")

    except Exception as e:
        logging.warning(f"attention 分組回測失敗：{e}")


def run_backtest(results_csv_path="global_quantamental_results.csv", lookback_years=2, backtest_output="backtest_results.csv"):
    """
    讀取評分結果，回測各 Tier 的歷史報酬。
    只跑美股（yfinance 歷史數據最準確）。
    """
    try:
        import datetime
        df = pd.read_csv(results_csv_path, encoding="utf-8-sig")
        us_stocks = df[df["region"] == "US"].copy()

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * lookback_years)

        tier_returns = {}

        for tier in ["S", "A", "B", "C"]:
            tickers = us_stocks[us_stocks["tier"] == tier]["ticker"].tolist()
            if not tickers:
                continue

            batch = tickers[:20]
            returns = []
            try:
                hist_all = yf.download(
                    batch, start=str(start_date), end=str(end_date),
                    progress=False, auto_adjust=True, group_by="ticker"
                )
                for t in batch:
                    try:
                        hist = hist_all[t] if len(batch) > 1 else hist_all
                        ret = _hist_total_return_pct(hist)
                        if ret is not None:
                            returns.append(ret)
                    except Exception:
                        continue
            except Exception as e:
                logging.warning(f"批量回測下載失敗 ({tier}): {e}")

            if returns:
                sorted_r = sorted(returns)
                tier_returns[tier] = {
                    "avg_return": round(float(np.mean(returns)), 1),
                    "median_return": round(sorted_r[len(sorted_r) // 2], 1),
                    "stocks_tested": len(returns),
                    "best": round(max(returns), 1),
                    "worst": round(min(returns), 1),
                }

        # 比較 S&P 500 基準
        try:
            spy = yf.download("SPY", start=start_date, end=end_date, progress=False, auto_adjust=True)
            spy_ret = _hist_total_return_pct(spy)
            benchmark = round(spy_ret, 1) if spy_ret is not None else None
        except Exception:
            benchmark = None

        print("\n" + "=" * 80)
        print(f"📈 回測報告（過去 {lookback_years} 年）")
        print("=" * 80)
        if benchmark is not None:
            print(f"S&P 500 基準報酬：{benchmark:+.1f}%")
        for tier, stats in tier_returns.items():
            alpha = round(stats["avg_return"] - benchmark, 1) if benchmark is not None else "N/A"
            print(f"\n{tier}-Tier（{stats['stocks_tested']}支）：")
            if isinstance(alpha, float):
                print(f"  平均報酬：{stats['avg_return']:+.1f}%  |  中位數：{stats['median_return']:+.1f}%  |  Alpha：{alpha:+.1f}%")
            else:
                print(f"  平均報酬：{stats['avg_return']:+.1f}%")
            print(f"  最佳：{stats['best']:+.1f}%  |  最差：{stats['worst']:+.1f}%")

        backtest_df = pd.DataFrame([
            {"tier": t, **s, "benchmark_spy": benchmark}
            for t, s in tier_returns.items()
        ])
        backtest_df.to_csv(backtest_output, index=False, encoding="utf-8-sig")
        print(f"\n✅ 回測結果已存至 {backtest_output}")

    except Exception as e:
        logging.warning(f"回測失敗：{e}")


# =================================================================
# 12b. SQLite 斷點續傳
# =================================================================
def init_checkpoint():
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            ticker TEXT PRIMARY KEY,
            data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_checkpoint(conn, ticker, result_dict):
    conn.execute(
        "INSERT OR REPLACE INTO results (ticker, data) VALUES (?, ?)",
        (ticker, json.dumps(result_dict, default=str))
    )
    conn.commit()


def load_checkpoint(conn):
    rows = conn.execute("SELECT ticker, data FROM results").fetchall()
    return {row[0]: json.loads(row[1]) for row in rows}


# =================================================================
# 13. 並發輔助函數 & 主執行流程
# =================================================================
def _save_checkpoint_thread(ck_db: str, ticker: str, result_dict: dict) -> None:
    """執行緒安全的斷點儲存（每次建立獨立 SQLite 連線，以鎖序列化寫入）。"""
    with _checkpoint_lock:
        try:
            conn = sqlite3.connect(ck_db)
            conn.execute(
                "INSERT OR REPLACE INTO results (ticker, data) VALUES (?, ?)",
                (ticker, json.dumps(result_dict, default=str))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.warning(f"Checkpoint save failed for {ticker}: {e}")


def process_ticker(args):
    """
    單支股票完整處理流程（fetch_data + 量化評分 + LLM 評分）。
    設計為在 ThreadPoolExecutor 內並發執行，所有評分邏輯保持不變。
    args = (ticker, region, mode, ck_db)
    """
    ticker, region, mode, ck_db = args
    debate_threshold = DEBATE_MIN_QSCORE if mode == "compounder" else 10
    try:
        d = fetch_data(ticker, region)
        time.sleep(0.5)

        if mode == "growth":
            q_score, q_details, roic, ex_yield = quant_score_growth(d)
        else:
            q_score, q_details, roic, ex_yield = quant_score(d)
        d["roic"] = roic
        d["ex_fcf_yield"] = ex_yield
        d["screen_mode"] = mode

        # SEC RAG（FAST_MODE=True 時 USE_SEC_RAG 已在 Config 設為 False，此區自動跳過）
        sec_ctx = ""
        if mode == "compounder" and USE_SEC_RAG and region == "US" and q_score >= SEC_RAG_MIN_QSCORE:
            logging.info(f"  → [SEC RAG] Fetching 10-K for {ticker} (q_score={q_score})...")
            sec_ctx = fetch_sec_context(ticker)
            if sec_ctx:
                logging.info(f"  → SEC context loaded ({len(sec_ctx)} chars)")
        elif mode == "compounder" and USE_SEC_RAG and region == "US" and q_score < SEC_RAG_MIN_QSCORE:
            logging.info(f"  → [SEC RAG] 跳過 {ticker}（q_score={q_score} < {SEC_RAG_MIN_QSCORE}，節省 token）")
        d["sec_context"] = sec_ctx

        kill, kill_reason = llm_kill_switches(d)
        if kill:
            _update_cost_tracker(ticker, 200)
            kill_qual = {"debate_summary": "", "investment_thesis": "", "kill_reason": kill_reason}
            result_row = {
                "ticker": ticker, "region": region, "mode": mode,
                "total_score": 0, "tier": "C (Kill)",
                "quant_score": q_score, "qual_score": 0,
                "roic": roic, "ex_fcf_yield": ex_yield,
                "kill_reason": kill_reason,
                "investment_thesis": "", "debate_summary": "",
                "debate_json": build_debate_json(kill_qual, llm_mode="kill", parse_ok=True),
                "valuation": q_details.get("valuation", "無法判斷"),
                "valuation_reason": q_details.get("valuation_reason", ""),
                "earnings_confidence": q_details.get("earnings_confidence", 5.0),
                "roic_trend": q_details.get("roic_trend", "N/A"),
                "grit": 0, "culture": 0, "catalyst": 0,
                "sbc_adjusted": q_details.get("sbc_adjusted", False),
                "golden_zone": q_details.get("golden_zone", False),
                "dupont_label": q_details.get("dupont_label", ""),
                "volume_z": q_details.get("volume_z"),
                "volume_z_delta": q_details.get("volume_z_delta"),
                "attention_signal": q_details.get("attention_signal", "N/A"),
                "volume_attention_score": q_details.get("volume_attention_score", 0),
                "llm_parse_ok": True,
            }
            if USE_CHECKPOINT:
                _save_checkpoint_thread(ck_db, ticker, result_row)
            return result_row

        if USE_DEBATE and q_score >= debate_threshold:
            logging.info(f"  → [DEBATE] 啟動三階段辯論（q_score={q_score} >= {debate_threshold}）")
            qual = llm_qualitative_debate(d)
            llm_mode = "debate"
        elif USE_LLM:
            logging.info(f"  → [SINGLE LLM] q_score={q_score} < {debate_threshold}，使用單一 LLM 評分")
            qual = llm_qualitative_score(d)
            llm_mode = "single"
            _update_cost_tracker(ticker, 600)
        else:
            qual = {"grit": 0, "culture": 0, "catalyst": 0}
            llm_mode = "none"

        parse_ok = _qual_parse_ok(qual)
        debate_json = build_debate_json(qual, llm_mode=llm_mode, parse_ok=parse_ok)

        grit = qual.get("grit", 0)
        culture = qual.get("culture", 0)
        catalyst = qual.get("catalyst", 0)
        total_qual = grit + culture + catalyst
        total_score = q_score + total_qual

        if q_details.get("grey_zone") and total_qual >= 40 and total_score < 50:
            total_score = 50

        if total_score >= S_TIER_MIN:
            tier = "S"
        elif total_score >= A_TIER_MIN:
            tier = "A"
        elif total_score >= B_TIER_MIN:
            tier = "B"
        else:
            tier = "C"

        result_row = {
            "ticker": ticker, "region": region, "mode": mode,
            "total_score": round(total_score, 1), "tier": tier,
            "quant_score": q_score, "qual_score": round(total_qual, 1),
            "roic": roic, "ex_fcf_yield": ex_yield,
            "grit": grit, "culture": culture, "catalyst": catalyst,
            "kill_reason": "",
            "investment_thesis": qual.get("investment_thesis", ""),
            "debate_summary": qual.get("debate_summary", ""),
            "debate_json": debate_json,
            "valuation": q_details.get("valuation", "無法判斷"),
            "valuation_reason": q_details.get("valuation_reason", ""),
            "earnings_confidence": q_details.get("earnings_confidence", 5.0),
            "roic_trend": q_details.get("roic_trend", "N/A"),
            "sbc_adjusted": q_details.get("sbc_adjusted", False),
            "golden_zone": q_details.get("golden_zone", False),
            "dupont_label": q_details.get("dupont_label", ""),
            "volume_z": q_details.get("volume_z"),
            "volume_z_delta": q_details.get("volume_z_delta"),
            "attention_signal": q_details.get("attention_signal", "N/A"),
            "volume_attention_score": q_details.get("volume_attention_score", 0),
            "llm_parse_ok": parse_ok,
        }
        if USE_CHECKPOINT:
            _save_checkpoint_thread(ck_db, ticker, result_row)
        return result_row
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")
        return None


def _run_mode(mode: str, tickers: list, region_map: dict) -> list:
    """
    執行單一模式的全股票評分流程（全並發 fetch + LLM），回傳 results 列表。
    mode: "compounder" 或 "growth"
    """
    # 斷點續傳初始化（每個模式獨立的 checkpoint DB）
    ck_db = f"checkpoint_{mode}.db"
    if USE_CHECKPOINT:
        conn = sqlite3.connect(ck_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                ticker TEXT PRIMARY KEY,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        completed_rows = conn.execute("SELECT ticker, data FROM results").fetchall()
        completed = {row[0]: json.loads(row[1]) for row in completed_rows}
        conn.close()
        logging.info(f"[{mode}] 斷點續傳：已完成 {len(completed)} 支，跳過")
    else:
        completed = {}

    tickers_to_process = [t for t in tickers if t not in completed]
    logging.info(
        f"[{mode}] 待處理：{len(tickers_to_process)} 支 | "
        f"並發數={MAX_WORKERS} | FAST_MODE={FAST_MODE}"
    )

    results = list(completed.values())

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_ticker, (t, region_map.get(t, "US"), mode, ck_db)): t
            for t in tickers_to_process
        }
        for future in as_completed(futures):
            t = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    logging.info(
                        f"[{mode}] [{len(results)}/{len(tickers)}] "
                        f"[{region_map.get(t, 'US')}] {t} - "
                        f"tier={result.get('tier', 'N/A')} score={result.get('total_score', 0)}"
                    )
            except Exception as e:
                logging.error(f"[{mode}] Error {t}: {e}")

    # 斷點收尾：全部完成才清除
    if USE_CHECKPOINT and len(results) >= len(tickers):
        try:
            os.remove(ck_db)
            logging.info(f"[{mode}] 已清除斷點檔案 {ck_db}")
        except OSError:
            pass

    return results


def _print_and_save_results(mode: str, results: list):
    """輸出榜單並儲存 CSV。"""
    df = pd.DataFrame(results)
    csv_name = f"results_{mode}.csv"
    df.to_csv(csv_name, index=False, encoding="utf-8-sig")

    print(f"\n{'='*80}")
    print(f"【{mode.upper()} 模式】全球投資層級榜單")
    print(f"{'='*80}")
    for tier in ["S", "A", "B", "C"]:
        subset = df[df["tier"] == tier]
        if not subset.empty:
            print(f"\n--- {tier}-Tier ---")
            display_cols = ["ticker", "region", "total_score", "quant_score", "qual_score", "valuation", "roic"]
            if mode == "compounder" and tier in ("S", "A"):
                for col in ("attention_signal", "volume_z", "roic_trend", "debate_summary", "earnings_confidence"):
                    if col in subset.columns:
                        display_cols.append(col)
            print(subset[display_cols].to_string(index=False))
            if tier in ("S", "A") and "investment_thesis" in subset.columns:
                print()
                for _, row in subset.iterrows():
                    thesis = row.get("investment_thesis", "")
                    val = row.get("valuation", "")
                    val_reason = row.get("valuation_reason", "")
                    if thesis:
                        print(f"  → {row['ticker']} [{val} - {val_reason}]: {thesis}")

    print(f"\n✅ {mode} 結果已存至 {csv_name}")
    return df


def main():
    # 數據源已切換至 FMP，清除舊 yfinance checkpoint
    import glob
    for old_ck in glob.glob("checkpoint_*.db"):
        try:
            os.remove(old_ck)
            logging.info(f"已清除舊 checkpoint: {old_ck}")
        except Exception:
            pass

    # Regions: any subset of ["US","EU","JP","HK","TW"], or None for all
    # max_stocks: None = entire universe, integer = cap for quick testing
    tickers, region_map = get_global_universe(max_stocks=None, regions=["US", "EU", "JP", "HK", "TW"])

    if SCREEN_MODE == "both":
        modes = ["compounder", "growth"]
    else:
        modes = [SCREEN_MODE]

    all_results = {}

    for mode in modes:
        logging.info(f"=== 運行模式：{mode} ===")
        all_results[mode] = _run_mode(mode, tickers, region_map)

    for mode, results in all_results.items():
        _print_and_save_results(mode, results)

    if GENERATE_ONE_PAGERS:
        for mode, results in all_results.items():
            df_mode = pd.DataFrame(results)
            if not df_mode.empty:
                generate_one_pagers(df_mode, mode=mode, output_dir=f"one_pagers_{mode}")

    if RUN_BACKTEST:
        for mode in all_results.keys():
            csv_path = f"results_{mode}.csv"
            bt_output = f"backtest_{mode}.csv"
            logging.info(f"開始回測 [{mode}]...")
            run_backtest(results_csv_path=csv_path, backtest_output=bt_output)

if __name__ == "__main__":
    main()