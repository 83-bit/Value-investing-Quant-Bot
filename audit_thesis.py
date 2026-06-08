"""
Thesis 審計與勝率歸因 — 讀取 results_*.csv，對照近期股價，產出 audit_failures.json

用法:
  python audit_thesis.py
  python audit_thesis.py --csv results_compounder.csv --lookback-days 90
  python audit_thesis.py --both --attribution attribution_compounder.csv
"""
import argparse
import datetime
import json
import logging
import os
import random

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_MIN_SCORE = 70
DEFAULT_MAX_RETURN = -15.0
DEFAULT_LOOKBACK_DAYS = 90
# 相對 SPY 落後超過此值 → 歸因為 thesis_failure；否則可能係 market_noise
DEFAULT_ALPHA_FAIL = -10.0
DEFAULT_ALPHA_NOISE = -5.0


def _fetch_return(ticker: str, start_date, end_date) -> float | None:
    try:
        hist = yf.download(
            ticker, start=str(start_date), end=str(end_date),
            progress=False, auto_adjust=True,
        )
        if hist.empty or len(hist) < 5:
            return None
        close = hist["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        return float((close.iloc[-1] / close.iloc[0] - 1) * 100)
    except Exception as e:
        logging.debug(f"return fetch failed {ticker}: {e}")
        return None


def _classify_failure(stock_ret: float, spy_ret: float | None) -> str:
    """區分 thesis 邏輯問題 vs 市場雜訊。"""
    if spy_ret is None:
        return "unknown"
    alpha = stock_ret - spy_ret
    if stock_ret >= DEFAULT_MAX_RETURN:
        return "not_failure"
    if alpha <= DEFAULT_ALPHA_FAIL:
        return "thesis_failure"
    if alpha >= DEFAULT_ALPHA_NOISE:
        return "market_noise"
    return "mixed"


def _parse_debate_json(row) -> dict:
    raw = row.get("debate_json")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except json.JSONDecodeError:
        return {}


def _infer_sector(row) -> str:
    dj = _parse_debate_json(row)
    return str(dj.get("sector") or row.get("sector") or "Unknown")


def run_audit(
    csv_path: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_score: float = DEFAULT_MIN_SCORE,
    max_return: float = DEFAULT_MAX_RETURN,
    output_json: str = "audit_failures.json",
    attribution_csv: str | None = None,
) -> dict:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=lookback_days)

    spy_ret = _fetch_return("SPY", start_date, end_date)
    logging.info(f"SPY {lookback_days}d return: {spy_ret:+.1f}%" if spy_ret is not None else "SPY return N/A")

    candidates = df[
        (df["total_score"] >= min_score) & (df["tier"].isin(["S", "A"]))
    ].copy()
    logging.info(f"Auditing {len(candidates)} S/A names from {csv_path}")

    failures = []
    attribution_rows = []

    for _, row in candidates.iterrows():
        ticker = row["ticker"]
        region = row.get("region", "US")
        if region != "US":
            continue

        stock_ret = _fetch_return(ticker, start_date, end_date)
        if stock_ret is None:
            continue

        alpha = (stock_ret - spy_ret) if spy_ret is not None else None
        failure_type = _classify_failure(stock_ret, spy_ret)
        dj = _parse_debate_json(row)
        thesis = str(row.get("investment_thesis") or "")
        summary = str(row.get("debate_summary") or "")

        ai_label = "ai_correct" if stock_ret > 0 else (
            "ai_miss_thesis" if failure_type == "thesis_failure" else (
                "ai_miss_market" if failure_type == "market_noise" else "ai_miss_mixed"
            )
        )

        attribution_rows.append({
            "ticker": ticker,
            "tier": row["tier"],
            "total_score": row["total_score"],
            "return_pct": round(stock_ret, 1),
            "spy_return_pct": round(spy_ret, 1) if spy_ret is not None else None,
            "alpha_pct": round(alpha, 1) if alpha is not None else None,
            "failure_type": failure_type,
            "ai_label": ai_label,
            "sbc_adjusted": row.get("sbc_adjusted"),
            "roic": row.get("roic"),
            "llm_parse_ok": row.get("llm_parse_ok"),
        })

        if stock_ret >= max_return:
            continue

        failures.append({
            "ticker": ticker,
            "region": region,
            "mode": row.get("mode", ""),
            "tier": row["tier"],
            "total_score": float(row["total_score"]),
            "return_pct": round(stock_ret, 1),
            "spy_return_pct": round(spy_ret, 1) if spy_ret is not None else None,
            "alpha_pct": round(alpha, 1) if alpha is not None else None,
            "failure_type": failure_type,
            "investment_thesis": thesis[:500],
            "debate_summary": summary[:300],
            "bull_points": dj.get("bull_points"),
            "bear_rebuttals": dj.get("bear_rebuttals"),
            "pseudo_highlight": _detect_pseudo_highlight(thesis, summary, stock_ret, alpha),
        })

    summary_text = _build_prompt_summary(failures, lookback_days)

    payload = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_csv": os.path.basename(csv_path),
        "lookback_days": lookback_days,
        "min_score": min_score,
        "max_return_threshold": max_return,
        "spy_return_pct": round(spy_ret, 1) if spy_ret is not None else None,
        "failure_count": len(failures),
        "failures": failures,
        "summary_for_prompt": summary_text,
        "attribution_stats": _attribution_stats(attribution_rows),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ Wrote {len(failures)} failures → {output_json}")

    if attribution_csv:
        pd.DataFrame(attribution_rows).to_csv(attribution_csv, index=False, encoding="utf-8-sig")
        logging.info(f"✅ Attribution → {attribution_csv}")

    return payload


def _detect_pseudo_highlight(thesis: str, summary: str, stock_ret: float, alpha) -> str:
    """偽亮點：Thesis 語氣正面但股價大幅跑輸。"""
    text = (thesis + " " + summary).lower()
    bullish = any(w in text for w in ["護城河", "龍頭", "優勢", "低估", "成長", "competitive", "undervalued"])
    if bullish and stock_ret < -15:
        if alpha is not None and alpha < -10:
            return "樂觀敘事與股價大幅跑輸大盤 — 可能高估護城河或忽略周期"
        return "敘事偏正面但報酬為負 — 需檢查是否僅為市場 beta"
    if "JSON解析失敗" in summary or thesis == "資料不足，無法判斷":
        return "定性評分不可靠（JSON 失敗）卻進入高 Tier — 系統性偽信號"
    return "高分解但報酬不佳 — 檢查量化/定性權重"


def _build_prompt_summary(failures: list, lookback_days: int, max_chars: int = 500) -> str:
    if not failures:
        return f"過去 {lookback_days} 天無 S/A 高分解但報酬 < {DEFAULT_MAX_RETURN}% 的案例。"

    thesis_fail = [f for f in failures if f["failure_type"] == "thesis_failure"]
    noise = [f for f in failures if f["failure_type"] == "market_noise"]
    lines = [
        f"【Thesis 審計】近 {lookback_days} 日共 {len(failures)} 宗高分解失利"
        f"（thesis_failure={len(thesis_fail)}, market_noise={len(noise)}）。",
    ]
    for f in failures[:4]:
        lines.append(
            f"- {f['ticker']}({f['tier']},{f['total_score']:.0f}分): "
            f"報酬{f['return_pct']:+.0f}%, α{f.get('alpha_pct', 0):+.0f}% — {f['pseudo_highlight'][:60]}"
        )
    text = " ".join(lines)
    return text[:max_chars]


def _attribution_stats(rows: list) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {
        "total_sa": len(df),
        "ai_correct": int((df["ai_label"] == "ai_correct").sum()),
        "ai_miss_thesis": int((df["ai_label"] == "ai_miss_thesis").sum()),
        "ai_miss_market": int((df["ai_label"] == "ai_miss_market").sum()),
        "avg_return_pct": round(float(df["return_pct"].mean()), 1),
    }


def load_failure_context_for_prompt(
    audit_path: str = "audit_failures.json",
    sector: str = "",
    n_samples: int = 4,
) -> str:
    """供 global_screener Judge 注入的失敗情境摘要。"""
    if not os.path.isfile(audit_path):
        return ""
    try:
        with open(audit_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""

    summary = data.get("summary_for_prompt", "")
    failures = data.get("failures") or []
    if not failures:
        return summary

    pool = [f for f in failures if f.get("failure_type") == "thesis_failure"]
    if not pool:
        pool = failures

    if sector:
        sector_lower = sector.lower()
        matched = [
            f for f in pool
            if sector_lower in str(f.get("investment_thesis", "")).lower()
        ]
        if matched:
            pool = matched

    samples = random.sample(pool, min(n_samples, len(pool)))
    lines = [summary, "", "【歷史失敗案例 — Judge 須預先警惕】"]
    for s in samples:
        lines.append(
            f"• {s['ticker']}: {s['pseudo_highlight']} "
            f"(報酬{s['return_pct']:+.0f}%, thesis_failure={s['failure_type']})"
        )
    return "\n".join(lines)[:1200]


def main():
    parser = argparse.ArgumentParser(description="Audit investment thesis vs realized returns")
    parser.add_argument("--csv", default="results_compounder.csv")
    parser.add_argument("--both", action="store_true", help="Audit compounder + growth")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--max-return", type=float, default=DEFAULT_MAX_RETURN)
    parser.add_argument("--output", default="audit_failures.json")
    parser.add_argument("--attribution", default=None, help="Optional attribution CSV path")
    args = parser.parse_args()

    if args.both:
        for mode in ("compounder", "growth"):
            run_audit(
                f"results_{mode}.csv",
                lookback_days=args.lookback_days,
                min_score=args.min_score,
                max_return=args.max_return,
                output_json=f"audit_failures_{mode}.json",
                attribution_csv=args.attribution or f"attribution_{mode}.csv",
            )
        return

    run_audit(
        args.csv,
        lookback_days=args.lookback_days,
        min_score=args.min_score,
        max_return=args.max_return,
        output_json=args.output,
        attribution_csv=args.attribution or "attribution.csv",
    )


if __name__ == "__main__":
    main()
