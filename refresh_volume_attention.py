"""
只重算 Volume Attention（Momentum-Attention Score），寫回現有 results_*.csv。
不呼叫 LLM、不重算 ROIC/FCF；quant/total/tier 僅在符合珍珠/轉折條件時加上 volume 加分。

用法:
  python refresh_volume_attention.py --dry-run
  python refresh_volume_attention.py --limit 20
  python refresh_volume_attention.py --mode compounder
  python refresh_volume_attention.py --fetch-fmp   # 預設只用 yfinance（較快、免 FMP 403）
"""
import argparse
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Tuple

import pandas as pd

from global_screener import (
    A_TIER_MIN,
    B_TIER_MIN,
    MAX_WORKERS,
    QUANT_CORE_HIGH_MIN,
    S_TIER_MIN,
    SCORE_VOLUME_BREAKOUT,
    SCORE_VOLUME_PEARL,
    fetch_volume_attention,
)


def _tier_from_score(total_score: float, grey_zone: bool = False, total_qual: float = 0) -> str:
    score = float(total_score)
    if grey_zone and total_qual >= 40 and score < 50:
        score = 50
    if score >= S_TIER_MIN:
        return "S"
    if score >= A_TIER_MIN:
        return "A"
    if score >= B_TIER_MIN:
        return "B"
    return "C"


def _truthy(val) -> bool:
    if pd.isna(val):
        return False
    return str(val).strip().lower() in ("true", "1", "yes")


def _volume_bonus(attn, base_quant: float) -> Tuple[float, str]:
    if not attn or base_quant < QUANT_CORE_HIGH_MIN:
        return 0.0, attn.get("attention_signal", "N/A") if attn else "N/A"

    sig = attn.get("attention_signal", "neutral")
    if sig == "潛在珍珠":
        return float(SCORE_VOLUME_PEARL), sig
    if sig == "共識轉折":
        return float(SCORE_VOLUME_BREAKOUT), sig
    return 0.0, sig


def _process_row(row: dict, fmp_first: bool) -> dict:
    ticker = str(row["ticker"])
    attn = fetch_volume_attention(ticker, fmp_first=fmp_first)

    old_bonus_raw = row.get("volume_attention_score")
    if pd.isna(old_bonus_raw) or old_bonus_raw is None or old_bonus_raw == "":
        old_bonus = 0.0
    else:
        old_bonus = float(old_bonus_raw)
    base_quant = float(row["quant_score"]) - old_bonus
    base_total = float(row["total_score"]) - old_bonus

    bonus, signal = _volume_bonus(attn, base_quant)
    new_quant = base_quant + bonus
    new_total = base_total + bonus

    grey = _truthy(row.get("grey_zone"))
    qual = float(row.get("qual_score") or 0)
    tier = row.get("tier", "C")
    if "Kill" not in str(tier):
        tier = _tier_from_score(new_total, grey_zone=grey, total_qual=qual)

    golden = _truthy(row.get("golden_zone")) or (bonus == SCORE_VOLUME_PEARL)

    out = dict(row)
    out["volume_z"] = attn.get("volume_z") if attn else None
    out["volume_z_delta"] = attn.get("volume_z_delta") if attn else None
    out["volume_surge_z"] = attn.get("volume_surge_z") if attn else None
    out["attention_signal"] = signal
    out["volume_attention_score"] = bonus
    out["volume_source"] = attn.get("source") if attn else None
    if "Kill" not in str(row.get("tier", "")):
        out["quant_score"] = round(new_quant, 1)
        out["total_score"] = round(new_total, 1)
        out["tier"] = tier
    out["golden_zone"] = golden
    return out


def refresh_csv(
    csv_path: str,
    limit: int | None = None,
    fmp_first: bool = False,
    backup: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    rows = df.to_dict("records")
    if limit:
        rows = rows[:limit]

    logging.info(f"{csv_path}: refreshing volume attention for {len(rows)} rows (fmp_first={fmp_first})")
    updated = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_process_row, r, fmp_first): i for i, r in enumerate(rows)}
        results_by_idx = {}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            results_by_idx[idx] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(rows):
                logging.info(f"  progress {done}/{len(rows)}")
    updated = [results_by_idx[i] for i in range(len(rows))]

    if limit:
        rest = df.iloc[limit:].to_dict("records")
        updated.extend(rest)

    out_df = pd.DataFrame(updated)
    # 保留原欄位順序，新欄位接在後面
    new_cols = [c for c in out_df.columns if c not in df.columns]
    col_order = list(df.columns) + new_cols
    for c in col_order:
        if c not in out_df.columns:
            out_df[c] = None
    out_df = out_df[col_order]

    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = csv_path.replace(".csv", f".pre_volume_{ts}.csv")
        shutil.copy2(csv_path, bak)
        logging.info(f"Backup: {bak}")

    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return out_df


def _summarize(df: pd.DataFrame, label: str):
    print(f"\n=== {label} ===")
    print(f"Rows: {len(df)}")
    if "attention_signal" not in df.columns:
        print("No attention_signal column")
        return
    print(df["attention_signal"].value_counts(dropna=False).to_string())
    pearls = df[df["attention_signal"] == "潛在珍珠"]
    breaks = df[df["attention_signal"] == "共識轉折"]
    print(f"Pearls: {len(pearls)}, Breakouts: {len(breaks)}, bonus>0: {(df['volume_attention_score'].fillna(0) > 0).sum()}")
    hi = df[df["volume_attention_score"].fillna(0) > 0].sort_values("quant_score", ascending=False)
    if not hi.empty:
        cols = [c for c in ["ticker", "region", "tier", "quant_score", "total_score", "volume_z", "attention_signal"] if c in hi.columns]
        print(hi[cols].head(12).to_string(index=False))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Refresh volume attention columns in results CSV")
    parser.add_argument("--mode", choices=["compounder", "growth", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Only first N rows (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Process but do not write CSV")
    parser.add_argument("--fetch-fmp", action="store_true", help="Try FMP before yfinance")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    modes = ["compounder", "growth"] if args.mode == "both" else [args.mode]
    fmp_first = args.fetch_fmp

    for mode in modes:
        path = f"results_{mode}.csv"
        if args.dry_run:
            df = pd.read_csv(path, encoding="utf-8-sig")
            sample = df.head(args.limit or 5)
            for _, row in sample.iterrows():
                attn = fetch_volume_attention(str(row["ticker"]), fmp_first=fmp_first)
                old_b = float(row.get("volume_attention_score") or 0) if "volume_attention_score" in row else 0
                base_q = float(row["quant_score"]) - old_b
                bonus, sig = _volume_bonus(attn, base_q)
                print(f"{row['ticker']}: base_quant={base_q:.1f} signal={sig} bonus={bonus} attn={attn}")
            continue

        out = refresh_csv(path, limit=args.limit, fmp_first=fmp_first, backup=not args.no_backup)
        _summarize(out, path)


if __name__ == "__main__":
    main()
