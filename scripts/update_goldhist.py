# scripts/update_goldhist.py
# Purpose:
# - Check if data/Goldhist.csv is current vs Stooq XAUUSD daily CSV.
# - If there are missing trading days, fetch them, recompute indicators & decisions,
#   save back to data/Goldhist.csv, and update docs/latest.json.
# - If nothing new, exit without touching files.

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

# ---- Config ----
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV  = REPO_ROOT / "data" / "Goldhist.csv"
LATEST_JSON = REPO_ROOT / "docs" / "latest.json"   # GitHub Pages uses /docs
STOOQ_URL = "https://stooq.com/q/d/l/?s=xauusd&i=d"

TRANCHE_USD = 500  # only used to label decision text
RULE_VERSION = "v1.1"

# ---- Indicator helpers ----
def rsi_wilder(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_gain = up.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    out = ohlc.copy()
    out["MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["MA50"] = out["Close"].rolling(50, min_periods=50).mean()

    hl   = out["High"] - out["Low"]
    h_pc = (out["High"] - out["Close"].shift(1)).abs()
    l_pc = (out["Low"]  - out["Close"].shift(1)).abs()
    out["TR"] = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
    out["ATR14"] = out["TR"].ewm(alpha=1/14, adjust=False).mean()

    out["RSI14"] = rsi_wilder(out["Close"], 14)
    return out

def decide_row(row: pd.Series) -> tuple[str, str, str]:
    """Your simple, robust rules."""
    close = row["Close"]; ma20 = row["MA20"]; ma50 = row["MA50"]; rsi = row["RSI14"]; atr = row["ATR14"]
    # Not enough data yet
    if pd.isna(ma20) or pd.isna(ma50) or pd.isna(rsi) or pd.isna(atr):
        return ("WAIT", "Init", "Insufficient history for indicators")

    # Rule A (repair): MA20 < MA50, Close <= MA50, RSI < 50
    if (ma20 < ma50) and (close <= ma50) and (rsi < 50):
        return (f"BUY 1 tranche (${TRANCHE_USD})", "A", "MA20<MA50 & Close<=MA50 & RSI<50")

    # Rule B (uptrend dip): MA20 > MA50, Close <= MA20; stretched check vs ATR
    if (ma20 > ma50) and (close <= ma20):
        if (ma20 - close) <= atr:
            return (f"BUY 1 tranche (${TRANCHE_USD})", "B", "Uptrend dip to/under MA20 (≤1×ATR)")
        else:
            return ("WAIT", "B*", "Dip >1×ATR below MA20 (stretched)")

    # Uptrend but above MA20 → wait for pullback
    if (ma20 > ma50) and (close > ma20):
        return ("WAIT", "B0", "Uptrend above MA20; wait for pullback")

    # Otherwise
    return ("WAIT", "None", "No rule matched")

# ---- Core updater ----
def load_goldhist(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        # Return empty frame with expected columns; we'll seed from Stooq later
        cols = ["Open","High","Low","Close","MA20","MA50","RSI14","TR","ATR14","decision","rule","reason"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    return df

def fetch_stooq() -> pd.DataFrame:
    df = pd.read_csv(STOOQ_URL, sep=None, engine="python", parse_dates=["Date"], dayfirst=False)
    df = df.rename(columns=str.title)
    for c in ["Open","High","Low","Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Date").set_index("Date").dropna(subset=["Close"])
    return df[["Open","High","Low","Close"]]

def write_latest_json(df_full: pd.DataFrame) -> None:
    last_row = df_full.iloc[-1]
    regime = "Uptrend" if last_row["MA20"] > last_row["MA50"] else "Repair"
    payload = {
        "as_of_trading_day": df_full.index[-1].date().isoformat(),
        "last_updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
        "close": round(float(last_row["Close"]), 3),
        "ma20": round(float(last_row["MA20"]), 4) if pd.notna(last_row["MA20"]) else None,
        "ma50": round(float(last_row["MA50"]), 4) if pd.notna(last_row["MA50"]) else None,
        "atr14": round(float(last_row["ATR14"]), 4) if pd.notna(last_row["ATR14"]) else None,
        "rsi14": round(float(last_row["RSI14"]), 4) if pd.notna(last_row["RSI14"]) else None,
        "regime": regime,
        "yday_rule": str(last_row.get("rule", "")),
        "yday_decision": str(last_row.get("decision", "")),
        "rule_version": RULE_VERSION,
        "symbol": "XAUUSD",
        "units": "USD_per_oz"
    }
    LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main() -> int:
    # 1) Load existing Goldhist (may be empty headers on first run)
    gh_old = load_goldhist(DATA_CSV)

    # 2) Fetch full Stooq series (official trading days only)
    src = fetch_stooq()

    # 3) Decide what to update
    if gh_old.empty:
        # First build from scratch using full Stooq data
        merged = src.copy()
    else:
        last_saved_date = gh_old.index.max()
        new_rows = src.loc[src.index > last_saved_date]
        if new_rows.empty:
            print(f"No new official OHLC after {last_saved_date.date()}. Goldhist unchanged.")
            return 0
        merged = pd.concat([gh_old[["Open","High","Low","Close"]], new_rows], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    # 4) Recompute indicators and decisions on the merged set
    hist = add_indicators(merged)
    decisions = hist.apply(decide_row, axis=1, result_type="expand")
    decisions.columns = ["decision","rule","reason"]
    goldhist = pd.concat([hist, decisions], axis=1)

    # 5) Write outputs
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    goldhist.reset_index().to_csv(DATA_CSV, index=False)

    write_latest_json(goldhist)

    print(f"Goldhist updated to {goldhist.index.max().date()}. Rows: {len(goldhist)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
