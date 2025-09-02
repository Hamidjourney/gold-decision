# scripts/update_goldhist.py
# Robust updater for Goldhist:
# - Appends only *new* trading days from Stooq (XAUUSD daily)
# - Drops today's partial bar (UTC)
# - Keeps a rolling window (e.g., last 5 years) with warmup days for indicators
# - Recomputes MA20, MA50, RSI14, TR, ATR14 and decisions
# - Writes data/Goldhist.csv and docs/latest.json

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Config ----------
REPO_ROOT     = Path(__file__).resolve().parents[1]
DATA_CSV      = REPO_ROOT / "data" / "Goldhist.csv"
LATEST_JSON   = REPO_ROOT / "docs" / "latest.json"
STOOQ_URL     = "https://stooq.com/q/d/l/?s=xauusd&i=d"

TRANCHE_USD   = 500
RULE_VERSION  = "v1.1"

# Control file size & perf:
KEEP_YEARS    = 5      # keep trailing N years in the saved CSV
WARMUP_DAYS   = 120    # include extra days before cutoff to stabilize indicators

# ---------- Utilities ----------
def prune_today_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any row dated 'today' (UTC) to avoid intraday/partial bars."""
    today_utc = pd.Timestamp(datetime.now(timezone.utc).date())
    return df[df.index < today_utc]

def limit_window(df: pd.DataFrame, keep_years: int, warmup_days: int) -> pd.DataFrame:
    """Limit to a trailing window (with warmup for indicators)."""
    if df.empty:
        return df
    last_date = df.index.max()
    cutoff = last_date - pd.DateOffset(years=keep_years)
    warmup_cut = cutoff - pd.Timedelta(days=warmup_days)
    return df[df.index >= warmup_cut]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns exist & are ordered."""
    cols = ["Open","High","Low","Close","MA20","MA50","RSI14","TR","ATR14","decision","rule","reason"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan if c not in ("decision","rule","reason") else ""
    return df[cols]

# ---------- Indicators ----------
def rsi_wilder(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    up    = delta.clip(lower=0.0)
    down  = (-delta).clip(lower=0.0)
    avg_gain = up.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    out = ohlc.copy()
    out["MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["MA50"] = out["Close"].rolling(50, min_periods=50).mean()

    hl   = out["High"] - out["Low"]
    h_pc = (out["High"] - out["Close"].shift(1)).abs()
    l_pc = (out["Low"]  - out["Close"].shift(1)).abs()
    out["TR"]    = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
    out["ATR14"] = out["TR"].ewm(alpha=1/14, adjust=False).mean()

    out["RSI14"] = rsi_wilder(out["Close"], 14)
    return out

# ---------- Decisions ----------
def decide_row(row: pd.Series) -> tuple[str, str, str]:
    close = row.get("Close"); ma20 = row.get("MA20"); ma50 = row.get("MA50")
    rsi   = row.get("RSI14"); atr  = row.get("ATR14")

    if pd.isna(close) or pd.isna(ma20) or pd.isna(ma50) or pd.isna(rsi) or pd.isna(atr):
        return ("WAIT", "Init", "Insufficient history for indicators")

    # Rule A: Repair buy — MA20 < MA50, Close <= MA50, RSI < 50
    if (ma20 < ma50) and (close <= ma50) and (rsi < 50):
        return (f"BUY 1 tranche (${TRANCHE_USD})", "A", "MA20<MA50 & Close<=MA50 & RSI<50")

    # Rule B: Uptrend dip buy — MA20 > MA50, Close <= MA20; stretched check
    if (ma20 > ma50) and (close <= ma20):
        if (ma20 - close) <= atr:
            return (f"BUY 1 tranche (${TRANCHE_USD})", "B", "Uptrend dip to/under MA20 (≤1×ATR)")
        else:
            return ("WAIT", "B*", "Dip >1×ATR below MA20 (stretched)")

    # Uptrend but above MA20 — wait for pullback
    if (ma20 > ma50) and (close > ma20):
        return ("WAIT", "B0", "Uptrend above MA20; wait for pullback")

    # Otherwise
    return ("WAIT", "None", "No rule matched")

# ---------- IO ----------
def load_goldhist(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["Open","High","Low","Close"])  # empty seed
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    # keep only columns we own; OHLC are authoritative
    df = df[["Open","High","Low","Close"]]
    return df

def fetch_stooq() -> pd.DataFrame:
    df = pd.read_csv(STOOQ_URL, sep=None, engine="python", parse_dates=["Date"], dayfirst=False)
    df = df.rename(columns=str.title)
    for c in ["Open","High","Low","Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Date").set_index("Date").dropna(subset=["Close"])
    df = prune_today_rows(df)  # drop today's partial bar
    return df[["Open","High","Low","Close"]]

def write_latest_json(df_full: pd.DataFrame) -> None:
    last = df_full.iloc[-1]
    regime = "Uptrend" if float(last["MA20"]) > float(last["MA50"]) else "Repair"
    payload = {
        "as_of_trading_day": df_full.index[-1].date().isoformat(),
        "last_updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
        "close": round(float(last["Close"]), 3),
        "ma20":  round(float(last["MA20"]), 4) if pd.notna(last["MA20"]) else None,
        "ma50":  round(float(last["MA50"]), 4) if pd.notna(last["MA50"]) else None,
        "atr14": round(float(last["ATR14"]), 4) if pd.notna(last["ATR14"]) else None,
        "rsi14": round(float(last["RSI14"]), 4) if pd.notna(last["RSI14"]) else None,
        "regime": regime,
        "yday_rule": str(last.get("rule", "")),
        "yday_decision": str(last.get("decision", "")),
        "rule_version": RULE_VERSION,
        "symbol": "XAUUSD",
        "units": "USD_per_oz"
    }
    LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ---------- Main ----------
def main() -> int:
    # 1) Load current saved OHLC (could be small/limited-window)
    gh_old_ohlc = load_goldhist(DATA_CSV)

    # 2) Fetch full Stooq OHLC (we'll only *use* rows after last saved date)
    try:
        src = fetch_stooq()
    except Exception as e:
        # If fetch fails but we have existing data, just recompute indicators/json on it
        if not gh_old_ohlc.empty:
            hist = add_indicators(gh_old_ohlc.copy())
            decisions = hist.apply(decide_row, axis=1, result_type="expand")
            decisions.columns = ["decision","rule","reason"]
            goldhist = ensure_columns(pd.concat([hist, decisions], axis=1))
            # Limit window even on fallback
            goldhist = limit_window(goldhist, KEEP_YEARS, WARMUP_DAYS)
            DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
            goldhist.reset_index().to_csv(DATA_CSV, index=False)
            write_latest_json(goldhist)
            print(f"[WARN] Stooq fetch failed: {e}. Wrote latest.json from existing data.")
            return 0
        raise

    # 3) Build merged OHLC — append only *new* rows
    if gh_old_ohlc.empty:
        merged_ohlc = src.copy()                   # first seed
    else:
        last_saved = gh_old_ohlc.index.max()
        new_rows = src.loc[src.index > last_saved] # ONLY dates after last saved
        if new_rows.empty:
            # No new trading day — just recompute indicators/json on current window
            hist = add_indicators(gh_old_ohlc.copy())
            decisions = hist.apply(decide_row, axis=1, result_type="expand")
            decisions.columns = ["decision","rule","reason"]
            goldhist = ensure_columns(pd.concat([hist, decisions], axis=1))
            goldhist = limit_window(goldhist, KEEP_YEARS, WARMUP_DAYS)
            DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
            goldhist.reset_index().to_csv(DATA_CSV, index=False)
            write_latest_json(goldhist)
            print(f"No new official OHLC after {last_saved.date()}. Goldhist unchanged (window maintained).")
            return 0
        merged_ohlc = pd.concat([gh_old_ohlc, new_rows], axis=0)
        merged_ohlc = merged_ohlc[~merged_ohlc.index.duplicated(keep="last")].sort_index()

    # Safety: drop today's partial if it slipped in (shouldn't, but double-guard)
    merged_ohlc = prune_today_rows(merged_ohlc)

    # 4) Recompute indicators & decisions on merged set
    hist = add_indicators(merged_ohlc)
    decisions = hist.apply(decide_row, axis=1, result_type="expand")
    decisions.columns = ["decision","rule","reason"]
    goldhist = ensure_columns(pd.concat([hist, decisions], axis=1))

    # 5) Limit saved window (with warmup), then write CSV + latest.json
    goldhist = limit_window(goldhist, KEEP_YEARS, WARMUP_DAYS)

    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    goldhist.reset_index().to_csv(DATA_CSV, index=False)

    write_latest_json(goldhist)

    print(f"Goldhist now up to {goldhist.index.max().date()} | Rows saved: {len(goldhist)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
