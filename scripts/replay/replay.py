"""Replay shadow entries under alternative exit rules / entry filters.

Usage: python replay.py <db> <settle.txt> [session_prefix,...]

<settle.txt> comes from fetch_settle.sh. The optional third argument keeps only trades
from sessions whose id starts with one of the comma-separated prefixes.
"""
import json, math, sqlite3, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timezone

DB, SETTLE = sys.argv[1], sys.argv[2]
SESSIONS = [x for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else []) if x]
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

settle = {}
for line in open(SETTLE):
    p = line.split()
    if len(p) >= 4:
        settle[p[0]] = {"status": p[1], "result": p[2] if p[1] == "finalized" else None,
                        "close": datetime.fromisoformat(p[-2 if p[1] == "finalized" else -1].replace("Z", "+00:00"))}

def fee_pc(p):  # Kalshi taker fee per contract (unrounded)
    return 0.07 * p * (1 - p)

fills = con.execute("""select f.*, d.decision_status, d.provider, d.llm_call_id, d.rationale, d.hold_to_expiry
  from orders_and_fills f left join ai_decisions d on d.intent_id=f.intent_id
  order by f.timestamp_ms, f.id""").fetchall()

# ---- rebuild round trips per (session,ticker) ----
pos = defaultdict(float)
open_tr = {}
trades = []
for f in fills:
    key = (f["session_id"], f["ticker"])
    c = float(f["count"]); px = float(f["fill_price"]); signed = c if f["side"] == "bid" else -c
    cur = pos[key]
    if cur == 0 or (cur > 0) == (signed > 0):
        if cur == 0:
            open_tr[key] = {"session": f["session_id"], "ticker": f["ticker"], "ts": f["timestamp_ms"],
                            "own": "yes" if signed > 0 else "no", "count": 0.0, "cost": 0.0, "fee": 0.0,
                            "provider": f["provider"], "llm_call_id": f["llm_call_id"],
                            "rationale": f["rationale"], "hold": f["hold_to_expiry"], "exits": [], "fill_id": f["id"]}
        t = open_tr[key]
        own_px = px if signed > 0 else 1 - px
        t["cost"] += own_px * c; t["count"] += c; t["fee"] += float(f["fee_dollars"])
    else:
        t = open_tr[key]
        own_px = px if t["own"] == "yes" else 1 - px
        t["exits"].append((f["timestamp_ms"], c, own_px, float(f["fee_dollars"]), f["rationale"]))
    pos[key] = cur + signed
    if abs(pos[key]) < 1e-9:
        pos[key] = 0
        trades.append(open_tr.pop(key))
for t in open_tr.values():
    t["orphan"] = True
    trades.append(t)
if SESSIONS:
    trades = [t for t in trades if any(t["session"].startswith(p) for p in SESSIONS)]

for t in trades:
    t["entry"] = t["cost"] / t["count"]
    t["entry_fee_pc"] = t["fee"] / t["count"]
    ex = t["exits"]
    if ex and not t.get("orphan"):
        t["actual_pnl"] = sum(c * (p - t["entry"]) - fe for _, c, p, fe, _ in ex) - t["fee"]
        t["actual_reason"] = (ex[-1][4] or "").split(":")[0].replace("Deterministic ", "")
    else:
        t["actual_pnl"] = None; t["actual_reason"] = "ORPHAN(session ended)"
    s = settle.get(t["ticker"])
    t["close"] = s["close"]
    t["tte_s"] = (s["close"].timestamp() * 1000 - t["ts"]) / 1000
    t["result"] = s["result"]
    t["series"] = t["ticker"].split("-")[0]

# ---- features from the LLM prompt that produced the entry ----
for t in trades:
    t["feat"] = {}
    if t["llm_call_id"]:
        r = con.execute("select user_prompt from llm_calls where llm_call_id=?", (t["llm_call_id"],)).fetchone()
        if r:
            js = json.loads(r[0][r[0].index("{"):])
            for m in js["markets"]:
                if m["ticker"] == t["ticker"]:
                    t["feat"] = {k: float(m[k]) for k in ("mid_delta_15s", "mid_delta_1m", "mid_delta_5m", "imbalance", "spread", "mid_price")}

# ---- price paths: own-side bid from snapshots ----
snaps = defaultdict(list)
for r in con.execute("select ticker,timestamp_ms,best_yes_bid,best_yes_ask from orderbook_snapshots where best_yes_bid is not null and best_yes_ask is not null order by timestamp_ms"):
    snaps[r[0]].append((r[1], float(r[2]), float(r[3])))

def path(t):
    out = []
    for ts, b, a in snaps[t["ticker"]]:
        if ts <= t["ts"] or ts >= t["close"].timestamp() * 1000:
            continue
        out.append((ts, b if t["own"] == "yes" else 1 - a, (a + b) / 2 if t["own"] == "yes" else 1 - (a + b) / 2))
    return out

def settle_val(t):
    if t["result"] is None:
        return None
    return 1.0 if t["result"] == t["own"] else 0.0

def sim(t, stop=None, trail=None, tp=True, stop_disable=120, fallback="settle"):
    """Return (pnl_dollars, reason, hold_s). trail=(activation,distance)."""
    e = t["entry"]; peak = e
    for ts, bid, _ in path(t):
        close_in = t["close"].timestamp() - ts / 1000
        peak = max(peak, bid)
        reason = None
        if tp and ((bid - e >= 0.14 and bid >= 0.75) or bid >= 0.88):
            reason = "TP"
        elif trail and peak - e >= trail[0] and peak - bid >= trail[1]:
            reason = "TRAIL"
        elif stop is not None and e - bid >= stop and close_in > stop_disable:
            reason = "STOP"
        if reason:
            return t["count"] * (bid - e - fee_pc(bid)) - t["fee"], reason, (ts - t["ts"]) / 1000
    v = settle_val(t)
    if v is None:
        return None, "PENDING", None
    return t["count"] * (v - e) - t["fee"], "SETTLE", t["tte_s"]

def move_after(t, secs):
    p = path(t)
    tgt = t["ts"] + secs * 1000
    for ts, bid, mid in p:
        if ts >= tgt:
            if ts - tgt > 15000:
                return None
            return mid - t["entry"]
    return None

if __name__ == "__main__":
    gem = [t for t in trades if t["provider"] == "gemini"]
    print(f"trades total={len(trades)} gemini={len(gem)}")
    print(f"{'fill':>4} {'time':8} {'ticker':32} {'own':3} {'n':>3} {'entry':>6} {'fee/c':>5} {'tte_m':>5} {'res':>4} {'actual':>7} {'reason':16} d1m    d5m    imb")
    for t in gem:
        f = t["feat"]
        tm = datetime.fromtimestamp(t["ts"] / 1000).strftime("%H:%M:%S")
        ap = f"{t['actual_pnl']:+.2f}" if t["actual_pnl"] is not None else "   -"
        print(f"{t['fill_id']:>4} {tm} {t['ticker']:32} {t['own']:3} {t['count']:>3.0f} {t['entry']:.3f} {t['entry_fee_pc']:.3f} {t['tte_s']/60:5.1f} {str(t['result']):>4} {ap:>7} {t['actual_reason'][:16]:16} {f.get('mid_delta_1m',0):+.3f} {f.get('mid_delta_5m',0):+.3f} {f.get('imbalance',0):+.2f}")
