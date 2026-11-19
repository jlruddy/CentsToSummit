import sys, json, statistics as st
from collections import defaultdict
import replay as R

def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n=0"
    hit = sum(1 for x in xs if x > 0) / len(xs)
    return f"n={len(xs):3d} mean={st.mean(xs)*100:+6.2f}c med={st.median(xs)*100:+6.2f}c win={hit:4.0%}"

gem = [t for t in R.trades if t["provider"] == "gemini"]
H = [30, 60, 120, 300]

print("=== A. Executed Gemini entries: own-side mid move after entry (vs fill price; includes ~half-spread+slip cost) ===")
for h in H:
    print(f"  +{h:>3}s  {fmt([R.move_after(t, h) for t in gem])}")

def bucket(name, keyf):
    print(f"\n  by {name}:")
    groups = defaultdict(list)
    for t in gem:
        groups[keyf(t)].append(t)
    for k in sorted(groups, key=str):
        ts = groups[k]
        print(f"    {str(k):22} " + " | ".join(f"{h}s {fmt([R.move_after(t,h) for t in ts])}" for h in (60, 300)))

bucket("series", lambda t: t["series"])
bucket("own price", lambda t: "<0.50" if t["entry"] < .5 else "0.50-0.60" if t["entry"] < .6 else ">=0.60")
bucket("tte", lambda t: "<15m" if t["tte_s"] < 900 else "15-45m" if t["tte_s"] < 2700 else ">45m")
bucket("|d5m| own-dir", lambda t: "none" if not t["feat"] else ("|d5m|>=.10" if abs(t["feat"]["mid_delta_5m"]) >= .10 else "|d5m|<.10"))
bucket("imb aligned", lambda t: "none" if not t["feat"] else ("aligned" if (t["feat"]["imbalance"] > 0) == (t["own"] == "yes") and abs(t["feat"]["imbalance"]) > .2 else "not"))

# ---------- B. All model proposals (PROPOSED + GATED + FILTERED), executed or not ----------
print("\n=== B. All Gemini proposals (executed or rejected): YES-mid move in proposed direction ===")
con = R.con
props = con.execute("""select timestamp_ms, ticker, side, decision_status, llm_call_id from ai_decisions
  where provider='gemini' and decision_status in ('PROPOSED','GATED','FILTERED')""").fetchall()
mids = defaultdict(list)
for tk, v in R.snaps.items():
    mids[tk] = [(ts, (b + a) / 2) for ts, b, a in v]

def fwd(tk, ts, h):
    m0 = m1 = None
    for t2, m in mids[tk]:
        if t2 <= ts:
            m0 = (t2, m)
        elif t2 >= ts + h * 1000:
            if t2 - (ts + h * 1000) <= 15000:
                m1 = m
            break
    if m0 is None or ts - m0[0] > 15000 or m1 is None:
        return None
    return m1 - m0[1]

seen = set(); rows = []
for ts, tk, side, status, cid in props:
    k = (ts // 30000, tk, side)  # dedupe repeated proposals within 30s
    if k in seen:
        continue
    seen.add(k)
    sgn = 1 if side == "bid" else -1
    rows.append({"status": status, "tk": tk, "series": tk.split("-")[0],
                 **{h: (None if fwd(tk, ts, h) is None else sgn * fwd(tk, ts, h)) for h in H}})
for h in H:
    print(f"  +{h:>3}s all      {fmt([r[h] for r in rows])}")
for s in ("PROPOSED", "GATED", "FILTERED"):
    print(f"  +300s {s:9} {fmt([r[300] for r in rows if r['status']==s])}")
for s in sorted({r["series"] for r in rows}):
    print(f"  +60s/+300s {s:9} {fmt([r[60] for r in rows if r['series']==s])} || {fmt([r[300] for r in rows if r['series']==s])}")

# ---------- C. Market-wide: does 1m / 5m momentum continue? (every snapshot) ----------
print("\n=== C. Every snapshot, all tickers: future mid move signed by trailing momentum (mid-to-mid, no costs) ===")
def back(tk, i, secs):
    arr = mids[tk]; ts0, m0 = arr[i]
    for j in range(i, -1, -1):
        if arr[j][0] <= ts0 - secs * 1000:
            return m0 - arr[j][1] if ts0 - secs * 1000 - arr[j][0] <= 15000 else None
    return None
def fwd_i(tk, i, secs):
    arr = mids[tk]; ts0, m0 = arr[i]
    for j in range(i, len(arr)):
        if arr[j][0] >= ts0 + secs * 1000:
            return arr[j][1] - m0 if arr[j][0] - ts0 - secs * 1000 <= 15000 else None
    return None
for lb in (60, 300):
    for thr in (0.03, 0.06, 0.10):
        for fh in (60, 300):
            res = defaultdict(list)
            for tk, arr in mids.items():
                ser = tk.split("-")[0]
                for i in range(0, len(arr), 3):  # ~30s stride to reduce overlap
                    b = back(tk, i, lb)
                    if b is None or abs(b) < thr:
                        continue
                    f = fwd_i(tk, i, fh)
                    if f is None:
                        continue
                    x = f if b > 0 else -f
                    res["ALL"].append(x); res[ser].append(x)
            line = " ".join(f"{k}:{fmt(v)}" for k, v in sorted(res.items()) if k == "ALL")
            per = "  ".join(f"{k} {st.mean(v)*100:+.1f}c(n{len(v)})" for k, v in sorted(res.items()) if k != "ALL" and v)
            print(f"  lookback {lb:3}s |move|>={thr:.2f} -> fwd {fh:3}s  {line}\n        {per}")
