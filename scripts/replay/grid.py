import sys
import replay as R

gem = [t for t in R.trades if t["provider"] == "gemini"]
# exclude positions still open in the running session (not orphaned by a restart)
if not R.trades:
    sys.exit("no trades in the selected sessions")
live_sess = max(R.trades, key=lambda t: t["ts"])["session"]
gem = [t for t in gem if not (t.get("orphan") and t["session"] == live_sess)]

print("=== fidelity: replay(old rules: stop .12, trail .10/.04, TP) vs actual ===")
for t in gem:
    if t["actual_pnl"] is None:
        continue
    p, why, hs = R.sim(t, stop=0.12, trail=(0.10, 0.04))
    print(f"  fill {t['fill_id']:>3} {t['ticker'][:26]:26} actual {t['actual_pnl']:+6.2f} {t['actual_reason'][:10]:10} replay {('%+6.2f' % p) if p is not None else '  pend'} {why}")
act = [t["actual_pnl"] for t in gem if t["actual_pnl"] is not None]
print(f"  actual total (closed, excl. orphans) = {sum(act):+.2f} over {len(act)}")

EXITS = {
    "stop.12 trail.10/.04 (old)": dict(stop=0.12, trail=(0.10, 0.04)),
    "stop.22 trail.15/.06 (new)": dict(stop=0.22, trail=(0.15, 0.06)),
    "stop.18 trail.10/.04": dict(stop=0.18, trail=(0.10, 0.04)),
    "stop.22 trail.10/.04": dict(stop=0.22, trail=(0.10, 0.04)),
    "stop.12 no-trail": dict(stop=0.12, trail=None),
    "stop.18 no-trail": dict(stop=0.18, trail=None),
    "stop.22 no-trail": dict(stop=0.22, trail=None),
    "no stop, trail.10/.04": dict(stop=None, trail=(0.10, 0.04)),
    "no stop, trail.15/.06": dict(stop=None, trail=(0.15, 0.06)),
    "stop.18 trail.15/.06": dict(stop=0.18, trail=(0.15, 0.06)),
    "stop.12 trail.06/.03": dict(stop=0.12, trail=(0.06, 0.03)),
    "no stop, TP only": dict(stop=None, trail=None),
    "hold to settle (no TP)": dict(stop=None, trail=None, tp=False),
}
FILTERS = {
    "all": lambda t: True,
    "own>=0.60": lambda t: t["entry"] >= 0.60,
    "own<0.60": lambda t: t["entry"] < 0.60,
    "tte>=10m": lambda t: t["tte_s"] >= 600,
    "tte<=30m": lambda t: t["tte_s"] <= 1800,
    "no KXINXU": lambda t: t["series"] != "KXINXU",
    "15M only": lambda t: "15M" in t["series"],
    "hourly crypto only": lambda t: t["series"] in ("KXBTCD", "KXETHD"),
    "|d1m|<0.10 (no chase)": lambda t: bool(t["feat"]) and abs(t["feat"]["mid_delta_1m"]) < 0.10,
    "antichase d1m_own<.06": lambda t: bool(t["feat"]) and (t["feat"]["mid_delta_1m"] * (1 if t["own"] == "yes" else -1)) < 0.06,
    "antichase+tte>=10m": lambda t: bool(t["feat"]) and (t["feat"]["mid_delta_1m"] * (1 if t["own"] == "yes" else -1)) < 0.06 and t["tte_s"] >= 600,
}
pend = [t for t in gem if R.settle_val(t) is None]
print(f"\n{len(gem)} entries; {len(pend)} still unsettled (results using settlement show PENDING count)")
hdr = f"{'exit rule':32}" + "".join(f"{k:>22}" for k in FILTERS)
print(hdr)
for name, kw in EXITS.items():
    row = f"{name:32}"
    for fk, ff in FILTERS.items():
        sub = [t for t in gem if ff(t)]
        res = [R.sim(t, **kw) for t in sub]
        done = [r for r in res if r[0] is not None]
        tot = sum(r[0] for r in done); w = sum(1 for r in done if r[0] > 0)
        npend = len(res) - len(done)
        cell = f"{tot:+7.2f} {w}/{len(done)}" + (f" p{npend}" if npend else "")
        row += f"{cell:>22}"
    print(row)
