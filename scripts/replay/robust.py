import sys, statistics as st, math
import replay as R
from collections import defaultdict
# tight-spread only (<=1c at both ends), non-overlapping 5-min windows per ticker, exclude last 3 min before close
for lb, thr in ((60, .06), (60, .10), (300, .10)):
    out = defaultdict(list)
    for tk, arr in R.snaps.items():
        close = R.settle[tk]["close"].timestamp()*1000 if tk in R.settle else None
        last_used = -1e18
        for i, (ts, b, a) in enumerate(arr):
            if a - b > 0.011 or ts - last_used < 300000: continue
            if close and close - ts < 480000: continue
            past = [x for x in arr[:i] if ts - lb*1000 - 15000 <= x[0] <= ts - lb*1000]
            fut = [x for x in arr[i:] if ts + 300000 <= x[0] <= ts + 315000]
            if not past or not fut or fut[0][2]-fut[0][1] > 0.011: continue
            m0=(b+a)/2; mp=(past[-1][1]+past[-1][2])/2; mf=(fut[0][1]+fut[0][2])/2
            if abs(m0-mp) < thr: continue
            x = (mf-m0) if m0>mp else (m0-mf)
            out[tk.split('-')[0]].append(x); out['ALL'].append(x); last_used = ts
    a = out['ALL']
    if len(a) > 1:
        se = st.stdev(a)/math.sqrt(len(a))
        print(f"lookback {lb}s |move|>={thr}: n={len(a)} mean={st.mean(a)*100:+.2f}c se={se*100:.2f}c cont={sum(x>0 for x in a)/len(a):.0%}  " + " ".join(f"{k}:{st.mean(v)*100:+.1f}c(n{len(v)})" for k,v in sorted(out.items()) if k!='ALL'))
