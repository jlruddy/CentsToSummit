# Shadow replay scripts

Research scripts for replaying Gemini entries and testing exit rules and entry filters. They read
`data/kalshi_telemetry.db`; settlement results come from Kalshi's public REST market endpoint.
Exit replays use `orderbook_snapshots`, which are throttled to one per ticker per 10 seconds, so
they run slightly pessimistic on trailing exits (they miss short-lived peaks).

Run from the repo root:

```sh
DB=data/kalshi_telemetry.db
scripts/replay/fetch_settle.sh $DB > scripts/replay/settle.txt   # rerun after markets settle

python3 scripts/replay/replay.py   $DB scripts/replay/settle.txt   # round trips + entry features
python3 scripts/replay/grid.py     $DB scripts/replay/settle.txt   # exit rules x entry filters, net P&L
python3 scripts/replay/analysis.py $DB scripts/replay/settle.txt   # price move after entry; momentum tests
python3 scripts/replay/robust.py   $DB scripts/replay/settle.txt   # non-overlapping, tight-spread momentum test
```

Add a comma-separated list of session-id prefixes as a third argument to restrict to those
sessions, e.g. `... settle.txt d6e97bc4`. `analysis.py` sections B and C (all proposals and
the market-wide snapshot test) always use the whole database.

Baseline (2026-11-19, 38 Gemini entries, old exits stop 0.12 / trail 0.10/0.04): −$104 net.
