# CentsToSummit — Full Code Review

**Date:** 2026-11-17
**Scope:** All of `src/kalshi_poc/` (~5,800 lines), `tests/`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `README.md`
**Commit reviewed:** `f191691` plus uncommitted working-tree changes
**Test suite:** 45 / 45 passing (`uv run pytest`)

## Verdict

The overall design is sound: Decimal arithmetic everywhere, a pure `RiskGate`, a shadow simulator that walks the real book, signed REST and WebSocket requests, and atomic config writes.

It is **not ready for live money**, and the shadow results it produces right now are wrong. The problems fall into four groups:

1. Accounting bugs that corrupt the P&L figures.
2. Risk controls that can be bypassed, or that are steered by the AI.
3. An unauthenticated control panel.
4. Telemetry that leaves out or deletes the data the experiment needs.

Findings marked **✔ verified** were reproduced with a standalone script against the current code. Findings marked **⚠ verify with Kalshi** depend on Kalshi API behaviour that the test suite only mocks.

---

## Summary table

| ID | Severity | Area | Finding |
|----|----------|------|---------|
| C1 | 🔴 Critical | Simulator | Closing a NO/short position destroys $1 per contract ✔ |
| C2 | 🔴 Critical | Safety | Live trading can be turned on at runtime without `--confirm-live` |
| C3 | 🔴 Critical | Security | Control panel has no auth, no Origin/Host checks, and is open to cross-site requests |
| C4 | 🔴 Critical | Safety | Kill-switch abandons open positions; "Emergency Flatten" doesn't flatten |
| H1 | 🟠 High | AI/risk separation | The AI's `hold_to_expiry` flag loosens the gate's fee and edge thresholds |
| H2 | 🟠 High | Risk gate | Gate guesses whether the edge figure is per contract or per order ✔ |
| H3 | 🟠 High | Risk gate | "Intraday" decided from the series name's last letter; KXFED misclassified ✔ |
| H4 | 🟠 High | Risk gate | Dynamic sizing ignores `MAX_POSITION_DOLLARS_PER_MARKET` ✔ |
| H5 | 🟠 High | Portfolio | Equity valued at cost basis, never marked to market |
| H6 | 🟠 High | Safety | Daily drawdown baseline never rolls over; a restart resets it |
| H7 | 🟠 High | Market data | Order book can drift silently (no `seq` checks, fake depth, subscription handling) |
| H8 | 🟠 High | Execution | Live order handling makes up fills, guesses fee units, counts fills twice |
| H9 | 🟠 High | Execution | Reaper cancels every resting order in the account, including manual ones |
| M1 | 🟡 Medium | Research telemetry | Data is deleted, capped, rewritten, or never stored |
| M2 | 🟡 Medium | Mode isolation | Shadow cash is overwritten by the real account balance |
| M3 | 🟡 Medium | Strategy | Every config save rebuilds the strategy and drops its state |
| M4 | 🟡 Medium | Exchange API | Unrequested account changes; API contracts only tested against mocks |
| M5 | 🟡 Medium | Config | `.env` injection, silent fallback, Docker loses UI settings |
| M6 | 🟡 Medium | LLM | Retired default model; `max_tokens` too small; tight timeout |
| M7 | 🟡 Medium | Web UI | Unpinned CDN scripts with no integrity hashes |
| L1–L7 | 🟢 Low | Various | Duplicated logic, parsing edge cases, dead code, test warnings |

---

## 🔴 Critical

### C1. Closing a NO/short position in the shadow simulator destroys $1 per contract ✔ verified

**Where:** [`simulator.py:256`](src/kalshi_poc/simulator.py#L256), `ShadowSimulator._apply_fill_to_portfolio`

The `BID` branch assumes a BID always opens or adds to a long. When the position is short it subtracts `count × price` from cash, then deletes the position because `net_contracts` reaches zero. It never credits the value of the NO contracts being closed.

**Reproduction** (fees set to zero):

| Step | Cash | Expected |
|------|------|----------|
| Start | $1000.00 | $1000.00 |
| Open 10 NO (ASK at YES 0.40) | $994.00 | $994.00 |
| Close via BID 10 at YES 0.40 | **$990.00** | $1000.00 |

A long round trip at the same prices correctly returns to $1000.00.

**Impact:**
- Every short exit made by the stub strategy, the deterministic exits, or `_on_fill` books a phantom loss of `count × $1`.
- Those losses can trip the daily kill-switch on their own.
- The shadow equity curve is invalid for any session that traded NO.

**Tests:** [`tests/test_simulator.py`](tests/test_simulator.py) has no tests for the ASK/short path.

**Fix:** In the BID branch, when `pos.net_contracts < 0`, close `min(count, |net|)` first. Credit `close_qty × (1 − fill_price) − fee`, realize P&L against the NO cost basis, and only treat any remainder as a new long. Add long and short round-trip tests.

---

### C2. Live trading can be turned on at runtime without `--confirm-live`

**Where:** [`main.py:1476`](src/kalshi_poc/main.py#L1476), [`web_server.py:75`](src/kalshi_poc/web_server.py#L75)

The `--confirm-live` flag is checked only once, at startup in `run_orchestrator`. `POST /api/config` accepts `execution_mode=live` and `kalshi_environment=production` with no server-side check. `/api/engine/start` doesn't check it either. The confirmation dialog exists only in the browser ([`index.html:698`](src/kalshi_poc/static/index.html#L698)).

**Fix:** Enforce the live gate in `ConfigManager.update_and_persist` or in the engine's reload callback. Reject a switch to `live` unless the process was started with `--confirm-live`, or require a typed confirmation token in the request.

---

### C3. The control panel has no authentication and is exposed to cross-site requests

**Where:** [`web_server.py`](src/kalshi_poc/web_server.py) (all routes), [`web_server.py:251`](src/kalshi_poc/web_server.py#L251) (WebSocket)

Binding to `127.0.0.1` does not protect against any of the following, because they run through the user's own browser:

- **Body-less POSTs:** `/api/engine/start`, `/api/engine/kill`, `/api/engine/eval-now` and `/api/engine/reset-shadow` take no body. A plain HTML `<form method="POST">` on any website the user visits can trigger them. `start` resumes the last mode used, which could be `live`.
- **Telemetry WebSocket:** `/ws/telemetry` does not check the `Origin` header. Any web page can open `ws://127.0.0.1:8000/ws/telemetry` and read balances, positions and decisions.
- **DNS rebinding:** the `Host` header isn't validated, so a DNS-rebinding page gets full same-origin access. It could:
  - set `llm_endpoint` to an attacker's server, after which the engine sends the LLM API key (`x-api-key` header) there;
  - change the environment or execution mode, combining with C2.

**Fix:**
- Add a bearer token or session cookie on every route and on the WebSocket.
- Add `TrustedHostMiddleware` and validate `Origin` on the WebSocket and on state-changing routes.
- Make every state-changing POST require a JSON body so a plain form can't send it.

---

### C4. The kill-switch abandons open positions, and "Emergency Flatten" doesn't flatten

**Where:** [`main.py:1457`](src/kalshi_poc/main.py#L1457), [`main.py:828`](src/kalshi_poc/main.py#L828), [`risk_gate.py:111`](src/kalshi_poc/risk_gate.py#L111)

- `emergency_kill` only cancels resting orders and pauses. Despite the README and UI calling it "Emergency Flatten & Halt", it does not close any positions.
- Once the switch trips, the engine loop skips `_evaluate_deterministic_exits`, so stop-losses stop running.
- `RiskGate.evaluate` also blocks exits that would reduce risk while the switch is tripped, and when the per-minute order limit is hit.

The result is that open positions are left unmanaged at exactly the moment risk is highest.

**Fix:** Have Flatten submit closing orders for every open position. Let risk-reducing exits (`is_reducing_exit`) pass the kill-switch and the rate limit. Keep the deterministic exit loop running while the switch is tripped.

---

## 🟠 High

### H1. The AI controls some of the deterministic thresholds

**Where:** [`risk_gate.py:253`](src/kalshi_poc/risk_gate.py#L253)

```python
hold_to_expiry_eligible = intent.hold_to_expiry or is_intraday_binary
```

If the LLM sets `"hold_to_expiry": true`, the gate charges only the entry fee (not the round trip) and lowers the minimum edge to $0.015. This breaks the AI/risk separation the experiment depends on.

**Fix:** Decide hold-to-expiry eligibility only from market metadata, never from a field the AI supplies.

---

### H2. The gate guesses whether the AI's edge figure is per contract or for the whole order ✔ verified

**Where:** [`risk_gate.py:277`](src/kalshi_poc/risk_gate.py#L277), [`strategy_ai.py:884`](src/kalshi_poc/strategy_ai.py#L884)

If `count > 1` and `expected_edge_dollars ≤ 0.20`, the gate treats the number as per contract. The strategy parser applies the same heuristic with a different cutoff (`≤ 0.15`).

**Reproduction:** a total edge of $0.15 on 10 contracts ($0.015 per contract, below the ~$0.018 per-contract fee) was **approved** with no rejection reasons.

Also, in the `else` branch `min_net_edge_per_c = min_net_edge_dollars / count`, so larger orders face a looser bar than smaller ones.

**Fix:** Require `expected_edge_per_contract` on every intent and remove the magnitude heuristics. Apply `min_net_edge_dollars` as a per-contract value, consistently.

---

### H3. "Intraday" is decided from the last letter of the series name ✔ verified

**Where:** [`risk_gate.py:250`](src/kalshi_poc/risk_gate.py#L250)

```python
series_prefix.endswith("15M") or series_prefix.endswith("D") or series_prefix.endswith("U")
```

| Series | Classified intraday? | Correct? |
|--------|----------------------|----------|
| KXBTCD | Yes | Yes |
| KXINXU | Yes | Yes |
| **KXFED** | **Yes** | **No** |
| KXCPI | No | Yes |

KXFED is on the default allowlist and gets the loosened fee and edge rules from H1.

**Fix:** Classify using market metadata (close time minus open time, or an explicit series list).

---

### H4. Dynamic sizing silently ignores `MAX_POSITION_DOLLARS_PER_MARKET` ✔ verified

**Where:** [`risk_gate.py:67`](src/kalshi_poc/risk_gate.py#L67)

With defaults (equity $5000, floor $2000, `house_money_risk_pct` 0.20, `dynamic_sizing_enabled` true), the configured cap is **$250** but the effective cap is **$600**. When dynamic sizing is on, the static cap is never consulted.

**Fix:** `return min(s.max_position_dollars_per_market, dynamic_value)`.

---

### H5. Equity is valued at cost basis, never marked to market

**Where:** [`models.py:460`](src/kalshi_poc/models.py#L460)

`total_position_value_dollars` uses cost basis, so `current_equity_dollars` and `daily_pnl_dollars` only move when a position is closed or settled. The drawdown kill-switch and the equity floor ignore losses on open positions.

**Fix:** Mark positions to the live bid (long) or ask (short), and use that for the kill-switch and floor checks. Keep cost basis for collateral.

---

### H6. The "daily" drawdown baseline never resets each day, and a restart resets it

**Where:** [`main.py:376`](src/kalshi_poc/main.py#L376), [`main.py:486`](src/kalshi_poc/main.py#L486)

- `daily_start_equity_dollars` is set at the first sync and never rolls over at a day boundary.
- The kill-switch state isn't saved, and `start()` explicitly clears it.
- `docker-compose.yml` uses `restart: unless-stopped`, so every crash-restart resets the daily loss limit.

**Fix:** Save the kill-switch state and the daily baseline (e.g. in SQLite) and roll the baseline at a fixed UTC time. Require an explicit operator action to clear a tripped switch.

---

### H7. The order book can drift silently

**Where:** [`client.py:952`](src/kalshi_poc/client.py#L952), [`client.py:982`](src/kalshi_poc/client.py#L982), [`client.py:807`](src/kalshi_poc/client.py#L807), [`client.py:816`](src/kalshi_poc/client.py#L816)

- **No sequence checks:** `orderbook_delta` messages are applied without checking `seq`. A missed message corrupts the book that the risk gate prices against.
- **Fake depth:** `ticker` messages create books with a depth of 1 contract per level when no book exists yet. Later deltas build on top of those phantom levels.
- **Subscription handling** ⚠ verify with Kalshi:
  - `update_subscriptions` re-subscribes to the full ticker list each time.
  - `unsubscribe_tickers` assumes one `sid` per ticker.
  - If Kalshi issues one `sid` per channel, unsubscribing one ticker drops them all, and duplicate subscriptions would apply each delta twice.
- **No staleness check:** the risk gate doesn't check how old the snapshot is.

**Fix:** Track `seq` per subscription and re-snapshot on any gap. Never build depth from `ticker` messages. Use incremental subscription updates. Reject intents in the gate when the snapshot is older than a few seconds.

---

### H8. Live order handling makes up data

**Where:** [`main.py:1202`](src/kalshi_poc/main.py#L1202), [`main.py:1227`](src/kalshi_poc/main.py#L1227), [`main.py:646`](src/kalshi_poc/main.py#L646), [`main.py:652`](src/kalshi_poc/main.py#L652)

- **Assumed fills:** if the order response has no `fill_count` and the status isn't resting, the order is recorded as fully filled at the estimated price.
- **Fee units:** the fee is guessed with `if actual_fee > 1 and filled_qty <= 10: actual_fee /= 100`.
- **Fills counted twice:** a fill is applied from the REST response *and* again from the WebSocket `fill` event. If a sync is already running, `sync_balance_from_api` returns early, so the doubled state lasts until the next 15-second sync, and the gate sees it.
- **Wrong direction:** `_on_fill` maps `side` (`yes`/`no`) to direction and ignores `action` (`buy`/`sell`), so selling YES is recorded as a buy. It also always assumes the taker fee.

**Fix:** Apply each fill exactly once, keyed on `trade_id`, from one source. Read fee and fill units from the documented fields. Record "unknown" rather than inventing a fill.

---

### H9. The reaper cancels every resting order in the account

**Where:** [`main.py:225`](src/kalshi_poc/main.py#L225)

`_reap_stale_resting_orders` cancels any resting order older than 20 seconds, including orders the user placed by hand in the same account.

**Fix:** Only cancel orders whose `order_id` this process created, or whose `client_order_id` carries a recognisable prefix.

---

## 🟡 Medium

### M1. Research telemetry is deleted, capped, rewritten, or never stored

This is the experiment's main purpose, so these matter more than their severity label suggests.

- **Order-book history is deleted.** Only the newest 5,000 rows are kept ([`telemetry.py:353`](src/kalshi_poc/telemetry.py#L353)). At one row per ticker every 10 seconds across 6 tickers, that is roughly 2 hours of data.
- **The Parquet export is silently capped** at 50,000 rows per table ([`telemetry.py:540`](src/kalshi_poc/telemetry.py#L540)).
- **`ai_decisions` doesn't store what the AI saw and said** ([`telemetry.py:48`](src/kalshi_poc/telemetry.py#L48)). Missing: the raw prompt, the raw response, `hold_to_expiry`, `expected_edge_per_contract`, and the portfolio state shown to the model.
- **`session_id` exists on only 2 of the 5 tables** (`orders_and_fills`, `equity_curve`).
- **AI output is changed before it is logged.** `_parse_batch_intents` clamps prices, caps counts, normalizes edge, and drops proposals it considers invalid. `_evaluate_and_execute_intent` clamps the price again under the same `intent_id`. Telemetry stores the filtered version, not what the model proposed.
- **One bad item loses the whole batch, logged as HOLD.** A single malformed item (unknown `side`, `confidence > 1`) raises inside the parsing loop. Every intent in the batch is discarded, logged at debug level only, and recorded as **HOLD** ([`strategy_ai.py:907`](src/kalshi_poc/strategy_ai.py#L907)).
- **Other data is mixed in.** Deterministic exits and the stub strategy's exits go into `ai_decisions` with made-up "expected edge" values (`max(0.08 × count, …)`).
- **Shadow settlement is inferred, not real.** It uses `mid ≥ 0.50` ([`simulator.py:329`](src/kalshi_poc/simulator.py#L329)), falling back to the entry price when there's no book, instead of the market's actual `result`.

**Fix:** Store raw prompt and response text and every untouched AI proposal (including rejected and malformed ones, with the reason). Add `session_id` to all tables. Remove the pruning and export caps. Settle from the market's `result` field.

### M2. Shadow mode is not isolated from the real account

**Where:** [`main.py:288`](src/kalshi_poc/main.py#L288)

`sync_balance_from_api` overwrites `cash_balance_dollars` with the real account balance in every mode. At startup and on `reset_shadow`, shadow capital is set to the real demo balance rather than `initial_capital` whenever credentials are configured. WebSocket `fill` events also mutate the shadow portfolio.

### M3. Every config save rebuilds the strategy

**Where:** [`main.py:128`](src/kalshi_poc/main.py#L128)

`_on_config_reload` calls `StrategyFactory.create` on every save, including Start and Pause, which also save config. This wipes the mid-price history and LLM cooldown state, and creates a new single-flight lock, so two LLM calls can run at the same time.

### M4. Unrequested account changes and unverified API contracts

- `upgrade_api_tier_once` changes the account's API tier after the first fill without asking ([`client.py:589`](src/kalshi_poc/client.py#L589)).
- Shard auto-funding moves collateral between exchange shards on its own.
- ⚠ verify with Kalshi: the `/portfolio/events/orders` bid/ask schema, `intra_exchange_instance_transfer` amount units, and `/account/api_usage_level/upgrade` are only exercised against mocks. They need a smoke test against the demo exchange.

### M5. Config and environment handling

- **`.env` injection:** values are written to `.env` unquoted ([`config.py:237`](src/kalshi_poc/config.py#L237)). A newline in the API key ID or LLM endpoint injects extra variables.
- **Silent fallback:** if `config.json` fails validation, the app silently falls back to `.env` defaults ([`config.py:149`](src/kalshi_poc/config.py#L149)).
- **Docker loses settings:** `config.json` and the in-container `.env` aren't on a volume, so changes made in the UI disappear when the container is recreated.
- **No cross-field validation:** e.g. `equity_floor < initial_capital` isn't checked.

### M6. LLM provider settings

**Where:** [`strategy_ai.py:1061`](src/kalshi_poc/strategy_ai.py#L1061)

- The default model `claude-3-5-sonnet-latest` is retired. Use `claude-sonnet-5-5`, and update the default in `config.py` and `.env.example`.
- `max_tokens: 500` truncates multi-market JSON responses, which makes the batch parse fail and get recorded as HOLD (see M1).
- The 8-second timeout is tight for batched prompts.
- Per-token cost rates are hardcoded and will drift.

### M7. Unpinned CDN scripts with no integrity hashes

**Where:** [`index.html:7-8`](src/kalshi_poc/static/index.html#L7)

`alpinejs@3.x.x` (a floating version) and Tailwind's JIT CDN load without Subresource Integrity, inside a page that can POST trading config. Pin exact versions with SRI hashes, or vendor them locally.

---

## 🟢 Low

- **L1. Strategy-layer risk checks:** `_effective_max_pos` is duplicated between `BaseStrategy` and `RiskGate`, so they can drift. The parser also repeats checks the gate already makes (price band, correlation, pyramiding).
- **L2. Legacy cent prices:** in the old `yes`/`no` orderbook format, a price of `1` is read as $1.00 instead of 1¢ ([`models.py:254`](src/kalshi_poc/models.py#L254)).
- **L3. Retry-After:** a `Retry-After` header given as an HTTP date crashes `float()` ([`client.py:97`](src/kalshi_poc/client.py#L97)). Network errors on order POSTs aren't reconciled.
- **L4. Import side effects:** the module-level `app = create_app()` ([`web_server.py:272`](src/kalshi_poc/web_server.py#L272)) builds a second `ConfigManager` and `TelemetryManager` whenever the module is imported.
- **L5. Discovery ignores the allowlist:** discovery fallbacks (`KXSOLD`, `KXXRPD`, the demo "any market" fallback) don't check it, which wastes LLM calls. The gate still blocks the trades.
- **L6. Dead code:**
  - `ShadowSimulator.open_positions` and `realized_pnl` are never read in a way that matters.
  - `KalshiWebSocketClient._apply_delta` is unused.
  - `RiskGate.daily_loss` is computed but unused.
- **L7. Test and repo hygiene:**
  - un-awaited `AsyncMock` warnings in `tests/test_web_api.py`;
  - the pytest-asyncio `asyncio_default_fixture_loop_scope` deprecation warning;
  - `.DS_Store` files should be added to `.gitignore`.

---

## Recommended fix order

1. **C1:** fix the short-close accounting and add round-trip tests for both long and short positions.
2. **C2 / C3:**
   - require a server-side live confirmation (a CLI flag or a typed confirmation token);
   - add a bearer token or session cookie on every route and the WebSocket;
   - add Host and Origin allowlists;
   - make the body-less POSTs require JSON.
3. **C4 / H6:**
   - make Flatten actually close positions;
   - let risk-reducing exits through the kill-switch and the rate limit;
   - save the kill-switch state and the daily baseline, and roll the baseline at UTC midnight.
4. **H1–H4:** make the gate's units explicit (require `expected_edge_per_contract`), take intraday classification from market metadata, apply the static cap as the upper bound, and stop letting the AI choose `hold_to_expiry`.
5. **M1 (telemetry):** store the raw prompt and response and the AI's untouched proposals, add `session_id` everywhere, and remove the order-book pruning and export caps.
6. **H7 / H8:** add sequence-gap detection with automatic re-snapshot, process each fill once by `trade_id`, and run a smoke test against the demo exchange.
